#include "targets/gemma4_31b_it/impl/runtime/reference_model.h"

#include <ninfer/ops/embedding.h>
#include <ninfer/ops/linear.h>
#include <ninfer/ops/residual_add.h>
#include <ninfer/ops/rmsnorm.h>
#include <ninfer/targets/gemma4_31b_it/config.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "core/arena.h"
#include "core/device.h"
#include "targets/gemma4_31b_it/impl/load/bindings.h"
#include "targets/gemma4_31b_it/impl/runtime/reference_kernels.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::gemma4_31b_it::detail {
namespace {

using artifact::NumericFormat;

struct AttentionWeights {
    Weight input;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
};

struct LayerWeights {
    Tensor input_norm;
    AttentionWeights attention;
    Tensor post_attention_norm;
    Tensor pre_feedforward_norm;
    Weight gate_up;
    Weight down;
    Tensor post_feedforward_norm;
    Tensor layer_scalar;
};

struct ModelWeights {
    Weight embedding;
    std::array<LayerWeights, TextConfig::layers> layers;
    Tensor final_norm;
};

ModelWeights load_weights(const artifact::MaterializedArtifact& artifact,
                          const BindingPlan& plan) {
    ModelWeights out;
    out.embedding = artifact::materialized_weight(
        artifact, plan.token_embedding, NumericFormat::FP8_E4M3FN_ROW_BF16S,
        TextConfig::vocabulary, TextConfig::hidden);
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const bool full = TextConfig::is_full_attention(layer);
        const auto& source = plan.layers[layer];
        auto& target = out.layers[layer];
        const int head_dim = full ? TextConfig::full_head_dim : TextConfig::sliding_head_dim;
        const int input_rows = full ? 18432 : 16384;
        const int output_columns = full ? 16384 : 8192;
        target.input_norm = artifact::materialized_tensor(
            artifact, source.input_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.attention.input = artifact::materialized_weight(
            artifact, source.attention.input_projection, NumericFormat::Q4G64_F16S,
            input_rows, TextConfig::hidden);
        target.attention.query_norm = artifact::materialized_tensor(
            artifact, source.attention.query_norm, NumericFormat::BF16, {head_dim});
        target.attention.key_norm = artifact::materialized_tensor(
            artifact, source.attention.key_norm, NumericFormat::BF16, {head_dim});
        target.attention.output = artifact::materialized_weight(
            artifact, source.attention.output, NumericFormat::Q4G64_F16S,
            TextConfig::hidden, output_columns);
        target.post_attention_norm = artifact::materialized_tensor(
            artifact, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.pre_feedforward_norm = artifact::materialized_tensor(
            artifact, source.pre_feedforward_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.gate_up = artifact::materialized_weight(
            artifact, source.mlp.gate_up, NumericFormat::Q4G64_F16S,
            2 * TextConfig::intermediate, TextConfig::hidden);
        target.down = artifact::materialized_weight(
            artifact, source.mlp.down, NumericFormat::Q4G64_F16S,
            TextConfig::hidden, TextConfig::intermediate);
        target.post_feedforward_norm = artifact::materialized_tensor(
            artifact, source.post_feedforward_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.layer_scalar = artifact::materialized_tensor(
            artifact, source.layer_scalar, NumericFormat::BF16, {1});
    }
    out.final_norm = artifact::materialized_tensor(
        artifact, plan.final_norm, NumericFormat::BF16, {TextConfig::hidden});
    return out;
}

std::size_t checked_workspace_bytes(std::int32_t tokens) {
    // All Phase 4 activations are BF16 except the final vocabulary logits. This deliberately
    // over-reserves the maximum local/global views so every layer executes allocation-free.
    const std::uint64_t bf16_elements =
        static_cast<std::uint64_t>(tokens) *
        (8ULL * TextConfig::hidden + 2ULL * TextConfig::intermediate + 43008ULL +
         4ULL * TextConfig::query_heads * TextConfig::full_head_dim +
         2ULL * TextConfig::sliding_kv_heads * TextConfig::full_head_dim);
    const std::uint64_t bytes = bf16_elements * 2ULL +
                                static_cast<std::uint64_t>(TextConfig::vocabulary) * 4ULL +
                                static_cast<std::uint64_t>(tokens) * 4ULL + 16ULL * 1024 * 1024;
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Gemma reference workspace size overflow");
    }
    return static_cast<std::size_t>(bytes);
}

void write_bf16(const std::filesystem::path& directory, const std::string& name,
                const Tensor& tensor) {
    if (directory.empty()) { return; }
    std::vector<std::uint16_t> host(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    std::ofstream output(directory / (name + ".bf16"), std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(host.data()),
                 static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
    if (!output) { throw std::runtime_error("failed to write Gemma parity dump " + name); }
}

void write_fp32(const std::filesystem::path& directory, const std::string& name,
                const Tensor& tensor) {
    if (directory.empty()) { return; }
    std::vector<float> host(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::ofstream output(directory / (name + ".f32"), std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(host.data()),
                 static_cast<std::streamsize>(host.size() * sizeof(float)));
    if (!output) { throw std::runtime_error("failed to write Gemma parity dump " + name); }
}

bool detailed_layer(std::size_t layer) { return layer == 0 || layer == 5; }

} // namespace

ReferenceRunResult run_reference_prefix(const std::filesystem::path& artifact_path,
                                        std::span<const std::int32_t> token_ids,
                                        const std::filesystem::path& dump_directory,
                                        int device_id) {
    if (token_ids.empty() || token_ids.size() > 4096) {
        throw std::invalid_argument("Gemma Phase 4 reference prefix must contain 1..4096 tokens");
    }
    for (const std::int32_t id : token_ids) {
        if (id < 0 || id >= static_cast<std::int32_t>(TextConfig::vocabulary)) {
            throw std::invalid_argument("Gemma reference prefix contains an invalid token id");
        }
    }
    if (!dump_directory.empty()) { std::filesystem::create_directories(dump_directory); }

    DeviceContext device(device_id);
    artifact::Reader reader(artifact_path);
    artifact::Binder binder(reader);
    ArtifactLoadPlan plan = bind_artifact(binder);
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, plan.materialization, device);
    const ModelWeights weights = load_weights(materialized, plan.bindings);

    const std::int32_t tokens = static_cast<std::int32_t>(token_ids.size());
    WorkspaceArena arena(checked_workspace_bytes(tokens));
    Tensor ids = arena.alloc(DType::I32, {tokens});
    CUDA_CHECK(cudaMemcpyAsync(ids.data, token_ids.data(), token_ids.size_bytes(),
                               cudaMemcpyHostToDevice, device.stream));
    Tensor hidden = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor normalized = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor packed_storage = arena.alloc(DType::BF16, {43008, tokens});
    Tensor query_storage = arena.alloc(
        DType::BF16, {TextConfig::full_head_dim, TextConfig::query_heads, tokens});
    Tensor key_storage = arena.alloc(
        DType::BF16, {TextConfig::full_head_dim, TextConfig::sliding_kv_heads, tokens});
    Tensor value_storage = arena.alloc(
        DType::BF16, {TextConfig::full_head_dim, TextConfig::sliding_kv_heads, tokens});
    Tensor attended_storage = arena.alloc(
        DType::BF16, {TextConfig::full_head_dim, TextConfig::query_heads, tokens});
    Tensor projected = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor post = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor product = arena.alloc(DType::BF16, {TextConfig::intermediate, tokens});
    Tensor feedforward = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor final_hidden = arena.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor logits = arena.alloc(DType::FP32, {TextConfig::vocabulary});

    ops::embedding(ids, weights.embedding, hidden, device.stream);
    scale_embedding(hidden, TextConfig::embedding_scale_bf16, device.stream);
    write_bf16(dump_directory, "target_embeddings", hidden);

    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const bool full = TextConfig::is_full_attention(layer);
        const auto& layer_weights = weights.layers[layer];
        const int head_dim = full ? TextConfig::full_head_dim : TextConfig::sliding_head_dim;
        const int kv_heads = full ? TextConfig::full_kv_heads : TextConfig::sliding_kv_heads;
        const int q_rows = TextConfig::query_heads * head_dim;
        const int input_rows = full ? 18432 : 16384;
        Tensor packed(packed_storage.data, DType::BF16, {input_rows, tokens});
        Tensor query(query_storage.data, DType::BF16,
                     {head_dim, static_cast<int>(TextConfig::query_heads), tokens});
        Tensor key(key_storage.data, DType::BF16, {head_dim, kv_heads, tokens});
        Tensor value(value_storage.data, DType::BF16, {head_dim, kv_heads, tokens});
        Tensor attended(attended_storage.data, DType::BF16,
                        {head_dim, static_cast<int>(TextConfig::query_heads), tokens});

        ops::rmsnorm(hidden, layer_weights.input_norm, TextConfig::rms_epsilon, false,
                     normalized, device.stream);
        if (detailed_layer(layer)) {
            write_bf16(dump_directory, "target_layer" + std::to_string(layer) +
                                           "_input_normalized", normalized);
        }
        ops::linear(normalized, layer_weights.attention.input, packed, device.stream);
        prepare_qkv(packed, layer_weights.attention.query_norm,
                    layer_weights.attention.key_norm, head_dim, kv_heads,
                    full ? TextConfig::full_rope_theta : TextConfig::sliding_rope_theta,
                    full ? TextConfig::full_rotary_active_dim / 2 : head_dim / 2,
                    query, key, value, device.stream);
        if (detailed_layer(layer)) {
            const std::string prefix = "target_layer" + std::to_string(layer);
            write_bf16(dump_directory, prefix + "_attention_input_packed", packed);
            write_bf16(dump_directory, prefix + "_q", query);
            write_bf16(dump_directory, prefix + "_k", key);
            write_bf16(dump_directory, prefix + "_v", value);
        }
        reference_attention(query, key, value, full ? 0 : TextConfig::sliding_window,
                            attended, device.stream);
        if (detailed_layer(layer)) {
            write_bf16(dump_directory, "target_layer" + std::to_string(layer) +
                                           "_attention_output", attended);
        }
        ops::linear(attended.view({q_rows, tokens}), layer_weights.attention.output,
                    projected, device.stream);
        if (detailed_layer(layer)) {
            write_bf16(dump_directory, "target_layer" + std::to_string(layer) +
                                           "_attention_projection", projected);
        }
        ops::rmsnorm(projected, layer_weights.post_attention_norm,
                     TextConfig::rms_epsilon, false, post, device.stream);
        ops::residual_add(post, hidden, device.stream);
        if (detailed_layer(layer)) {
            write_bf16(dump_directory, "target_layer" + std::to_string(layer) +
                                           "_attention_residual", hidden);
        }
        ops::rmsnorm(hidden, layer_weights.pre_feedforward_norm,
                     TextConfig::rms_epsilon, false, normalized, device.stream);
        if (detailed_layer(layer)) {
            write_bf16(dump_directory, "target_layer" + std::to_string(layer) +
                                           "_pre_feedforward", normalized);
        }
        Tensor gate_up(packed_storage.data, DType::BF16,
                       {static_cast<int>(2 * TextConfig::intermediate), tokens});
        ops::linear(normalized, layer_weights.gate_up, gate_up, device.stream);
        if (detailed_layer(layer)) {
            write_bf16(dump_directory, "target_layer" + std::to_string(layer) +
                                           "_mlp_gate_up", gate_up);
        }
        gelu_tanh_mul_packed(gate_up, product, device.stream);
        if (detailed_layer(layer)) {
            write_bf16(dump_directory, "target_layer" + std::to_string(layer) +
                                           "_mlp_product", product);
        }
        ops::linear(product, layer_weights.down, feedforward, device.stream);
        if (detailed_layer(layer)) {
            write_bf16(dump_directory, "target_layer" + std::to_string(layer) +
                                           "_mlp_output", feedforward);
        }
        ops::rmsnorm(feedforward, layer_weights.post_feedforward_norm,
                     TextConfig::rms_epsilon, false, post, device.stream);
        add_scaled(post, layer_weights.layer_scalar, hidden, device.stream);
        write_bf16(dump_directory, "target_layer" + std::to_string(layer) + "_output", hidden);
    }

    ops::rmsnorm(hidden, weights.final_norm, TextConfig::rms_epsilon, false,
                 final_hidden, device.stream);
    write_bf16(dump_directory, "target_final_hidden", final_hidden);
    Tensor last_hidden = final_hidden.slice(1, tokens - 1, 1).view({TextConfig::hidden});
    fp8_tied_logits(last_hidden, weights.embedding, TextConfig::final_logit_softcap,
                    logits, device.stream);
    device.synchronize();
    write_fp32(dump_directory, "target_last_logits", logits);

    ReferenceRunResult result;
    result.logits.resize(TextConfig::vocabulary);
    CUDA_CHECK(cudaMemcpy(result.logits.data(), logits.data,
                          result.logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    result.greedy_token = static_cast<std::int32_t>(
        std::max_element(result.logits.begin(), result.logits.end()) - result.logits.begin());
    return result;
}

} // namespace ninfer::targets::gemma4_31b_it::detail
