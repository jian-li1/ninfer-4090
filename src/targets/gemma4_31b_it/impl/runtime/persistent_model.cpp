#include "targets/gemma4_31b_it/impl/runtime/persistent_model.h"

#include <ninfer/ops/embedding.h>
#include <ninfer/ops/linear.h>
#include <ninfer/ops/residual_add.h>
#include <ninfer/ops/rmsnorm.h>
#include <ninfer/ops/softmax_attention.h>
#include <ninfer/targets/gemma4_31b_it/config.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/heterogeneous_kv_cache.h"
#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "targets/gemma4_31b_it/impl/load/bindings.h"
#include "targets/gemma4_31b_it/impl/runtime/kv_groups.h"
#include "targets/gemma4_31b_it/impl/runtime/model_weights.h"
#include "targets/gemma4_31b_it/impl/runtime/reference_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::gemma4_31b_it::detail {
namespace {

constexpr std::size_t kActivationWorkspaceBytes = 32ULL * 1024 * 1024;
constexpr ops::AttentionHeadGeometry kSlidingGeometry{
    TextConfig::sliding_head_dim, TextConfig::query_heads, TextConfig::sliding_kv_heads};
constexpr ops::AttentionHeadGeometry kFullGeometry{
    TextConfig::full_head_dim, TextConfig::query_heads, TextConfig::full_kv_heads};

std::size_t free_device_bytes() {
    std::size_t free = 0;
    std::size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

PagedKVLayerView layer_cache(HeterogeneousKVTransaction& transaction,
                             TextKvLayerAddress address, std::int32_t head_dim,
                             std::int32_t kv_heads) {
    const std::size_t base = static_cast<std::size_t>(address.group_layer) * 4;
    return {
        .k_pages = transaction.plane(address.group_id, base),
        .v_pages = transaction.plane(address.group_id, base + 1),
        .k_scale_pages = transaction.plane(address.group_id, base + 2),
        .v_scale_pages = transaction.plane(address.group_id, base + 3),
        .block_table = transaction.execution_view(address.group_id).block_table,
        .head_dim = head_dim,
        .num_kv_heads = kv_heads,
        .storage = KvCacheStorage::RK4V4E8,
    };
}

class EventPair {
public:
    EventPair() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~EventPair() {
        cudaEventDestroy(stop_);
        cudaEventDestroy(start_);
    }
    void start(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
    double stop(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

void write_bf16(const std::filesystem::path& directory, const std::string& name,
                const Tensor& tensor) {
    if (directory.empty()) return;
    std::vector<std::uint16_t> host(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    std::ofstream output(directory / (name + ".bf16"), std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(host.data()),
                 static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
    if (!output) throw std::runtime_error("failed to write Gemma persistent parity dump");
}

std::int32_t execute_chunk(const ModelWeights& weights, HeterogeneousKVCache& cache,
                           std::uint32_t first, std::span<const std::int32_t> input_tokens,
                           WorkspaceArena& activations,
                           WorkspaceArena& attention_workspace, cudaStream_t stream,
                           bool produce_logits,
                           const std::filesystem::path& dump_directory = {}) {
    const std::int32_t tokens = static_cast<std::int32_t>(input_tokens.size());
    auto activation_scope = activations.scope();
    Tensor ids = activations.alloc(DType::I32, {tokens});
    Tensor positions = activations.alloc(DType::I32, {tokens});
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        host_positions[static_cast<std::size_t>(token)] =
            static_cast<std::int32_t>(first) + token;
    }
    CUDA_CHECK(cudaMemcpyAsync(ids.data, input_tokens.data(),
                               input_tokens.size_bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(positions.data, host_positions.data(),
                               host_positions.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));

    Tensor hidden = activations.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor normalized = activations.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor packed_storage = activations.alloc(DType::BF16, {43008, tokens});
    Tensor query_storage = activations.alloc(
        DType::BF16, {TextConfig::full_head_dim, TextConfig::query_heads, tokens});
    Tensor key_storage = activations.alloc(
        DType::BF16, {TextConfig::full_head_dim, TextConfig::sliding_kv_heads, tokens});
    Tensor value_storage = activations.alloc(
        DType::BF16, {TextConfig::full_head_dim, TextConfig::sliding_kv_heads, tokens});
    Tensor attended_storage = activations.alloc(
        DType::BF16, {TextConfig::full_head_dim, TextConfig::query_heads, tokens});
    Tensor projected = activations.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor post = activations.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor product = activations.alloc(DType::BF16, {TextConfig::intermediate, tokens});
    Tensor feedforward = activations.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor final_hidden = activations.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor logits = activations.alloc(DType::FP32, {TextConfig::vocabulary});

    auto transaction = cache.begin_transaction(0, first, static_cast<std::uint32_t>(tokens), stream);
    ops::embedding(ids, weights.embedding, hidden, stream);
    scale_embedding(hidden, TextConfig::embedding_scale_bf16, stream);
    write_bf16(dump_directory, "persistent_embeddings", hidden);

    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const bool full = TextConfig::is_full_attention(layer);
        const auto& layer_weights = weights.layers[layer];
        const std::int32_t head_dim =
            full ? TextConfig::full_head_dim : TextConfig::sliding_head_dim;
        const std::int32_t kv_heads =
            full ? TextConfig::full_kv_heads : TextConfig::sliding_kv_heads;
        const std::int32_t q_rows = TextConfig::query_heads * head_dim;
        const std::int32_t input_rows = full ? 18432 : 16384;
        Tensor packed(packed_storage.data, DType::BF16, {input_rows, tokens});
        Tensor query(query_storage.data, DType::BF16,
                     {head_dim, static_cast<int>(TextConfig::query_heads), tokens});
        Tensor key(key_storage.data, DType::BF16, {head_dim, kv_heads, tokens});
        Tensor value(value_storage.data, DType::BF16, {head_dim, kv_heads, tokens});
        Tensor attended(attended_storage.data, DType::BF16,
                        {head_dim, static_cast<int>(TextConfig::query_heads), tokens});

        ops::rmsnorm(hidden, layer_weights.input_norm, TextConfig::rms_epsilon, false,
                     normalized, stream);
        ops::linear(normalized, layer_weights.attention.input, packed, stream);
        if (!dump_directory.empty() && layer == 0) {
            write_bf16(dump_directory, "persistent_layer0_normalized", normalized);
            write_bf16(dump_directory, "persistent_layer0_packed", packed);
        }
        prepare_qkv(packed, layer_weights.attention.query_norm,
                    layer_weights.attention.key_norm, head_dim, kv_heads,
                    full ? TextConfig::full_rope_theta : TextConfig::sliding_rope_theta,
                    full ? TextConfig::full_rotary_active_dim / 2 : head_dim / 2,
                    static_cast<std::int32_t>(first), query, key, value, stream);
        if (!dump_directory.empty() && (layer == 0 || layer == 5)) {
            const std::string prefix = "persistent_layer" + std::to_string(layer);
            write_bf16(dump_directory, prefix + "_q", query);
            write_bf16(dump_directory, prefix + "_k", key);
            write_bf16(dump_directory, prefix + "_v", value);
        }

        const TextKvLayerAddress address = text_kv_layer_address(static_cast<std::uint32_t>(layer));
        PagedKVLayerView view = layer_cache(transaction, address, head_dim, kv_heads);
        if (full) {
            const ops::CausalAttentionExecutionEnvelope envelope{
                first + 1U, first + static_cast<std::uint32_t>(tokens)};
            ops::causal_full_softmax_attention(query, key, value, positions, kFullGeometry, 1.0F,
                                               view, envelope, attention_workspace, attended,
                                               stream);
        } else {
            ops::causal_sliding_softmax_attention(
                query, key, value, positions, kSlidingGeometry, TextConfig::sliding_window, 1.0F,
                view, attended, stream);
        }
        if (!dump_directory.empty() && (layer == 0 || layer == 5)) {
            write_bf16(dump_directory, "persistent_layer" + std::to_string(layer) +
                                           "_attention_output", attended);
        }

        ops::linear(attended.view({q_rows, tokens}), layer_weights.attention.output,
                    projected, stream);
        ops::rmsnorm(projected, layer_weights.post_attention_norm,
                     TextConfig::rms_epsilon, false, post, stream);
        ops::residual_add(post, hidden, stream);
        ops::rmsnorm(hidden, layer_weights.pre_feedforward_norm,
                     TextConfig::rms_epsilon, false, normalized, stream);
        Tensor gate_up(packed_storage.data, DType::BF16,
                       {static_cast<int>(2 * TextConfig::intermediate), tokens});
        ops::linear(normalized, layer_weights.gate_up, gate_up, stream);
        gelu_tanh_mul_packed(gate_up, product, stream);
        ops::linear(product, layer_weights.down, feedforward, stream);
        ops::rmsnorm(feedforward, layer_weights.post_feedforward_norm,
                     TextConfig::rms_epsilon, false, post, stream);
        add_scaled(post, layer_weights.layer_scalar, hidden, stream);
        write_bf16(dump_directory, "persistent_layer" + std::to_string(layer) + "_output",
                   hidden);
    }
    transaction.commit(stream);

    if (!produce_logits) return -1;
    ops::rmsnorm(hidden, weights.final_norm, TextConfig::rms_epsilon, false,
                 final_hidden, stream);
    Tensor last_hidden = final_hidden.slice(1, tokens - 1, 1).view({TextConfig::hidden});
    fp8_tied_logits(last_hidden, weights.embedding, TextConfig::final_logit_softcap,
                    logits, stream);
    std::vector<float> host_logits(TextConfig::vocabulary);
    CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), logits.data,
                               host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return static_cast<std::int32_t>(
        std::max_element(host_logits.begin(), host_logits.end()) - host_logits.begin());
}

} // namespace

PersistentRunResult run_persistent_target(const std::filesystem::path& artifact_path,
                                          const PersistentRunOptions& options) {
    const std::uint32_t prefill_tokens = options.input_tokens.empty()
                                             ? options.prefill_tokens
                                             : static_cast<std::uint32_t>(options.input_tokens.size());
    const std::uint32_t appended_tokens = options.generation_tokens > 0
                                              ? options.generation_tokens - 1
                                              : static_cast<std::uint32_t>(options.run_deep_decode);
    if (options.maximum_context < TextConfig::sliding_window ||
        options.maximum_context > TextConfig::maximum_position ||
        prefill_tokens == 0 || options.chunk_tokens == 0 || options.chunk_tokens > 64 ||
        prefill_tokens + appended_tokens > options.maximum_context ||
        (options.generation_tokens > 0 && options.run_deep_decode) ||
        options.input_token < 0 ||
        options.input_token >= static_cast<std::int32_t>(TextConfig::vocabulary)) {
        throw std::invalid_argument("Gemma persistent target options are invalid");
    }
    if (std::any_of(options.input_tokens.begin(), options.input_tokens.end(), [](std::int32_t token) {
            return token < 0 || token >= static_cast<std::int32_t>(TextConfig::vocabulary);
        })) {
        throw std::invalid_argument("Gemma persistent target input token is invalid");
    }

    DeviceContext device(options.device_id);
    artifact::Reader reader(artifact_path);
    artifact::Binder binder(reader);
    ArtifactLoadPlan plan = bind_artifact(binder);
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, plan.materialization, device);
    const ModelWeights weights = load_weights(materialized, plan.bindings);
    if (!options.dump_directory.empty()) {
        std::filesystem::create_directories(options.dump_directory);
    }

    PersistentRunResult result;
    result.weights_bytes = materialized.stats().device_capacity_bytes;
    result.free_after_weights = free_device_bytes();

    const auto specs = make_text_kv_group_specs({
        .maximum_context = options.maximum_context,
        .maximum_transaction_tokens = options.chunk_tokens,
        .global_resident_token_capacity = options.maximum_context,
        .table_rows = 1,
    });
    LayoutBuilder cache_builder;
    const HeterogeneousKVCacheLayout cache_layout =
        plan_heterogeneous_kv_cache(cache_builder, specs);
    DeviceBuffer cache_backing(cache_builder.finish(256));
    HeterogeneousKVCache cache({cache_backing.p, cache_backing.bytes}, cache_layout);
    cache.activate(0, device.stream);
    result.kv_payload_bytes = cache_layout.payload_bytes();
    result.kv_metadata_bytes = cache_layout.metadata_bytes();
    result.free_after_cache = free_device_bytes();

    const ops::CausalAttentionExecutionEnvelope maximum_envelope{
        1, options.maximum_context};
    const std::size_t attention_bytes =
        ops::causal_full_softmax_attention_workspace_capacity_bytes(
            kFullGeometry, KvCacheStorage::RK4V4E8, maximum_envelope, 1,
            static_cast<std::int32_t>(options.chunk_tokens));
    DeviceBuffer activation_backing(kActivationWorkspaceBytes);
    DeviceBuffer attention_backing(std::max<std::size_t>(attention_bytes, 256));
    WorkspaceArena activations({activation_backing.p, activation_backing.bytes});
    WorkspaceArena attention_workspace({attention_backing.p, attention_backing.bytes});
    result.workspace_bytes = activation_backing.bytes + attention_backing.bytes;
    result.free_after_workspace = free_device_bytes();

    EventPair timer;
    timer.start(device.stream);
    std::uint32_t first = 0;
    while (first < prefill_tokens) {
        const std::int32_t count = static_cast<std::int32_t>(std::min(
            options.chunk_tokens, prefill_tokens - first));
        const bool produce_logits = (options.generation_tokens > 0 || !options.run_deep_decode) &&
                                    first + static_cast<std::uint32_t>(count) ==
                                        prefill_tokens;
        std::vector<std::int32_t> repeated_tokens;
        std::span<const std::int32_t> chunk;
        if (options.input_tokens.empty()) {
            repeated_tokens.assign(static_cast<std::size_t>(count), options.input_token);
            chunk = repeated_tokens;
        } else {
            chunk = std::span<const std::int32_t>(options.input_tokens).subspan(first, count);
        }
        const std::int32_t greedy = execute_chunk(
            weights, cache, first, chunk, activations, attention_workspace, device.stream,
            produce_logits,
            produce_logits ? options.dump_directory : std::filesystem::path{});
        if (produce_logits) result.greedy_token = greedy;
        first += static_cast<std::uint32_t>(count);
    }
    result.prefill_milliseconds = timer.stop(device.stream);

    if (options.generation_tokens > 0) {
        result.generated_tokens.push_back(result.greedy_token);
        timer.start(device.stream);
        while (result.generated_tokens.size() < options.generation_tokens) {
            const std::int32_t token = result.generated_tokens.back();
            result.greedy_token = execute_chunk(weights, cache, first,
                                                 std::span<const std::int32_t>(&token, 1),
                                                 activations, attention_workspace,
                                                 device.stream, true);
            result.generated_tokens.push_back(result.greedy_token);
            ++first;
        }
        result.decode_milliseconds = timer.stop(device.stream);
    } else if (options.run_deep_decode) {
        timer.start(device.stream);
        result.greedy_token = execute_chunk(
            weights, cache, first, std::span<const std::int32_t>(&options.input_token, 1),
            activations, attention_workspace, device.stream, true);
        result.decode_milliseconds = timer.stop(device.stream);
    }
    result.final_frontier = cache.frontier(0);
    return result;
}

} // namespace ninfer::targets::gemma4_31b_it::detail
