#include "targets/gemma4_31b_it/impl/runtime/persistent_model.h"
#include "targets/gemma4_31b_it/impl/runtime/runtime_chunk.h"

#include <ninfer/ops/embedding.h>
#include <ninfer/ops/linear.h>
#include <ninfer/ops/residual_add.h>
#include <ninfer/ops/rmsnorm.h>
#include <ninfer/ops/softmax_attention.h>
#include <ninfer/targets/gemma4_31b_it/config.h>
#include <ninfer/targets/gemma4_31b_it/package.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/heterogeneous_kv_cache.h"
#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "targets/gemma4/impl/runtime/continuation.h"
#include "targets/gemma4_31b_it/impl/load/bindings.h"
#include "targets/gemma4_31b_it/impl/runtime/kv_groups.h"
#include "targets/gemma4_31b_it/impl/runtime/model_weights.h"
#include "targets/gemma4_31b_it/impl/runtime/reference_kernels.h"

#include <algorithm>
#include <chrono>
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

constexpr std::uint32_t kMaximumChunkTokens = 2048;
constexpr ops::AttentionHeadGeometry kSlidingGeometry{
    TextConfig::sliding_head_dim, TextConfig::query_heads, TextConfig::sliding_kv_heads};
constexpr ops::AttentionHeadGeometry kFullGeometry{
    TextConfig::full_head_dim, TextConfig::query_heads, TextConfig::full_kv_heads};
constexpr std::string_view kContinuationModelBinding =
    "gemma4_31b_it\ngemma4-31b-it\ngroupwise-int";

double elapsed_milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

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

PagedKVLayerView layer_cache(const HeterogeneousKVCache& cache, std::int32_t row,
                             TextKvLayerAddress address, std::int32_t head_dim,
                             std::int32_t kv_heads) {
    const std::size_t base = static_cast<std::size_t>(address.group_layer) * 4;
    return {
        .k_pages = cache.plane(address.group_id, base),
        .v_pages = cache.plane(address.group_id, base + 1),
        .k_scale_pages = cache.plane(address.group_id, base + 2),
        .v_scale_pages = cache.plane(address.group_id, base + 3),
        .block_table = cache.execution_view(row, address.group_id).block_table,
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

struct ModelBuffers {
    Tensor ids;
    Tensor positions;
    Tensor hidden;
    Tensor normalized;
    Tensor packed_storage;
    Tensor query_storage;
    Tensor key_storage;
    Tensor value_storage;
    Tensor attended_storage;
    Tensor projected;
    Tensor post;
    Tensor product;
    Tensor feedforward;
    Tensor final_hidden;
    Tensor logits;
};

template <class Allocator>
ModelBuffers allocate_model_buffers(Allocator& activations, std::int32_t tokens,
                                    bool all_logits = false) {
    return {
        .ids = activations.alloc(DType::I32, {tokens}),
        .positions = activations.alloc(DType::I32, {tokens}),
        .hidden = activations.alloc(DType::BF16, {TextConfig::hidden, tokens}),
        .normalized = activations.alloc(DType::BF16, {TextConfig::hidden, tokens}),
        .packed_storage = activations.alloc(DType::BF16, {43008, tokens}),
        .query_storage = activations.alloc(
            DType::BF16, {TextConfig::full_head_dim, TextConfig::query_heads, tokens}),
        .key_storage = activations.alloc(
            DType::BF16, {TextConfig::full_head_dim, TextConfig::sliding_kv_heads, tokens}),
        .value_storage = activations.alloc(
            DType::BF16, {TextConfig::full_head_dim, TextConfig::sliding_kv_heads, tokens}),
        .attended_storage = activations.alloc(
            DType::BF16, {TextConfig::full_head_dim, TextConfig::query_heads, tokens}),
        .projected = activations.alloc(DType::BF16, {TextConfig::hidden, tokens}),
        .post = activations.alloc(DType::BF16, {TextConfig::hidden, tokens}),
        .product = activations.alloc(DType::BF16, {TextConfig::intermediate, tokens}),
        .feedforward = activations.alloc(DType::BF16, {TextConfig::hidden, tokens}),
        .final_hidden = activations.alloc(DType::BF16, {TextConfig::hidden, tokens}),
        .logits = activations.alloc(
            DType::FP32, {TextConfig::vocabulary, all_logits ? tokens : 1}),
    };
}

void upload_decode_parameters(ModelBuffers& buffers, std::uint32_t first,
                              std::span<const std::int32_t> input_tokens,
                              cudaStream_t stream) {
    std::vector<std::int32_t> host_positions(input_tokens.size());
    for (std::size_t token = 0; token < input_tokens.size(); ++token) {
        host_positions[token] = static_cast<std::int32_t>(first + token);
    }
    CUDA_CHECK(cudaMemcpyAsync(buffers.ids.data, input_tokens.data(), input_tokens.size_bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.positions.data, host_positions.data(),
                               host_positions.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
}

void enqueue_model(const ModelWeights& weights, HeterogeneousKVTransaction& transaction,
                   ModelBuffers& buffers,
                   ops::CausalAttentionExecutionEnvelope full_envelope,
                   WorkspaceArena& attention_workspace, cudaStream_t stream,
                   bool produce_logits, bool produce_all_logits = false,
                   const std::filesystem::path& dump_directory = {}) {
    const std::int32_t tokens = buffers.ids.ne[0];
    Tensor& hidden = buffers.hidden;
    Tensor& normalized = buffers.normalized;
    Tensor& packed_storage = buffers.packed_storage;
    Tensor& query_storage = buffers.query_storage;
    Tensor& key_storage = buffers.key_storage;
    Tensor& value_storage = buffers.value_storage;
    Tensor& attended_storage = buffers.attended_storage;
    Tensor& projected = buffers.projected;
    Tensor& post = buffers.post;
    Tensor& product = buffers.product;
    Tensor& feedforward = buffers.feedforward;

    ops::embedding(buffers.ids, weights.embedding, hidden, stream);
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
                    buffers.positions, query, key, value, stream);
        if (!dump_directory.empty() && (layer == 0 || layer == 5)) {
            const std::string prefix = "persistent_layer" + std::to_string(layer);
            write_bf16(dump_directory, prefix + "_q", query);
            write_bf16(dump_directory, prefix + "_k", key);
            write_bf16(dump_directory, prefix + "_v", value);
        }

        const TextKvLayerAddress address = text_kv_layer_address(static_cast<std::uint32_t>(layer));
        PagedKVLayerView view = layer_cache(transaction, address, head_dim, kv_heads);
        // Cache publication is page-local even when the surrounding projections and MLP use a
        // wider model chunk. Sequential attention tiles preserve causal visibility and the local
        // ring's eviction order while allowing GEMMs to run at T=256..1024.
        for (std::int32_t begin = 0; begin < tokens; begin += kPagedKVPageSize) {
            const std::int32_t count = std::min<std::int32_t>(kPagedKVPageSize, tokens - begin);
            const Tensor query_tile = query.slice(2, begin, count);
            const Tensor key_tile = key.slice(2, begin, count);
            const Tensor value_tile = value.slice(2, begin, count);
            const Tensor position_tile = buffers.positions.slice(0, begin, count);
            Tensor attended_tile = attended.slice(2, begin, count);
            if (full) {
                ops::CausalAttentionExecutionEnvelope tile_envelope = full_envelope;
                if (tokens > kPagedKVPageSize) {
                    tile_envelope = {
                        full_envelope.min_visible_keys + static_cast<std::uint32_t>(begin),
                        full_envelope.min_visible_keys + static_cast<std::uint32_t>(begin + count - 1),
                    };
                }
                ops::causal_full_softmax_attention(
                    query_tile, key_tile, value_tile, position_tile, kFullGeometry, 1.0F,
                    view, tile_envelope, attention_workspace, attended_tile, stream);
            } else {
                ops::causal_sliding_softmax_attention(
                    query_tile, key_tile, value_tile, position_tile, kSlidingGeometry,
                    TextConfig::sliding_window, 1.0F, view, attended_tile, stream);
            }
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

    if (!produce_logits) return;
    ops::rmsnorm(hidden, weights.final_norm, TextConfig::rms_epsilon, false,
                 buffers.final_hidden, stream);
    const std::int32_t first_logit = produce_all_logits ? 0 : tokens - 1;
    for (std::int32_t token = first_logit; token < tokens; ++token) {
        Tensor selected_hidden =
            buffers.final_hidden.slice(1, token, 1).view({TextConfig::hidden});
        Tensor selected_logits = buffers.logits
                                     .slice(1, produce_all_logits ? token : 0, 1)
                                     .view({TextConfig::vocabulary});
        fp8_tied_logits(selected_hidden, weights.embedding,
                        TextConfig::final_logit_softcap, selected_logits, stream);
    }
}

std::vector<std::int32_t> read_greedy_tokens(const Tensor& logits, cudaStream_t stream) {
    const std::int32_t columns = logits.ne[1];
    std::vector<float> host_logits(static_cast<std::size_t>(TextConfig::vocabulary) * columns);
    CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), logits.data,
                               host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::int32_t> out(static_cast<std::size_t>(columns));
    for (std::int32_t column = 0; column < columns; ++column) {
        const auto begin = host_logits.begin() +
                           static_cast<std::ptrdiff_t>(column) * TextConfig::vocabulary;
        out[static_cast<std::size_t>(column)] = static_cast<std::int32_t>(
            std::max_element(begin, begin + TextConfig::vocabulary) - begin);
    }
    return out;
}

std::int32_t read_greedy_token(const Tensor& logits, cudaStream_t stream) {
    return read_greedy_tokens(logits, stream).front();
}

std::int32_t execute_chunk(const ModelWeights& weights, HeterogeneousKVCache& cache,
                           std::int32_t row, std::uint32_t first,
                           std::span<const std::int32_t> input_tokens,
                           std::uint32_t maximum_context, WorkspaceArena& activations,
                           WorkspaceArena& attention_workspace, cudaStream_t stream,
                           bool produce_logits, const Tensor& hidden_state = {},
                           const std::filesystem::path& dump_directory = {}) {
    const std::int32_t tokens = static_cast<std::int32_t>(input_tokens.size());
    auto activation_scope = activations.scope();
    ModelBuffers buffers = allocate_model_buffers(activations, tokens);
    upload_decode_parameters(buffers, first, input_tokens, stream);
    auto transaction = cache.begin_transaction(row, first, static_cast<std::uint32_t>(tokens), stream);
    const ops::CausalAttentionExecutionEnvelope envelope =
        tokens == 1 ? ops::CausalAttentionExecutionEnvelope{1, maximum_context}
                    : ops::CausalAttentionExecutionEnvelope{
                          first + 1U, first + static_cast<std::uint32_t>(tokens)};
    enqueue_model(weights, transaction, buffers, envelope, attention_workspace, stream,
                  produce_logits, false, dump_directory);
    if (produce_logits && hidden_state.data != nullptr) {
        const Tensor last_hidden =
            buffers.final_hidden.slice(1, tokens - 1, 1).view({TextConfig::hidden});
        CUDA_CHECK(cudaMemcpyAsync(hidden_state.data, last_hidden.data, last_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    transaction.commit(stream);
    return produce_logits ? read_greedy_token(buffers.logits, stream) : -1;
}

struct AssistantBuffers {
    Tensor ids;
    Tensor positions;
    Tensor last_key_positions;
    Tensor target_embedding;
    Tensor concatenated;
    Tensor hidden;
    Tensor normalized;
    Tensor query_projected;
    Tensor query_storage;
    Tensor attended_storage;
    Tensor projected;
    Tensor packed_storage;
    Tensor product;
    Tensor feedforward;
    Tensor post;
    Tensor body_output;
    Tensor logits;
};

AssistantBuffers allocate_assistant_buffers(WorkspaceArena& activations) {
    return {
        .ids = activations.alloc(DType::I32, {1}),
        .positions = activations.alloc(DType::I32, {1}),
        .last_key_positions = activations.alloc(DType::I32, {1}),
        .target_embedding = activations.alloc(DType::BF16, {TextConfig::hidden}),
        .concatenated = activations.alloc(DType::BF16, {AssistantConfig::input_rows}),
        .hidden = activations.alloc(DType::BF16, {AssistantConfig::hidden}),
        .normalized = activations.alloc(DType::BF16, {AssistantConfig::hidden}),
        .query_projected = activations.alloc(
            DType::BF16, {AssistantConfig::full_head_dim, AssistantConfig::query_heads}),
        .query_storage = activations.alloc(
            DType::BF16, {AssistantConfig::full_head_dim, AssistantConfig::query_heads}),
        .attended_storage = activations.alloc(
            DType::BF16, {AssistantConfig::full_head_dim, AssistantConfig::query_heads}),
        .projected = activations.alloc(DType::BF16, {AssistantConfig::hidden}),
        .packed_storage = activations.alloc(
            DType::BF16, {2 * AssistantConfig::intermediate}),
        .product = activations.alloc(DType::BF16, {AssistantConfig::intermediate}),
        .feedforward = activations.alloc(DType::BF16, {AssistantConfig::hidden}),
        .post = activations.alloc(DType::BF16, {AssistantConfig::hidden}),
        .body_output = activations.alloc(DType::BF16, {AssistantConfig::hidden}),
        .logits = activations.alloc(DType::FP32, {AssistantConfig::vocabulary}),
    };
}

struct AssistantStepResult {
    std::int32_t token = -1;
    double proposal_head_milliseconds = 0.0;
};

AssistantStepResult execute_assistant(const ModelWeights& target_weights,
                                      const AssistantWeights& assistant_weights,
                                      const HeterogeneousKVCache& cache, std::int32_t row,
                                      std::int32_t current_token, std::uint32_t position,
                                      std::uint32_t maximum_context,
                                      const Tensor& input_hidden_state,
                                      Tensor& output_hidden_state,
                                      WorkspaceArena& activations,
                                      WorkspaceArena& attention_workspace,
                                      cudaStream_t stream) {
    if (position == 0) {
        throw std::logic_error("Gemma assistant requires a populated target cache");
    }
    auto activation_scope = activations.scope();
    AssistantBuffers buffers = allocate_assistant_buffers(activations);
    const std::int32_t logical_position = static_cast<std::int32_t>(position);
    const std::int32_t last_key_position = logical_position - 1;
    CUDA_CHECK(cudaMemcpyAsync(buffers.ids.data, &current_token, sizeof(current_token),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.positions.data, &logical_position,
                               sizeof(logical_position), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.last_key_positions.data, &last_key_position,
                               sizeof(last_key_position), cudaMemcpyHostToDevice, stream));
    ops::embedding(buffers.ids, target_weights.embedding, buffers.target_embedding, stream);
    scale_embedding(buffers.target_embedding, TextConfig::embedding_scale_bf16, stream);
    CUDA_CHECK(cudaMemcpyAsync(buffers.concatenated.data, buffers.target_embedding.data,
                               buffers.target_embedding.bytes(), cudaMemcpyDeviceToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<std::byte*>(buffers.concatenated.data) + buffers.target_embedding.bytes(),
        input_hidden_state.data, input_hidden_state.bytes(), cudaMemcpyDeviceToDevice,
        stream));
    ops::linear(buffers.concatenated, assistant_weights.input_projection,
                buffers.hidden, stream);

    const ops::CausalAttentionExecutionEnvelope full_envelope{1, maximum_context};
    for (std::size_t layer = 0; layer < assistant_weights.layers.size(); ++layer) {
        const bool full = AssistantConfig::layer_types[layer] == gemma4::AttentionType::Full;
        const std::int32_t head_dim =
            full ? AssistantConfig::full_head_dim : AssistantConfig::sliding_head_dim;
        const std::int32_t kv_heads =
            full ? AssistantConfig::full_kv_heads : AssistantConfig::sliding_kv_heads;
        const std::int32_t query_rows = AssistantConfig::query_heads * head_dim;
        const auto& layer_weights = assistant_weights.layers[layer];
        Tensor query(buffers.query_storage.data, DType::BF16,
                     {head_dim, static_cast<std::int32_t>(AssistantConfig::query_heads), 1});
        Tensor attended(buffers.attended_storage.data, DType::BF16,
                        {head_dim, static_cast<std::int32_t>(AssistantConfig::query_heads), 1});
        Tensor projected_query(buffers.query_projected.data, DType::BF16, {query_rows, 1});

        ops::rmsnorm(buffers.hidden, layer_weights.input_norm,
                     TextConfig::rms_epsilon, false, buffers.normalized, stream);
        ops::linear(buffers.normalized, layer_weights.attention.query,
                    projected_query, stream);
        prepare_query(projected_query, layer_weights.attention.query_norm, head_dim,
                      full ? AssistantConfig::full_rope_theta
                           : AssistantConfig::sliding_rope_theta,
                      full ? AssistantConfig::full_rotary_active_dim / 2 : head_dim / 2,
                      buffers.positions, query, stream);
        const TextKvLayerAddress address = text_kv_layer_address(
            AssistantConfig::shared_target_kv_layers[layer]);
        const PagedKVLayerView view = layer_cache(cache, row, address, head_dim, kv_heads);
        if (full) {
            ops::shared_kv_full_softmax_attention(
                query, buffers.last_key_positions, kFullGeometry, 1.0F, view,
                full_envelope, attention_workspace, attended, stream);
        } else {
            ops::shared_kv_sliding_softmax_attention(
                query, buffers.last_key_positions, kSlidingGeometry,
                AssistantConfig::sliding_window, 1.0F, view, attended, stream);
        }
        ops::linear(attended.view({query_rows}), layer_weights.attention.output,
                    buffers.projected, stream);
        ops::rmsnorm(buffers.projected, layer_weights.post_attention_norm,
                     TextConfig::rms_epsilon, false, buffers.post, stream);
        ops::residual_add(buffers.post, buffers.hidden, stream);
        ops::rmsnorm(buffers.hidden, layer_weights.pre_feedforward_norm,
                     TextConfig::rms_epsilon, false, buffers.normalized, stream);
        Tensor gate_up(buffers.packed_storage.data, DType::BF16,
                       {static_cast<std::int32_t>(2 * AssistantConfig::intermediate), 1});
        Tensor product(buffers.product.data, DType::BF16,
                       {static_cast<std::int32_t>(AssistantConfig::intermediate), 1});
        ops::linear(buffers.normalized, layer_weights.gate_up, gate_up, stream);
        gelu_tanh_mul_packed(gate_up, product, stream);
        ops::linear(product, layer_weights.down, buffers.feedforward, stream);
        ops::rmsnorm(buffers.feedforward, layer_weights.post_feedforward_norm,
                     TextConfig::rms_epsilon, false, buffers.post, stream);
        add_scaled(buffers.post, layer_weights.layer_scalar, buffers.hidden, stream);
    }
    ops::rmsnorm(buffers.hidden, assistant_weights.final_norm,
                 TextConfig::rms_epsilon, false, buffers.body_output, stream);
    ops::linear(buffers.body_output, assistant_weights.output_projection,
                output_hidden_state, stream);
    EventPair proposal_head_timer;
    proposal_head_timer.start(stream);
    fp8_tied_logits_raw(buffers.body_output, assistant_weights.output_head,
                        buffers.logits, stream);
    const double proposal_head_milliseconds = proposal_head_timer.stop(stream);
    return {
        .token = read_greedy_token(buffers.logits, stream),
        .proposal_head_milliseconds = proposal_head_milliseconds,
    };
}

struct VerifyResult {
    std::array<std::int32_t, AssistantConfig::maximum_draft_size + 1> greedy{};
    std::uint32_t accepted_drafts = 0;
    std::uint32_t committed_tokens = 0;
};

VerifyResult finalize_mtp_verify(ModelBuffers& buffers,
                                 HeterogeneousKVTransaction& transaction,
                                 std::span<const std::int32_t> draft_tokens,
                                 std::uint32_t remaining_outputs,
                                 const Tensor& target_hidden_state,
                                 cudaStream_t stream) {
    const std::vector<std::int32_t> greedy = read_greedy_tokens(buffers.logits, stream);
    VerifyResult result;
    std::copy(greedy.begin(), greedy.end(), result.greedy.begin());
    while (result.accepted_drafts < draft_tokens.size() &&
           draft_tokens[result.accepted_drafts] == greedy[result.accepted_drafts]) {
        ++result.accepted_drafts;
    }
    result.committed_tokens = std::min(result.accepted_drafts + 1U, remaining_outputs);
    const Tensor selected_hidden = buffers.final_hidden
                                       .slice(1, result.committed_tokens - 1, 1)
                                       .view({TextConfig::hidden});
    CUDA_CHECK(cudaMemcpyAsync(target_hidden_state.data, selected_hidden.data,
                               selected_hidden.bytes(), cudaMemcpyDeviceToDevice, stream));
    transaction.commit_prefix(result.committed_tokens, stream);
    return result;
}

VerifyResult execute_mtp_verify(const ModelWeights& weights,
                                HeterogeneousKVCache& cache, std::int32_t row,
                                std::uint32_t first,
                                std::int32_t current_token,
                                std::span<const std::int32_t> draft_tokens,
                                std::uint32_t maximum_context,
                                std::uint32_t remaining_outputs,
                                const Tensor& target_hidden_state,
                                WorkspaceArena& activations,
                                WorkspaceArena& attention_workspace,
                                cudaStream_t stream) {
    auto activation_scope = activations.scope();
    const std::int32_t token_count = static_cast<std::int32_t>(draft_tokens.size() + 1);
    ModelBuffers buffers = allocate_model_buffers(activations, token_count, true);
    std::vector<std::int32_t> tokens;
    tokens.reserve(static_cast<std::size_t>(token_count));
    tokens.push_back(current_token);
    tokens.insert(tokens.end(), draft_tokens.begin(), draft_tokens.end());
    upload_decode_parameters(buffers, first, tokens, stream);
    auto transaction = cache.begin_transaction(
        row, first, static_cast<std::uint32_t>(token_count), stream);
    const ops::CausalAttentionExecutionEnvelope envelope{1, maximum_context};
    enqueue_model(weights, transaction, buffers, envelope, attention_workspace, stream,
                  true, true);
    return finalize_mtp_verify(buffers, transaction, draft_tokens, remaining_outputs,
                               target_hidden_state, stream);
}

class PersistentDecodeGraph {
public:
    void capture(const ModelWeights& weights, HeterogeneousKVCache& cache,
                 std::uint32_t maximum_context, WorkspaceArena& activations,
                 WorkspaceArena& attention_workspace, cudaStream_t stream) {
        if (executable_.ready()) {
            throw std::logic_error("Gemma decode graph cannot be recaptured");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t free_before = free_device_bytes();
        {
            auto activation_scope = activations.scope();
            ModelBuffers buffers = allocate_model_buffers(activations, 1);
            const std::int32_t token = 0;
            const std::uint32_t first = cache.frontier(0);
            upload_decode_parameters(buffers, first,
                                     std::span<const std::int32_t>(&token, 1), stream);
            auto transaction = cache.begin_transaction(0, first, 1, stream);
            const ops::CausalAttentionExecutionEnvelope envelope{1, maximum_context};
            definition_.capture(stream, [&] {
                enqueue_model(weights, transaction, buffers, envelope, attention_workspace,
                              stream, true, false);
            });
            transaction.rollback();
        }
        executable_.instantiate(definition_);
        executable_.upload(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t free_after = free_device_bytes();
        device_bytes_ = free_before > free_after ? free_before - free_after : 0;
        capture_count_ = 1;
    }

    std::int32_t replay(HeterogeneousKVCache& cache, std::uint32_t first,
                        std::int32_t input_token, WorkspaceArena& activations,
                        cudaStream_t stream, bool commit) {
        if (!executable_.ready()) {
            throw std::logic_error("Gemma decode graph was not captured");
        }
        auto activation_scope = activations.scope();
        ModelBuffers buffers = allocate_model_buffers(activations, 1);
        upload_decode_parameters(buffers, first,
                                 std::span<const std::int32_t>(&input_token, 1), stream);
        auto transaction = cache.begin_transaction(0, first, 1, stream);
        executable_.launch(stream);
        ++replay_count_;
        if (commit) {
            transaction.commit(stream);
            return read_greedy_token(buffers.logits, stream);
        }
        const std::int32_t greedy = read_greedy_token(buffers.logits, stream);
        transaction.rollback();
        return greedy;
    }

    [[nodiscard]] std::size_t device_bytes() const noexcept { return device_bytes_; }
    [[nodiscard]] std::uint32_t capture_count() const noexcept { return capture_count_; }
    [[nodiscard]] std::uint32_t replay_count() const noexcept { return replay_count_; }

private:
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable executable_;
    std::size_t device_bytes_ = 0;
    std::uint32_t capture_count_ = 0;
    std::uint32_t replay_count_ = 0;
};

class PersistentMtpVerifyGraph {
public:
    void capture(const ModelWeights& weights, HeterogeneousKVCache& cache,
                 std::uint32_t maximum_context, std::uint32_t draft_width,
                 WorkspaceArena& activations,
                 WorkspaceArena& attention_workspace, cudaStream_t stream) {
        if (executable_.ready()) {
            throw std::logic_error("Gemma MTP verify graph cannot be recaptured");
        }
        if (draft_width == 0 || draft_width > AssistantConfig::maximum_draft_size) {
            throw std::invalid_argument("Gemma MTP verify graph width must be in [1,6]");
        }
        draft_width_ = draft_width;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t free_before = free_device_bytes();
        {
            auto activation_scope = activations.scope();
            const std::int32_t token_count = static_cast<std::int32_t>(draft_width + 1);
            ModelBuffers buffers = allocate_model_buffers(activations, token_count, true);
            const std::vector<std::int32_t> tokens(static_cast<std::size_t>(token_count), 0);
            const std::uint32_t first = cache.frontier(0);
            upload_decode_parameters(buffers, first, tokens, stream);
            auto transaction = cache.begin_transaction(
                0, first, static_cast<std::uint32_t>(token_count), stream);
            definition_.capture(stream, [&] {
                enqueue_model(weights, transaction, buffers, {1, maximum_context},
                              attention_workspace, stream, true, true);
            });
            transaction.rollback();
        }
        executable_.instantiate(definition_);
        executable_.upload(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t free_after = free_device_bytes();
        device_bytes_ = free_before > free_after ? free_before - free_after : 0;
        capture_count_ = 1;
    }

    VerifyResult replay(HeterogeneousKVCache& cache, std::uint32_t first,
                        std::int32_t current_token,
                        std::span<const std::int32_t> draft_tokens,
                        std::uint32_t remaining_outputs,
                        const Tensor& target_hidden_state,
                        WorkspaceArena& activations, cudaStream_t stream) {
        if (!executable_.ready()) {
            throw std::logic_error("Gemma MTP verify graph was not captured");
        }
        if (draft_tokens.size() != draft_width_) {
            throw std::invalid_argument("Gemma MTP verify graph replay width changed");
        }
        auto activation_scope = activations.scope();
        const std::int32_t token_count = static_cast<std::int32_t>(draft_width_ + 1);
        ModelBuffers buffers = allocate_model_buffers(activations, token_count, true);
        std::vector<std::int32_t> tokens;
        tokens.reserve(static_cast<std::size_t>(token_count));
        tokens.push_back(current_token);
        tokens.insert(tokens.end(), draft_tokens.begin(), draft_tokens.end());
        upload_decode_parameters(buffers, first, tokens, stream);
        auto transaction = cache.begin_transaction(
            0, first, static_cast<std::uint32_t>(token_count), stream);
        executable_.launch(stream);
        ++replay_count_;
        return finalize_mtp_verify(buffers, transaction, draft_tokens,
                                   remaining_outputs, target_hidden_state, stream);
    }

    [[nodiscard]] bool ready() const noexcept { return executable_.ready(); }
    [[nodiscard]] std::size_t device_bytes() const noexcept { return device_bytes_; }
    [[nodiscard]] std::uint32_t capture_count() const noexcept { return capture_count_; }
    [[nodiscard]] std::uint32_t replay_count() const noexcept { return replay_count_; }

private:
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable executable_;
    std::uint32_t draft_width_ = 0;
    std::size_t device_bytes_ = 0;
    std::uint32_t capture_count_ = 0;
    std::uint32_t replay_count_ = 0;
};

double nearest_rank_percentile(std::vector<double> values, std::uint32_t percentile) {
    if (values.empty() || percentile == 0 || percentile > 100) { return 0.0; }
    std::sort(values.begin(), values.end());
    const std::size_t rank =
        (static_cast<std::size_t>(percentile) * values.size() + 99U) / 100U;
    return values[std::max<std::size_t>(rank, 1U) - 1U];
}

} // namespace

std::size_t runtime_activation_workspace_capacity_bytes(std::int32_t tokens,
                                                        bool all_logits) {
    if (tokens <= 0 || tokens > static_cast<std::int32_t>(kMaximumChunkTokens)) {
        throw std::invalid_argument("Gemma activation workspace token count must be in [1,2048]");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_model_buffers(layout, tokens, all_logits);
    return layout.peak_bytes();
}

PersistentRunResult run_persistent_target(const std::filesystem::path& artifact_path,
                                          const PersistentRunOptions& options) {
    const std::uint32_t prefill_tokens = options.input_tokens.empty()
                                             ? options.prefill_tokens
                                             : static_cast<std::uint32_t>(options.input_tokens.size());
    const std::uint32_t appended_tokens = options.generation_tokens > 0
                                              ? options.generation_tokens - 1
                                              : static_cast<std::uint32_t>(options.run_deep_decode);
    const bool use_mtp = options.mtp_draft_tokens != 0;
    if (options.maximum_context < TextConfig::sliding_window ||
        options.maximum_context > TextConfig::maximum_position ||
        prefill_tokens == 0 || options.chunk_tokens == 0 ||
        options.chunk_tokens > kMaximumChunkTokens ||
        prefill_tokens + appended_tokens > options.maximum_context ||
        (options.generation_tokens > 0 && options.run_deep_decode) ||
        (options.qualify_graph_transactions &&
         (!options.use_cuda_graph || appended_tokens == 0)) ||
        options.mtp_draft_tokens > AssistantConfig::maximum_draft_size ||
        (use_mtp &&
         (options.generation_tokens < 2 || options.qualify_graph_transactions)) ||
        (options.save_continuation && !options.restore_continuation.empty()) ||
        (!options.save_continuation && !options.continuation_anchors.empty()) ||
        (options.save_continuation && prefill_tokens < 2) ||
        options.input_token < 0 ||
        options.input_token >= static_cast<std::int32_t>(TextConfig::vocabulary)) {
        throw std::invalid_argument("Gemma persistent target options are invalid");
    }
    if (std::any_of(options.input_tokens.begin(), options.input_tokens.end(), [](std::int32_t token) {
            return token < 0 || token >= static_cast<std::int32_t>(TextConfig::vocabulary);
        })) {
        throw std::invalid_argument("Gemma persistent target input token is invalid");
    }
    std::vector<std::uint32_t> continuation_anchors = options.continuation_anchors;
    std::sort(continuation_anchors.begin(), continuation_anchors.end());
    if (std::adjacent_find(continuation_anchors.begin(), continuation_anchors.end()) !=
            continuation_anchors.end() ||
        std::any_of(continuation_anchors.begin(), continuation_anchors.end(),
                    [prefill_tokens](std::uint32_t anchor) {
                        return anchor == 0 || anchor >= prefill_tokens - 1;
                    })) {
        throw std::invalid_argument("Gemma continuation anchors are invalid");
    }
    std::vector<std::int32_t> ledger;
    if (options.input_tokens.empty()) {
        ledger.assign(prefill_tokens, options.input_token);
    } else {
        ledger = options.input_tokens;
    }

    DeviceContext device(options.device_id);
    artifact::Reader reader(artifact_path);
    std::size_t target_weights_bytes = 0;
    if (use_mtp) {
        artifact::Binder target_binder(reader);
        target_weights_bytes =
            bind_artifact(target_binder, false).materialization.device_capacity_bytes;
    }
    artifact::Binder binder(reader);
    ArtifactLoadPlan plan = bind_artifact(binder, use_mtp);
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, plan.materialization, device);
    const ModelWeights weights = load_weights(materialized, plan.bindings);
    const std::optional<AssistantWeights> assistant_weights =
        use_mtp
            ? std::optional<AssistantWeights>(load_assistant_weights(
                  materialized, plan.bindings.assistant.value()))
            : std::nullopt;
    if (!options.dump_directory.empty()) {
        std::filesystem::create_directories(options.dump_directory);
    }

    PersistentRunResult result;
    result.weights_bytes = use_mtp ? target_weights_bytes
                                            : materialized.stats().device_capacity_bytes;
    result.assistant_weights_bytes =
        materialized.stats().device_capacity_bytes - result.weights_bytes;
    result.free_after_weights = free_device_bytes();
    result.free_after_assistant = result.free_after_weights;
    result.mtp_draft_width = options.mtp_draft_tokens;
    result.mtp_accepted_per_position.assign(options.mtp_draft_tokens, 0);

    const std::uint32_t maximum_transaction_tokens =
        std::max(options.chunk_tokens, use_mtp ? options.mtp_draft_tokens + 1U : 1U);

    const auto specs = make_text_kv_group_specs({
        .maximum_context = options.maximum_context,
        .maximum_transaction_tokens = maximum_transaction_tokens,
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
            static_cast<std::int32_t>(maximum_transaction_tokens));
    std::size_t activation_bytes = runtime_activation_workspace_capacity_bytes(
        static_cast<std::int32_t>(options.chunk_tokens), false);
    if (use_mtp) {
        activation_bytes = std::max(
            activation_bytes,
            runtime_activation_workspace_capacity_bytes(
                static_cast<std::int32_t>(options.mtp_draft_tokens + 1U), true));
    }
    DeviceBuffer activation_backing(activation_bytes);
    DeviceBuffer attention_backing(std::max<std::size_t>(attention_bytes, 256));
    WorkspaceArena activations({activation_backing.p, activation_backing.bytes});
    WorkspaceArena attention_workspace({attention_backing.p, attention_backing.bytes});
    result.workspace_bytes = activation_backing.bytes + attention_backing.bytes;
    result.largest_temporary_bytes =
        std::max(activation_backing.bytes, attention_backing.bytes);
    result.free_after_workspace = free_device_bytes();
    result.free_after_graph = result.free_after_workspace;
    std::optional<DeviceBuffer> target_hidden_backing;
    std::optional<DeviceBuffer> assistant_feedback_backing;
    Tensor target_hidden_state;
    Tensor assistant_feedback_state;
    if (use_mtp) {
        target_hidden_backing.emplace(
            static_cast<std::size_t>(TextConfig::hidden) * dtype_size(DType::BF16));
        assistant_feedback_backing.emplace(
            static_cast<std::size_t>(TextConfig::hidden) * dtype_size(DType::BF16));
        target_hidden_state = Tensor(target_hidden_backing->p, DType::BF16,
                                     {static_cast<std::int32_t>(TextConfig::hidden)});
        assistant_feedback_state = Tensor(
            assistant_feedback_backing->p, DType::BF16,
            {static_cast<std::int32_t>(TextConfig::hidden)});
        result.mtp_workspace_bytes =
            target_hidden_backing->bytes + assistant_feedback_backing->bytes;
    }

    EventPair timer;
    std::uint32_t first = 0;
    if (!options.restore_continuation.empty()) {
        const auto restore_start = std::chrono::steady_clock::now();
        const auto state = targets::gemma4::detail::decode_continuation(
            options.restore_continuation, kContinuationModelBinding,
            Package::artifact_compatibility_fingerprint, specs,
            static_cast<std::int32_t>(TextConfig::vocabulary), kSlidingKvGroup,
            kGlobalKvGroup);
        auto selection = targets::gemma4::detail::select_continuation_restore(
            state, ledger, specs, kSlidingKvGroup, kGlobalKvGroup);
        if (selection.frontier != 0) {
            cache.import_host(0, selection.kv, device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            first = selection.frontier;
            result.restored_tokens = first;
        }
        result.continuation_restore_milliseconds = elapsed_milliseconds(restore_start);
    }

    const std::uint32_t endpoint_frontier = options.save_continuation ? prefill_tokens - 1 : 0;
    std::vector<std::uint32_t> capture_frontiers = continuation_anchors;
    if (options.save_continuation) capture_frontiers.push_back(endpoint_frontier);
    std::vector<HeterogeneousKVHostImage> captured_anchors;
    captured_anchors.reserve(continuation_anchors.size());
    HeterogeneousKVHostImage captured_endpoint;

    timer.start(device.stream);
    while (first < prefill_tokens) {
        std::uint32_t next = std::min(first + options.chunk_tokens, prefill_tokens);
        if (options.save_continuation) {
            const auto capture = std::upper_bound(capture_frontiers.begin(),
                                                  capture_frontiers.end(), first);
            if (capture != capture_frontiers.end()) next = std::min(next, *capture);
        }
        const std::int32_t count = static_cast<std::int32_t>(next - first);
        const bool produce_logits = (options.generation_tokens > 0 || !options.run_deep_decode) &&
                                    first + static_cast<std::uint32_t>(count) ==
                                        prefill_tokens;
        const auto chunk = std::span<const std::int32_t>(ledger).subspan(first, count);
        const std::int32_t greedy = execute_chunk(
            weights, cache, 0, first, chunk, options.maximum_context, activations,
            attention_workspace, device.stream, produce_logits,
            produce_logits ? target_hidden_state : Tensor{},
            produce_logits ? options.dump_directory : std::filesystem::path{});
        if (produce_logits) result.greedy_token = greedy;
        first += static_cast<std::uint32_t>(count);
        result.computed_prefill_tokens += static_cast<std::uint32_t>(count);
        if (options.save_continuation &&
            std::binary_search(capture_frontiers.begin(), capture_frontiers.end(), first)) {
            const auto save_start = std::chrono::steady_clock::now();
            HeterogeneousKVHostImage image = cache.export_host(0, device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            result.continuation_save_milliseconds += elapsed_milliseconds(save_start);
            if (first == endpoint_frontier) {
                captured_endpoint = std::move(image);
            } else {
                captured_anchors.push_back(std::move(image));
            }
        }
    }
    result.prefill_milliseconds = timer.stop(device.stream);

    if (options.save_continuation) {
        const auto save_start = std::chrono::steady_clock::now();
        const auto state = targets::gemma4::detail::make_continuation_state(
            ledger, std::move(captured_endpoint), std::move(captured_anchors),
            kSlidingKvGroup, kGlobalKvGroup);
        result.continuation_payload_bytes = state.endpoint.payload_bytes();
        for (const auto& anchor : state.anchors) {
            result.continuation_payload_bytes += anchor.sliding.payload.size();
        }
        result.continuation_anchor_count =
            static_cast<std::uint32_t>(state.anchors.size());
        result.continuation_snapshot = targets::gemma4::detail::encode_continuation(
            state, kContinuationModelBinding, Package::artifact_compatibility_fingerprint,
            specs, kSlidingKvGroup, kGlobalKvGroup);
        result.continuation_save_milliseconds += elapsed_milliseconds(save_start);
    }

    PersistentDecodeGraph decode_graph;
    PersistentMtpVerifyGraph mtp_verify_graph;
    if (options.use_cuda_graph && appended_tokens > 0) {
        if (use_mtp &&
            options.mtp_draft_tokens == AssistantConfig::production_draft_size) {
            if (first + options.mtp_draft_tokens + 1U <= options.maximum_context) {
                mtp_verify_graph.capture(weights, cache, options.maximum_context,
                                         options.mtp_draft_tokens, activations,
                                         attention_workspace, device.stream);
                result.graph_device_bytes = mtp_verify_graph.device_bytes();
                result.graph_capture_count = mtp_verify_graph.capture_count();
            }
        } else {
            decode_graph.capture(weights, cache, options.maximum_context, activations,
                                 attention_workspace, device.stream);
            result.graph_device_bytes = decode_graph.device_bytes();
            result.graph_capture_count = decode_graph.capture_count();
        }
        result.free_after_graph = free_device_bytes();
    }

    std::int32_t rollback_greedy = -1;
    if (options.qualify_graph_transactions) {
        const std::int32_t token = options.generation_tokens > 0
                                       ? result.greedy_token
                                       : options.input_token;
        rollback_greedy = decode_graph.replay(cache, first, token, activations,
                                              device.stream, false);
        if (cache.frontier(0) != first) {
            throw std::runtime_error("Gemma graph rollback published its transaction");
        }
    }

    const auto decode_one = [&](std::int32_t token) {
        if (options.use_cuda_graph) {
            return decode_graph.replay(cache, first, token, activations, device.stream, true);
        }
        return execute_chunk(weights, cache, 0, first,
                             std::span<const std::int32_t>(&token, 1),
                             options.maximum_context, activations, attention_workspace,
                             device.stream, true);
    };

    if (use_mtp) {
        result.generated_tokens.push_back(result.greedy_token);
        EventPair assistant_timer;
        EventPair verify_timer;
        std::vector<double> round_milliseconds;
        timer.start(device.stream);
        while (result.generated_tokens.size() < options.generation_tokens) {
            const std::int32_t current_token = result.generated_tokens.back();
            const std::uint32_t remaining = options.generation_tokens -
                                            static_cast<std::uint32_t>(
                                                result.generated_tokens.size());
            const std::uint32_t draft_count =
                std::min(options.mtp_draft_tokens, remaining);
            std::vector<std::int32_t> draft_tokens;
            draft_tokens.reserve(draft_count);
            const Tensor* input_hidden = &target_hidden_state;
            const double assistant_before = result.assistant_milliseconds;
            const double verify_before = result.verify_milliseconds;
            for (std::uint32_t draft = 0; draft < draft_count; ++draft) {
                const std::int32_t input_token =
                    draft == 0 ? current_token : draft_tokens.back();
                assistant_timer.start(device.stream);
                const AssistantStepResult step = execute_assistant(
                    weights, *assistant_weights, cache, 0, input_token, first,
                    options.maximum_context, *input_hidden, assistant_feedback_state,
                    activations, attention_workspace, device.stream);
                result.assistant_milliseconds += assistant_timer.stop(device.stream);
                result.proposal_head_milliseconds += step.proposal_head_milliseconds;
                draft_tokens.push_back(step.token);
                input_hidden = &assistant_feedback_state;
            }
            if (result.mtp_rounds == 0) {
                result.mtp_first_round_drafts = draft_tokens;
                result.mtp_first_draft_token = draft_tokens.front();
            }
            result.mtp_proposed_tokens += draft_count;

            verify_timer.start(device.stream);
            const VerifyResult verified =
                mtp_verify_graph.ready() && draft_count == options.mtp_draft_tokens
                    ? mtp_verify_graph.replay(
                          cache, first, current_token, draft_tokens, remaining,
                          target_hidden_state, activations, device.stream)
                    : execute_mtp_verify(
                          weights, cache, 0, first, current_token, draft_tokens,
                          options.maximum_context, remaining, target_hidden_state,
                          activations, attention_workspace, device.stream);
            result.verify_milliseconds += verify_timer.stop(device.stream);
            const std::uint32_t accepted_outputs =
                std::min(verified.accepted_drafts, remaining);
            result.mtp_accepted_tokens += accepted_outputs;
            for (std::uint32_t accepted = 0; accepted < accepted_outputs; ++accepted) {
                ++result.mtp_accepted_per_position[accepted];
                result.generated_tokens.push_back(draft_tokens[accepted]);
            }
            if (accepted_outputs < remaining) {
                result.generated_tokens.push_back(verified.greedy[accepted_outputs]);
            }
            first += verified.committed_tokens;
            result.greedy_token = result.generated_tokens.back();
            ++result.mtp_rounds;
            round_milliseconds.push_back(
                (result.assistant_milliseconds - assistant_before) +
                (result.verify_milliseconds - verify_before));
        }
        result.decode_milliseconds = timer.stop(device.stream);
        result.mtp_round_p50_milliseconds = nearest_rank_percentile(round_milliseconds, 50);
        result.mtp_round_p95_milliseconds = nearest_rank_percentile(round_milliseconds, 95);
    } else if (options.generation_tokens > 0) {
        result.generated_tokens.push_back(result.greedy_token);
        timer.start(device.stream);
        while (result.generated_tokens.size() < options.generation_tokens) {
            const std::int32_t token = result.generated_tokens.back();
            result.greedy_token = decode_one(token);
            if (rollback_greedy >= 0) {
                if (result.greedy_token != rollback_greedy) {
                    throw std::runtime_error(
                        "Gemma graph rollback replay changed its output");
                }
                rollback_greedy = -1;
                result.graph_transaction_checks_passed = true;
            }
            result.generated_tokens.push_back(result.greedy_token);
            ++first;
        }
        result.decode_milliseconds = timer.stop(device.stream);
    } else if (options.run_deep_decode) {
        timer.start(device.stream);
        result.greedy_token = decode_one(options.input_token);
        result.decode_milliseconds = timer.stop(device.stream);
        if (rollback_greedy >= 0) {
            if (result.greedy_token != rollback_greedy) {
                throw std::runtime_error("Gemma graph rollback replay changed its output");
            }
            result.graph_transaction_checks_passed = true;
        }
    }
    result.graph_capture_count = decode_graph.capture_count() + mtp_verify_graph.capture_count();
    result.graph_replay_count = decode_graph.replay_count() + mtp_verify_graph.replay_count();
    result.final_frontier = cache.frontier(0);
    return result;
}

std::vector<float> execute_runtime_chunk(
    const ModelWeights& weights, HeterogeneousKVCache& cache, std::int32_t row,
    std::uint32_t first, std::span<const std::int32_t> input_tokens,
    std::uint32_t maximum_context, WorkspaceArena& activations,
    WorkspaceArena& attention_workspace, cudaStream_t stream, bool produce_logits,
    const Tensor& final_hidden) {
    const std::int32_t tokens = static_cast<std::int32_t>(input_tokens.size());
    auto activation_scope = activations.scope();
    ModelBuffers buffers = allocate_model_buffers(activations, tokens);
    upload_decode_parameters(buffers, first, input_tokens, stream);
    auto transaction = cache.begin_transaction(row, first, static_cast<std::uint32_t>(tokens), stream);
    const ops::CausalAttentionExecutionEnvelope envelope =
        tokens == 1 ? ops::CausalAttentionExecutionEnvelope{1, maximum_context}
                    : ops::CausalAttentionExecutionEnvelope{
                          first + 1U, first + static_cast<std::uint32_t>(tokens)};
    enqueue_model(weights, transaction, buffers, envelope, attention_workspace, stream,
                  produce_logits);
    if (produce_logits && final_hidden.data != nullptr) {
        const Tensor selected =
            buffers.final_hidden.slice(1, tokens - 1, 1).view({TextConfig::hidden});
        CUDA_CHECK(cudaMemcpyAsync(final_hidden.data, selected.data, selected.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    transaction.commit(stream);
    if (!produce_logits) { return {}; }
    std::vector<float> logits(TextConfig::vocabulary);
    CUDA_CHECK(cudaMemcpyAsync(logits.data(), buffers.logits.data,
                               logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return logits;
}

class RuntimeDecodeGraph::Impl {
public:
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    std::int32_t row = -1;
    std::size_t device_bytes = 0;
};

RuntimeDecodeGraph::RuntimeDecodeGraph() : impl_(std::make_unique<Impl>()) {}
RuntimeDecodeGraph::~RuntimeDecodeGraph() = default;
RuntimeDecodeGraph::RuntimeDecodeGraph(RuntimeDecodeGraph&&) noexcept = default;
RuntimeDecodeGraph& RuntimeDecodeGraph::operator=(RuntimeDecodeGraph&&) noexcept = default;

void RuntimeDecodeGraph::capture(const ModelWeights& weights, HeterogeneousKVCache& cache,
                                 std::int32_t row, std::uint32_t maximum_context,
                                 WorkspaceArena& activations,
                                 WorkspaceArena& attention_workspace, cudaStream_t stream,
                                 const Tensor& final_hidden) {
    if (impl_->executable.ready()) {
        throw std::logic_error("Gemma runtime decode graph cannot be recaptured");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t free_before = free_device_bytes();
    {
        auto activation_scope = activations.scope();
        ModelBuffers buffers = allocate_model_buffers(activations, 1);
        const std::int32_t token = 0;
        const std::uint32_t first = cache.frontier(row);
        upload_decode_parameters(buffers, first,
                                 std::span<const std::int32_t>(&token, 1), stream);
        auto transaction = cache.begin_transaction(row, first, 1, stream);
        impl_->definition.capture(stream, [&] {
            enqueue_model(weights, transaction, buffers, {1, maximum_context},
                          attention_workspace, stream, true, false);
            if (final_hidden.data != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(final_hidden.data, buffers.final_hidden.data,
                                           final_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                           stream));
            }
        });
        transaction.rollback();
    }
    impl_->executable.instantiate(impl_->definition);
    impl_->executable.upload(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t free_after = free_device_bytes();
    impl_->device_bytes = free_before > free_after ? free_before - free_after : 0;
    impl_->row = row;
}

std::vector<float> RuntimeDecodeGraph::replay(HeterogeneousKVCache& cache,
                                              std::int32_t row, std::uint32_t first,
                                              std::int32_t input_token,
                                              WorkspaceArena& activations,
                                              cudaStream_t stream) {
    if (!impl_->executable.ready() || row != impl_->row) {
        throw std::logic_error("Gemma runtime decode graph row is not captured");
    }
    auto activation_scope = activations.scope();
    ModelBuffers buffers = allocate_model_buffers(activations, 1);
    upload_decode_parameters(buffers, first,
                             std::span<const std::int32_t>(&input_token, 1), stream);
    auto transaction = cache.begin_transaction(row, first, 1, stream);
    impl_->executable.launch(stream);
    transaction.commit(stream);
    std::vector<float> logits(TextConfig::vocabulary);
    CUDA_CHECK(cudaMemcpyAsync(logits.data(), buffers.logits.data,
                               logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return logits;
}

bool RuntimeDecodeGraph::ready() const noexcept { return impl_->executable.ready(); }
std::size_t RuntimeDecodeGraph::device_bytes() const noexcept { return impl_->device_bytes; }

class RuntimeMtpVerifyGraph::Impl {
public:
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    std::int32_t row = -1;
    std::uint32_t draft_width = 0;
    std::size_t device_bytes = 0;
};

RuntimeMtpVerifyGraph::RuntimeMtpVerifyGraph() : impl_(std::make_unique<Impl>()) {}
RuntimeMtpVerifyGraph::~RuntimeMtpVerifyGraph() = default;
RuntimeMtpVerifyGraph::RuntimeMtpVerifyGraph(RuntimeMtpVerifyGraph&&) noexcept = default;
RuntimeMtpVerifyGraph& RuntimeMtpVerifyGraph::operator=(RuntimeMtpVerifyGraph&&) noexcept = default;

void RuntimeMtpVerifyGraph::capture(const ModelWeights& weights,
                                    HeterogeneousKVCache& cache, std::int32_t row,
                                    std::uint32_t maximum_context,
                                    std::uint32_t draft_width,
                                    WorkspaceArena& activations,
                                    WorkspaceArena& attention_workspace,
                                    cudaStream_t stream) {
    if (impl_->executable.ready() || draft_width == 0 ||
        draft_width > AssistantConfig::maximum_draft_size) {
        throw std::logic_error("Gemma runtime MTP graph has invalid capture state");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t free_before = free_device_bytes();
    {
        auto activation_scope = activations.scope();
        const std::int32_t token_count = static_cast<std::int32_t>(draft_width + 1U);
        ModelBuffers buffers = allocate_model_buffers(activations, token_count, true);
        const std::vector<std::int32_t> inputs(static_cast<std::size_t>(token_count), 0);
        const std::uint32_t first = cache.frontier(row);
        upload_decode_parameters(buffers, first, inputs, stream);
        auto transaction = cache.begin_transaction(
            row, first, static_cast<std::uint32_t>(token_count), stream);
        impl_->definition.capture(stream, [&] {
            enqueue_model(weights, transaction, buffers, {1, maximum_context},
                          attention_workspace, stream, true, true);
        });
        transaction.rollback();
    }
    impl_->executable.instantiate(impl_->definition);
    impl_->executable.upload(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t free_after = free_device_bytes();
    impl_->device_bytes = free_before > free_after ? free_before - free_after : 0;
    impl_->row = row;
    impl_->draft_width = draft_width;
}

RuntimeMtpRound RuntimeMtpVerifyGraph::replay(
    HeterogeneousKVCache& cache, std::int32_t row, std::uint32_t first,
    std::int32_t current_token, std::span<const std::int32_t> drafts,
    std::uint32_t remaining_outputs, const Tensor& target_hidden_state,
    WorkspaceArena& activations, cudaStream_t stream,
    const RuntimeMtpSampler& sample_target) {
    if (!impl_->executable.ready() || row != impl_->row ||
        drafts.size() != impl_->draft_width || remaining_outputs == 0) {
        throw std::logic_error("Gemma runtime MTP graph replay shape changed");
    }
    auto activation_scope = activations.scope();
    const std::int32_t token_count = static_cast<std::int32_t>(drafts.size() + 1U);
    ModelBuffers buffers = allocate_model_buffers(activations, token_count, true);
    std::vector<std::int32_t> inputs;
    inputs.reserve(static_cast<std::size_t>(token_count));
    inputs.push_back(current_token);
    inputs.insert(inputs.end(), drafts.begin(), drafts.end());
    upload_decode_parameters(buffers, first, inputs, stream);
    auto transaction = cache.begin_transaction(
        row, first, static_cast<std::uint32_t>(token_count), stream);
    impl_->executable.launch(stream);
    std::vector<float> host_logits(static_cast<std::size_t>(TextConfig::vocabulary) *
                                   static_cast<std::size_t>(token_count));
    CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), buffers.logits.data,
                               host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    RuntimeMtpRound result;
    result.drafted_tokens = static_cast<std::uint32_t>(drafts.size());
    for (std::uint32_t output = 0; output < remaining_outputs; ++output) {
        const auto column = std::span<const float>(host_logits).subspan(
            static_cast<std::size_t>(output) * TextConfig::vocabulary,
            TextConfig::vocabulary);
        const std::int32_t target = sample_target
                                        ? sample_target(column, output)
                                        : static_cast<std::int32_t>(
                                              std::max_element(column.begin(), column.end()) -
                                              column.begin());
        if (output < drafts.size() && target == drafts[output]) {
            result.output_tokens.push_back(drafts[output]);
            ++result.accepted_drafts;
            continue;
        }
        result.output_tokens.push_back(target);
        break;
    }
    const std::uint32_t produced = static_cast<std::uint32_t>(result.output_tokens.size());
    const Tensor selected =
        buffers.final_hidden.slice(1, produced - 1U, 1).view({TextConfig::hidden});
    CUDA_CHECK(cudaMemcpyAsync(target_hidden_state.data, selected.data, selected.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
    result.transaction = std::move(transaction);
    return result;
}

bool RuntimeMtpVerifyGraph::ready() const noexcept { return impl_->executable.ready(); }
std::size_t RuntimeMtpVerifyGraph::device_bytes() const noexcept { return impl_->device_bytes; }

RuntimeMtpRound execute_runtime_mtp_round(
    const ModelWeights& target_weights, const AssistantWeights& assistant_weights,
    HeterogeneousKVCache& cache, std::int32_t row, std::uint32_t first,
    std::int32_t current_token, std::uint32_t draft_count, std::uint32_t maximum_context,
    std::uint32_t remaining_outputs, const Tensor& target_hidden_state,
    Tensor assistant_feedback_state, WorkspaceArena& activations,
    WorkspaceArena& attention_workspace, cudaStream_t stream,
    const RuntimeMtpSampler& sample_target, RuntimeMtpVerifyGraph* verify_graph) {
    if (draft_count == 0 || draft_count > AssistantConfig::maximum_draft_size ||
        remaining_outputs == 0 || target_hidden_state.data == nullptr ||
        assistant_feedback_state.data == nullptr) {
        throw std::invalid_argument("Gemma runtime MTP round has invalid extents");
    }
    draft_count = std::min(draft_count, remaining_outputs);
    std::vector<std::int32_t> drafts;
    drafts.reserve(draft_count);
    const Tensor* input_hidden = &target_hidden_state;
    for (std::uint32_t draft = 0; draft < draft_count; ++draft) {
        const std::int32_t input = draft == 0 ? current_token : drafts.back();
        const AssistantStepResult step = execute_assistant(
            target_weights, assistant_weights, cache, row, input, first, maximum_context,
            *input_hidden, assistant_feedback_state, activations, attention_workspace, stream);
        drafts.push_back(step.token);
        input_hidden = &assistant_feedback_state;
    }

    if (verify_graph != nullptr && !verify_graph->ready()) {
        verify_graph->capture(target_weights, cache, row, maximum_context, draft_count,
                              activations, attention_workspace, stream);
    }
    if (verify_graph != nullptr && verify_graph->ready() &&
        draft_count == static_cast<std::uint32_t>(drafts.size())) {
        return verify_graph->replay(cache, row, first, current_token, drafts,
                                    remaining_outputs, target_hidden_state, activations,
                                    stream, sample_target);
    }

    auto activation_scope = activations.scope();
    const std::int32_t token_count = static_cast<std::int32_t>(drafts.size() + 1U);
    ModelBuffers buffers = allocate_model_buffers(activations, token_count, true);
    std::vector<std::int32_t> inputs;
    inputs.reserve(static_cast<std::size_t>(token_count));
    inputs.push_back(current_token);
    inputs.insert(inputs.end(), drafts.begin(), drafts.end());
    upload_decode_parameters(buffers, first, inputs, stream);
    auto transaction = cache.begin_transaction(row, first,
                                               static_cast<std::uint32_t>(token_count), stream);
    enqueue_model(target_weights, transaction, buffers, {1, maximum_context},
                  attention_workspace, stream, true, true);
    std::vector<float> host_logits(static_cast<std::size_t>(TextConfig::vocabulary) *
                                   static_cast<std::size_t>(token_count));
    CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), buffers.logits.data,
                               host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::uint32_t accepted = 0;
    RuntimeMtpRound result;
    result.drafted_tokens = draft_count;
    for (std::uint32_t output = 0; output < remaining_outputs; ++output) {
        const auto column = std::span<const float>(host_logits).subspan(
            static_cast<std::size_t>(output) * TextConfig::vocabulary,
            TextConfig::vocabulary);
        const std::int32_t target = sample_target
                                        ? sample_target(column, output)
                                        : static_cast<std::int32_t>(
                                              std::max_element(column.begin(), column.end()) -
                                              column.begin());
        if (output < drafts.size() && target == drafts[output]) {
            result.output_tokens.push_back(drafts[output]);
            ++accepted;
            continue;
        }
        result.output_tokens.push_back(target);
        break;
    }
    result.accepted_drafts = accepted;
    const std::uint32_t produced = static_cast<std::uint32_t>(result.output_tokens.size());
    const Tensor selected =
        buffers.final_hidden.slice(1, produced - 1U, 1).view({TextConfig::hidden});
    CUDA_CHECK(cudaMemcpyAsync(target_hidden_state.data, selected.data, selected.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
    result.transaction = std::move(transaction);
    return result;
}

} // namespace ninfer::targets::gemma4_31b_it::detail
