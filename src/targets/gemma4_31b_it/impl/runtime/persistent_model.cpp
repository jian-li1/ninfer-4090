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
#include "core/decode_graph.h"
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

ModelBuffers allocate_model_buffers(WorkspaceArena& activations, std::int32_t tokens) {
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
        .logits = activations.alloc(DType::FP32, {TextConfig::vocabulary}),
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
                   bool produce_logits,
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
        if (full) {
            ops::causal_full_softmax_attention(
                query, key, value, buffers.positions, kFullGeometry, 1.0F, view,
                full_envelope, attention_workspace, attended, stream);
        } else {
            ops::causal_sliding_softmax_attention(
                query, key, value, buffers.positions, kSlidingGeometry,
                TextConfig::sliding_window, 1.0F, view, attended, stream);
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
    Tensor last_hidden =
        buffers.final_hidden.slice(1, tokens - 1, 1).view({TextConfig::hidden});
    fp8_tied_logits(last_hidden, weights.embedding, TextConfig::final_logit_softcap,
                    buffers.logits, stream);
}

std::int32_t read_greedy_token(const Tensor& logits, cudaStream_t stream) {
    std::vector<float> host_logits(TextConfig::vocabulary);
    CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), logits.data,
                               host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return static_cast<std::int32_t>(
        std::max_element(host_logits.begin(), host_logits.end()) - host_logits.begin());
}

std::int32_t execute_chunk(const ModelWeights& weights, HeterogeneousKVCache& cache,
                           std::uint32_t first, std::span<const std::int32_t> input_tokens,
                           std::uint32_t maximum_context, WorkspaceArena& activations,
                           WorkspaceArena& attention_workspace, cudaStream_t stream,
                           bool produce_logits,
                           const std::filesystem::path& dump_directory = {}) {
    const std::int32_t tokens = static_cast<std::int32_t>(input_tokens.size());
    auto activation_scope = activations.scope();
    ModelBuffers buffers = allocate_model_buffers(activations, tokens);
    upload_decode_parameters(buffers, first, input_tokens, stream);
    auto transaction = cache.begin_transaction(0, first, static_cast<std::uint32_t>(tokens), stream);
    const ops::CausalAttentionExecutionEnvelope envelope =
        tokens == 1 ? ops::CausalAttentionExecutionEnvelope{1, maximum_context}
                    : ops::CausalAttentionExecutionEnvelope{
                          first + 1U, first + static_cast<std::uint32_t>(tokens)};
    enqueue_model(weights, transaction, buffers, envelope, attention_workspace, stream,
                  produce_logits, dump_directory);
    transaction.commit(stream);
    return produce_logits ? read_greedy_token(buffers.logits, stream) : -1;
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
                              stream, true);
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
        (options.qualify_graph_transactions &&
         (!options.use_cuda_graph || appended_tokens == 0)) ||
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
    result.largest_temporary_bytes =
        std::max(activation_backing.bytes, attention_backing.bytes);
    result.free_after_workspace = free_device_bytes();
    result.free_after_graph = result.free_after_workspace;

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
            weights, cache, first, chunk, options.maximum_context, activations,
            attention_workspace, device.stream, produce_logits,
            produce_logits ? options.dump_directory : std::filesystem::path{});
        if (produce_logits) result.greedy_token = greedy;
        first += static_cast<std::uint32_t>(count);
    }
    result.prefill_milliseconds = timer.stop(device.stream);

    PersistentDecodeGraph decode_graph;
    if (options.use_cuda_graph && appended_tokens > 0) {
        decode_graph.capture(weights, cache, options.maximum_context, activations,
                             attention_workspace, device.stream);
        result.graph_device_bytes = decode_graph.device_bytes();
        result.graph_capture_count = decode_graph.capture_count();
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
        return execute_chunk(weights, cache, first,
                             std::span<const std::int32_t>(&token, 1),
                             options.maximum_context, activations, attention_workspace,
                             device.stream, true);
    };

    if (options.generation_tokens > 0) {
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
    result.graph_capture_count = decode_graph.capture_count();
    result.graph_replay_count = decode_graph.replay_count();
    result.final_frontier = cache.frontier(0);
    return result;
}

} // namespace ninfer::targets::gemma4_31b_it::detail
