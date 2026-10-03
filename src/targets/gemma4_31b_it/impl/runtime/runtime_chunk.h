#pragma once

#include "core/arena.h"
#include "core/heterogeneous_kv_cache.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer {
class HeterogeneousKVCache;
}

namespace ninfer::targets::gemma4_31b_it::detail {

struct ModelWeights;
struct AssistantWeights;

// Exact peak activation storage for one target-model execution. Startup planning uses the same
// allocation recipe as execution so larger prefill chunks cannot drift from the real arena.
[[nodiscard]] std::size_t runtime_activation_workspace_capacity_bytes(
    std::int32_t tokens, bool all_logits = false);

struct RuntimeMtpRound {
    std::vector<std::int32_t> output_tokens;
    std::uint32_t drafted_tokens = 0;
    std::uint32_t accepted_drafts = 0;
    HeterogeneousKVTransaction transaction;
};

using RuntimeMtpSampler =
    std::function<std::int32_t(std::span<const float> logits, std::uint32_t output_offset)>;

// One row-specific, shape-stable decode graph. Capture is lazy so the row already has a valid
// heterogeneous-cache execution view, and replay retains the Program's ordinary transaction
// semantics: the consumed input token is committed before its sampled successor is published.
class RuntimeDecodeGraph {
public:
    RuntimeDecodeGraph();
    ~RuntimeDecodeGraph();
    RuntimeDecodeGraph(const RuntimeDecodeGraph&) = delete;
    RuntimeDecodeGraph& operator=(const RuntimeDecodeGraph&) = delete;
    RuntimeDecodeGraph(RuntimeDecodeGraph&&) noexcept;
    RuntimeDecodeGraph& operator=(RuntimeDecodeGraph&&) noexcept;

    void capture(const ModelWeights& weights, HeterogeneousKVCache& cache, std::int32_t row,
                 std::uint32_t maximum_context, WorkspaceArena& activations,
                 WorkspaceArena& attention_workspace, cudaStream_t stream,
                 const Tensor& final_hidden = {});
    [[nodiscard]] std::vector<float> replay(
        HeterogeneousKVCache& cache, std::int32_t row, std::uint32_t first,
        std::int32_t input_token, WorkspaceArena& activations, cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] std::size_t device_bytes() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Row-specific graph for the fixed-width target verification half of MTP. Assistant proposal
// remains eager; a short final round whose width is smaller than the captured width also uses the
// eager verifier. The returned KV transaction is intentionally unpublished until Engine commits
// the Frontend-accepted prefix.
class RuntimeMtpVerifyGraph {
public:
    RuntimeMtpVerifyGraph();
    ~RuntimeMtpVerifyGraph();
    RuntimeMtpVerifyGraph(const RuntimeMtpVerifyGraph&) = delete;
    RuntimeMtpVerifyGraph& operator=(const RuntimeMtpVerifyGraph&) = delete;
    RuntimeMtpVerifyGraph(RuntimeMtpVerifyGraph&&) noexcept;
    RuntimeMtpVerifyGraph& operator=(RuntimeMtpVerifyGraph&&) noexcept;

    void capture(const ModelWeights& weights, HeterogeneousKVCache& cache, std::int32_t row,
                 std::uint32_t maximum_context, std::uint32_t draft_width,
                 WorkspaceArena& activations, WorkspaceArena& attention_workspace,
                 cudaStream_t stream);
    [[nodiscard]] RuntimeMtpRound replay(
        HeterogeneousKVCache& cache, std::int32_t row, std::uint32_t first,
        std::int32_t current_token, std::span<const std::int32_t> drafts,
        std::uint32_t remaining_outputs, const Tensor& target_hidden_state,
        WorkspaceArena& activations, cudaStream_t stream,
        const RuntimeMtpSampler& sample_target);
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] std::size_t device_bytes() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Production execution leaf shared by the public Program and the phase qualification runner.
// It publishes the target KV transaction before returning and returns the complete FP32 logits
// only when requested. The caller owns sampling and output publication.
[[nodiscard]] std::vector<float> execute_runtime_chunk(
    const ModelWeights& weights, HeterogeneousKVCache& cache, std::int32_t row,
    std::uint32_t first, std::span<const std::int32_t> input_tokens,
    std::uint32_t maximum_context, WorkspaceArena& activations,
    WorkspaceArena& attention_workspace, cudaStream_t stream, bool produce_logits,
    const Tensor& final_hidden = {});

// Executes one Gemma MTP proposal/target-verify round without publishing its KV suffix.
// The caller commits the Frontend-licensed prefix of transaction after output interpretation.
[[nodiscard]] RuntimeMtpRound execute_runtime_mtp_round(
    const ModelWeights& target_weights, const AssistantWeights& assistant_weights,
    HeterogeneousKVCache& cache, std::int32_t row, std::uint32_t first,
    std::int32_t current_token, std::uint32_t draft_tokens, std::uint32_t maximum_context,
    std::uint32_t remaining_outputs, const Tensor& target_hidden_state,
    Tensor assistant_feedback_state, WorkspaceArena& activations,
    WorkspaceArena& attention_workspace, cudaStream_t stream,
    const RuntimeMtpSampler& sample_target, RuntimeMtpVerifyGraph* verify_graph = nullptr);

} // namespace ninfer::targets::gemma4_31b_it::detail
