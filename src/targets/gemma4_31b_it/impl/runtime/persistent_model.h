#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ninfer::targets::gemma4_31b_it::detail {

struct PersistentRunOptions {
    std::uint32_t maximum_context = 262144;
    std::uint32_t prefill_tokens = 0;
    std::uint32_t chunk_tokens = 64;
    std::int32_t input_token = 2;
    std::vector<std::int32_t> input_tokens;
    std::uint32_t generation_tokens = 0;
    std::filesystem::path dump_directory;
    bool run_deep_decode = true;
    bool use_cuda_graph = false;
    bool qualify_graph_transactions = false;
    std::uint32_t mtp_draft_tokens = 0;
    bool save_continuation = false;
    std::vector<std::uint8_t> restore_continuation;
    std::vector<std::uint32_t> continuation_anchors;
    int device_id = 0;
};

struct PersistentRunResult {
    std::int32_t greedy_token = -1;
    std::size_t weights_bytes = 0;
    std::size_t kv_payload_bytes = 0;
    std::size_t kv_metadata_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::size_t free_after_weights = 0;
    std::size_t free_after_cache = 0;
    std::size_t free_after_workspace = 0;
    std::size_t free_after_graph = 0;
    std::size_t graph_device_bytes = 0;
    std::size_t assistant_weights_bytes = 0;
    std::size_t mtp_workspace_bytes = 0;
    std::size_t largest_temporary_bytes = 0;
    std::size_t free_after_assistant = 0;
    std::uint32_t final_frontier = 0;
    std::uint32_t graph_capture_count = 0;
    std::uint32_t graph_replay_count = 0;
    std::uint32_t restored_tokens = 0;
    std::uint32_t computed_prefill_tokens = 0;
    std::uint32_t continuation_anchor_count = 0;
    bool graph_transaction_checks_passed = false;
    bool graph_uses_existing_workspace = true;
    std::uint64_t mtp_proposed_tokens = 0;
    std::uint64_t mtp_accepted_tokens = 0;
    std::uint64_t mtp_rounds = 0;
    std::uint32_t mtp_draft_width = 0;
    std::int32_t mtp_first_draft_token = -1;
    std::vector<std::int32_t> mtp_first_round_drafts;
    std::vector<std::uint64_t> mtp_accepted_per_position;
    std::vector<std::int32_t> generated_tokens;
    std::vector<std::uint8_t> continuation_snapshot;
    std::size_t continuation_payload_bytes = 0;
    double prefill_milliseconds = 0.0;
    double decode_milliseconds = 0.0;
    double continuation_save_milliseconds = 0.0;
    double continuation_restore_milliseconds = 0.0;
    double assistant_milliseconds = 0.0;
    double proposal_head_milliseconds = 0.0;
    double verify_milliseconds = 0.0;
    double mtp_round_p50_milliseconds = 0.0;
    double mtp_round_p95_milliseconds = 0.0;
};

[[nodiscard]] PersistentRunResult run_persistent_target(
    const std::filesystem::path& artifact_path, const PersistentRunOptions& options);

} // namespace ninfer::targets::gemma4_31b_it::detail
