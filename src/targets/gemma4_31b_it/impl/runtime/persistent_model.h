#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace ninfer::targets::gemma4_31b_it::detail {

struct PersistentRunOptions {
    std::uint32_t maximum_context = 262144;
    std::uint32_t prefill_tokens = 0;
    std::uint32_t chunk_tokens = 64;
    std::int32_t input_token = 2;
    bool run_deep_decode = true;
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
    std::uint32_t final_frontier = 0;
    double prefill_milliseconds = 0.0;
    double decode_milliseconds = 0.0;
};

[[nodiscard]] PersistentRunResult run_persistent_target(
    const std::filesystem::path& artifact_path, const PersistentRunOptions& options);

} // namespace ninfer::targets::gemma4_31b_it::detail
