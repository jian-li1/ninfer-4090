#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ninfer::targets::gemma4_31b_it::detail {

struct ReferenceRunResult {
    std::vector<float> logits;
    std::int32_t greedy_token = -1;
};

// Correctness-first, target-owned execution route for Phase 4 qualification. It reruns a complete
// prefix and intentionally owns no KV state; the production heterogeneous cache is introduced by
// the following phase without coupling this mathematical oracle route to Qwen cache geometry.
[[nodiscard]] ReferenceRunResult run_reference_prefix(
    const std::filesystem::path& artifact_path, std::span<const std::int32_t> token_ids,
    const std::filesystem::path& dump_directory = {}, int device_id = 0);

} // namespace ninfer::targets::gemma4_31b_it::detail
