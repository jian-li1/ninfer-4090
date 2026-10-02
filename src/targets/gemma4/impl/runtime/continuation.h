#pragma once

#include "core/heterogeneous_kv_cache.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::targets::gemma4::detail {

struct ContinuationAnchorImage {
    std::uint32_t frontier = 0;
    HeterogeneousKVHostGroupImage sliding;
};

struct ContinuationState {
    std::vector<std::int32_t> ledger;
    std::uint32_t execution_frontier = 0;
    HeterogeneousKVHostImage endpoint;
    std::vector<ContinuationAnchorImage> anchors;
};

struct ContinuationRestoreSelection {
    std::uint32_t frontier = 0;
    bool endpoint = false;
    HeterogeneousKVHostImage kv;
};

// Captures the endpoint in full and retains only the sliding group for each older anchor. The
// endpoint global payload is prefix-addressable and therefore stored once for every anchor.
[[nodiscard]] ContinuationState make_continuation_state(
    std::span<const std::int32_t> ledger, HeterogeneousKVHostImage endpoint,
    std::vector<HeterogeneousKVHostImage> anchors, std::uint32_t sliding_group_id,
    std::uint32_t global_group_id);

[[nodiscard]] std::vector<std::uint8_t> encode_continuation(
    const ContinuationState& state, std::string_view model_binding,
    std::string_view artifact_fingerprint, std::span<const KvGroupSpec> groups,
    std::uint32_t sliding_group_id, std::uint32_t global_group_id);

[[nodiscard]] ContinuationState decode_continuation(
    std::span<const std::uint8_t> encoded, std::string_view expected_model_binding,
    std::string_view expected_artifact_fingerprint, std::span<const KvGroupSpec> expected_groups,
    std::int32_t token_domain, std::uint32_t sliding_group_id,
    std::uint32_t global_group_id);

// Returns the deepest endpoint/anchor whose exact token prefix matches incoming. A zero frontier
// means no reusable checkpoint. The selected image is ready for HeterogeneousKVCache::import_host.
[[nodiscard]] ContinuationRestoreSelection select_continuation_restore(
    const ContinuationState& state, std::span<const std::int32_t> incoming,
    std::span<const KvGroupSpec> groups, std::uint32_t sliding_group_id,
    std::uint32_t global_group_id);

} // namespace ninfer::targets::gemma4::detail
