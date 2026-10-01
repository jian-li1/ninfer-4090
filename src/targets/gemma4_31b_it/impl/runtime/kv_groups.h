#pragma once

#include "core/heterogeneous_kv_cache.h"
#include "core/paged_kv_storage.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::gemma4_31b_it {

inline constexpr std::uint32_t kSlidingKvGroup = 0;
inline constexpr std::uint32_t kGlobalKvGroup = 1;

struct TextKvGroupPlanSpec {
    std::uint32_t maximum_context = 0;
    std::uint32_t maximum_transaction_tokens = 0;
    // Aggregate global tokens across active rows and resident checkpoint copies.
    std::uint32_t global_resident_token_capacity = 0;
    // Additional local-ring copies retained outside active rows.
    std::uint32_t local_checkpoint_capacity = 0;
    std::int32_t table_rows = 0;
    KvCacheStorage sliding_storage = KvCacheStorage::BFloat16;
    KvCacheStorage global_storage = KvCacheStorage::BFloat16;
};

struct TextKvLayerAddress {
    std::uint32_t group_id = 0;
    std::uint32_t group_layer = 0;

    friend bool operator==(const TextKvLayerAddress&, const TextKvLayerAddress&) = default;
};

[[nodiscard]] std::array<KvGroupSpec, 2>
make_text_kv_group_specs(const TextKvGroupPlanSpec& spec);

[[nodiscard]] TextKvLayerAddress text_kv_layer_address(std::uint32_t model_layer);

} // namespace ninfer::targets::gemma4_31b_it
