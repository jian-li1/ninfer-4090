#include "runtime/kv_groups.h"

#include <ninfer/targets/gemma4_31b_it/config.h>

#include <limits>
#include <stdexcept>

namespace ninfer::targets::gemma4_31b_it {
namespace {

std::uint32_t page_count(std::uint32_t tokens) {
    if (tokens == 0) { throw std::invalid_argument("Gemma KV token extent must be positive"); }
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint32_t checked_add(std::uint32_t left, std::uint32_t right, const char* label) {
    if (right > std::numeric_limits<std::uint32_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

std::uint32_t checked_u32(std::uint64_t value, const char* label) {
    if (value > std::numeric_limits<std::uint32_t>::max()) { throw std::overflow_error(label); }
    return static_cast<std::uint32_t>(value);
}

KVPageGeometry make_geometry(std::uint32_t layers, std::int32_t heads, std::int32_t head_dim,
                             KvCacheStorage storage) {
    const PagedKVStorageLayout layer = paged_kv_storage_layout(storage, head_dim);
    KVPageGeometry geometry;
    geometry.page_tokens = kPagedKVPageSize;
    geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * layer.planes_per_layer());
    for (std::uint32_t index = 0; index < layers; ++index) {
        geometry.planes.push_back(
            {layer.key.data_dtype, layer.key.data_leading_extent, heads, 256});
        geometry.planes.push_back(
            {layer.value.data_dtype, layer.value.data_leading_extent, heads, 256});
        if (layer.key.has_scale()) {
            geometry.planes.push_back(
                {layer.key.scale_dtype, layer.key.scale_leading_extent, heads, 256});
        }
        if (layer.value.has_scale()) {
            geometry.planes.push_back(
                {layer.value.scale_dtype, layer.value.scale_leading_extent, heads, 256});
        }
    }
    return geometry;
}

} // namespace

std::array<KvGroupSpec, 2> make_text_kv_group_specs(const TextKvGroupPlanSpec& spec) {
    if (spec.maximum_context == 0 || spec.maximum_context > TextConfig::maximum_position ||
        spec.maximum_transaction_tokens == 0 || spec.table_rows <= 0 ||
        spec.global_resident_token_capacity < spec.maximum_context) {
        throw std::invalid_argument("Gemma KV group plan has an invalid capacity");
    }
    const std::uint32_t touched_pages = page_count(checked_add(
        spec.maximum_transaction_tokens, static_cast<std::uint32_t>(kPagedKVPageSize) - 1U,
        "Gemma KV transaction extent overflow"));
    const std::uint32_t sliding_table_pages = page_count(
        TextConfig::sliding_window + static_cast<std::uint32_t>(kPagedKVPageSize) - 1U);
    const std::uint32_t sliding_physical = checked_u32(
        (static_cast<std::uint64_t>(spec.table_rows) + spec.local_checkpoint_capacity) *
                sliding_table_pages +
            touched_pages,
        "Gemma sliding KV physical capacity overflow");

    // Aggregate global token capacity is shared by the active lanes. One page per lane absorbs
    // fragmentation and one extra page preserves a partially filled tail during copy-on-write.
    const std::uint32_t global_physical = checked_u32(
        static_cast<std::uint64_t>(page_count(spec.global_resident_token_capacity)) +
            static_cast<std::uint32_t>(spec.table_rows) + 1U,
        "Gemma global KV physical capacity overflow");

    return {
        KvGroupSpec{
            .group_id = kSlidingKvGroup,
            .retention = KvGroupRetention::SlidingWindow,
            .layer_count = static_cast<std::uint32_t>(TextConfig::sliding_layers),
            .maximum_context = spec.maximum_context,
            .window_tokens = TextConfig::sliding_window,
            .maximum_transaction_tokens = spec.maximum_transaction_tokens,
            .table_rows = spec.table_rows,
            .physical_page_groups = sliding_physical,
            .geometry = make_geometry(static_cast<std::uint32_t>(TextConfig::sliding_layers),
                                      TextConfig::sliding_kv_heads,
                                      TextConfig::sliding_head_dim, spec.sliding_storage),
        },
        KvGroupSpec{
            .group_id = kGlobalKvGroup,
            .retention = KvGroupRetention::FullHistory,
            .layer_count = static_cast<std::uint32_t>(TextConfig::full_layers),
            .maximum_context = spec.maximum_context,
            .window_tokens = 0,
            .maximum_transaction_tokens = spec.maximum_transaction_tokens,
            .table_rows = spec.table_rows,
            .physical_page_groups = global_physical,
            .geometry = make_geometry(static_cast<std::uint32_t>(TextConfig::full_layers),
                                      TextConfig::full_kv_heads, TextConfig::full_head_dim,
                                      spec.global_storage),
        },
    };
}

TextKvLayerAddress text_kv_layer_address(std::uint32_t model_layer) {
    if (model_layer >= TextConfig::layers) {
        throw std::out_of_range("Gemma model layer is out of range");
    }
    std::uint32_t sliding_layer = 0;
    std::uint32_t global_layer = 0;
    for (std::uint32_t layer = 0; layer < model_layer; ++layer) {
        if (TextConfig::is_full_attention(layer)) {
            ++global_layer;
        } else {
            ++sliding_layer;
        }
    }
    return TextConfig::is_full_attention(model_layer)
               ? TextKvLayerAddress{.group_id = kGlobalKvGroup, .group_layer = global_layer}
               : TextKvLayerAddress{.group_id = kSlidingKvGroup, .group_layer = sliding_layer};
}

} // namespace ninfer::targets::gemma4_31b_it
