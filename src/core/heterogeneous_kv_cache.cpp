#include "core/heterogeneous_kv_cache.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

std::uint32_t page_count(std::uint32_t tokens, std::uint32_t page_tokens) {
    if (tokens == 0 || page_tokens == 0) {
        throw std::invalid_argument("heterogeneous KV token/page extent must be positive");
    }
    return 1U + (tokens - 1U) / page_tokens;
}

std::uint32_t checked_add(std::uint32_t left, std::uint32_t right, const char* message) {
    if (right > std::numeric_limits<std::uint32_t>::max() - left) {
        throw std::overflow_error(message);
    }
    return left + right;
}

std::uint32_t sliding_table_pages(const KvGroupSpec& spec) {
    return page_count(checked_add(spec.window_tokens, spec.geometry.page_tokens - 1U,
                                  "heterogeneous KV sliding extent overflow"),
                      spec.geometry.page_tokens);
}

std::uint32_t table_pages(const KvGroupSpec& spec) {
    return spec.retention == KvGroupRetention::SlidingWindow
               ? sliding_table_pages(spec)
               : page_count(spec.maximum_context, spec.geometry.page_tokens);
}

void validate_group(const KvGroupSpec& spec) {
    if (spec.layer_count == 0 || spec.maximum_context == 0 ||
        spec.maximum_transaction_tokens == 0 || spec.table_rows <= 0 ||
        spec.physical_page_groups == 0 || spec.geometry.page_tokens == 0 ||
        spec.geometry.planes.empty()) {
        throw std::invalid_argument("heterogeneous KV group has an empty extent");
    }
    if (spec.maximum_context >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("heterogeneous KV context exceeds device state range");
    }
    if (spec.retention == KvGroupRetention::SlidingWindow) {
        if (spec.window_tokens == 0 || spec.window_tokens > spec.maximum_context) {
            throw std::invalid_argument("heterogeneous KV sliding window is invalid");
        }
    } else if (spec.window_tokens != 0) {
        throw std::invalid_argument("full-history KV group cannot declare a window");
    }
    for (const KVPlaneGeometry& plane : spec.geometry.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0 || plane.alignment == 0) {
            throw std::invalid_argument("heterogeneous KV plane geometry is invalid");
        }
        (void)dtype_size(plane.dtype);
    }
    const std::uint32_t logical = table_pages(spec);
    const std::uint32_t transaction = page_count(
        checked_add(spec.maximum_transaction_tokens, spec.geometry.page_tokens - 1U,
                    "heterogeneous KV transaction extent overflow"),
        spec.geometry.page_tokens);
    const std::uint64_t minimum = spec.retention == KvGroupRetention::SlidingWindow
                                      ? static_cast<std::uint64_t>(spec.table_rows) * logical +
                                            transaction
                                      : std::max<std::uint64_t>(logical, spec.table_rows);
    if (spec.physical_page_groups < minimum) {
        throw std::invalid_argument("heterogeneous KV physical capacity is below its guarantee");
    }
}

std::uint32_t visible_begin_for(const KvGroupSpec& spec, std::uint32_t frontier) {
    return spec.retention == KvGroupRetention::SlidingWindow && frontier > spec.window_tokens
               ? frontier - spec.window_tokens
               : 0U;
}

struct Binding {
    std::uint32_t logical_block = 0;
    DeviceKVPageLease page;
};

Binding* find_binding(std::vector<Binding>& bindings, std::uint32_t logical_block) {
    const auto found = std::lower_bound(
        bindings.begin(), bindings.end(), logical_block,
        [](const Binding& binding, std::uint32_t block) { return binding.logical_block < block; });
    return found == bindings.end() || found->logical_block != logical_block ? nullptr : &*found;
}

const Binding* find_binding(const std::vector<Binding>& bindings, std::uint32_t logical_block) {
    const auto found = std::lower_bound(
        bindings.begin(), bindings.end(), logical_block,
        [](const Binding& binding, std::uint32_t block) { return binding.logical_block < block; });
    return found == bindings.end() || found->logical_block != logical_block ? nullptr : &*found;
}

constexpr std::array<std::byte, 8> kDescriptorMagic{
    std::byte{'N'}, std::byte{'K'}, std::byte{'V'}, std::byte{'G'},
    std::byte{'R'}, std::byte{'P'}, std::byte{'1'}, std::byte{0}};

void append_u32(std::vector<std::byte>& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

std::uint32_t read_u32(std::span<const std::byte> encoded, std::size_t& cursor) {
    if (cursor > encoded.size() || encoded.size() - cursor < 4) {
        throw std::invalid_argument("heterogeneous KV descriptor is truncated");
    }
    std::uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        value |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(encoded[cursor++]))
                 << shift;
    }
    return value;
}

} // namespace

std::size_t KvGroupLayout::payload_bytes() const noexcept { return pages.payload_bytes(); }

std::size_t KvGroupLayout::metadata_bytes() const noexcept {
    return execution_tables.metadata_bytes() + transaction_tables.metadata_bytes() +
           device_state.region.bytes;
}

std::size_t HeterogeneousKVCacheLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const KvGroupLayout& group : groups) { total += group.payload_bytes(); }
    return total;
}

std::size_t HeterogeneousKVCacheLayout::metadata_bytes() const noexcept {
    std::size_t total = 0;
    for (const KvGroupLayout& group : groups) { total += group.metadata_bytes(); }
    return total;
}

HeterogeneousKVCacheLayout plan_heterogeneous_kv_cache(
    LayoutBuilder& builder, std::span<const KvGroupSpec> specs) {
    if (specs.empty()) { throw std::invalid_argument("heterogeneous KV requires a group"); }
    HeterogeneousKVCacheLayout out;
    out.table_rows = specs.front().table_rows;
    out.groups.reserve(specs.size());
    for (std::size_t index = 0; index < specs.size(); ++index) {
        const KvGroupSpec& spec = specs[index];
        validate_group(spec);
        if (spec.table_rows != out.table_rows) {
            throw std::invalid_argument("heterogeneous KV groups disagree on table rows");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (specs[previous].group_id == spec.group_id) {
                throw std::invalid_argument("heterogeneous KV group id is duplicated");
            }
        }
        const std::uint32_t capacity = table_pages(spec);
        KvGroupLayout group;
        group.spec = spec;
        group.table_page_capacity = capacity;
        group.pages = plan_device_kv_page_pool(
            builder, {.page_group_count = spec.physical_page_groups, .geometry = spec.geometry});
        group.execution_tables = plan_kv_execution_tables(
            builder, {.logical_page_capacity = capacity, .table_rows = spec.table_rows});
        group.transaction_tables = plan_kv_execution_tables(
            builder, {.logical_page_capacity = capacity, .table_rows = spec.table_rows});
        group.device_state = builder.add_tensor(DType::I32, {2, spec.table_rows}, 256,
                                                "heterogeneous KV group state");
        out.groups.push_back(std::move(group));
    }
    return out;
}

KvGroupDescriptor kv_group_descriptor(const KvGroupSpec& spec) {
    validate_group(spec);
    return KvGroupDescriptor{.group_id = spec.group_id,
                             .retention = spec.retention,
                             .layer_count = spec.layer_count,
                             .maximum_context = spec.maximum_context,
                             .window_tokens = spec.window_tokens,
                             .geometry = spec.geometry};
}

std::vector<std::byte> encode_kv_group_descriptors(std::span<const KvGroupSpec> groups) {
    if (groups.empty() || groups.size() > 64) {
        throw std::invalid_argument("cannot encode an invalid KV group count");
    }
    std::vector<std::byte> out(kDescriptorMagic.begin(), kDescriptorMagic.end());
    append_u32(out, static_cast<std::uint32_t>(groups.size()));
    for (const KvGroupSpec& spec : groups) {
        const KvGroupDescriptor descriptor = kv_group_descriptor(spec);
        append_u32(out, descriptor.group_id);
        append_u32(out, static_cast<std::uint32_t>(descriptor.retention));
        append_u32(out, descriptor.layer_count);
        append_u32(out, descriptor.maximum_context);
        append_u32(out, descriptor.window_tokens);
        append_u32(out, descriptor.geometry.page_tokens);
        append_u32(out, static_cast<std::uint32_t>(descriptor.geometry.device_plane_order));
        append_u32(out, static_cast<std::uint32_t>(descriptor.geometry.planes.size()));
        for (const KVPlaneGeometry& plane : descriptor.geometry.planes) {
            append_u32(out, static_cast<std::uint32_t>(plane.dtype));
            append_u32(out, static_cast<std::uint32_t>(plane.leading_extent));
            append_u32(out, static_cast<std::uint32_t>(plane.head_extent));
            if (plane.alignment > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("KV group plane alignment cannot be serialized");
            }
            append_u32(out, static_cast<std::uint32_t>(plane.alignment));
        }
    }
    return out;
}

std::vector<KvGroupDescriptor> decode_kv_group_descriptors(std::span<const std::byte> encoded) {
    if (encoded.size() < kDescriptorMagic.size() ||
        !std::equal(kDescriptorMagic.begin(), kDescriptorMagic.end(), encoded.begin())) {
        throw std::invalid_argument("heterogeneous KV descriptor magic/version is invalid");
    }
    std::size_t cursor = kDescriptorMagic.size();
    const std::uint32_t count = read_u32(encoded, cursor);
    if (count == 0 || count > 64) {
        throw std::invalid_argument("heterogeneous KV descriptor group count is invalid");
    }
    std::vector<KvGroupDescriptor> out;
    out.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        KvGroupDescriptor descriptor;
        descriptor.group_id = read_u32(encoded, cursor);
        const std::uint32_t retention = read_u32(encoded, cursor);
        if (retention > static_cast<std::uint32_t>(KvGroupRetention::SlidingWindow)) {
            throw std::invalid_argument("heterogeneous KV descriptor retention is invalid");
        }
        descriptor.retention = static_cast<KvGroupRetention>(retention);
        descriptor.layer_count = read_u32(encoded, cursor);
        descriptor.maximum_context = read_u32(encoded, cursor);
        descriptor.window_tokens = read_u32(encoded, cursor);
        descriptor.geometry.page_tokens = read_u32(encoded, cursor);
        const std::uint32_t order = read_u32(encoded, cursor);
        if (order > static_cast<std::uint32_t>(PagedKVPlaneOrder::HeadMajor)) {
            throw std::invalid_argument("heterogeneous KV descriptor plane order is invalid");
        }
        descriptor.geometry.device_plane_order = static_cast<PagedKVPlaneOrder>(order);
        const std::uint32_t planes = read_u32(encoded, cursor);
        if (descriptor.layer_count == 0 || descriptor.maximum_context == 0 ||
            descriptor.geometry.page_tokens == 0 || planes == 0 || planes > 4096) {
            throw std::invalid_argument("heterogeneous KV descriptor extent is invalid");
        }
        descriptor.geometry.planes.reserve(planes);
        for (std::uint32_t plane_index = 0; plane_index < planes; ++plane_index) {
            const std::uint32_t dtype = read_u32(encoded, cursor);
            const std::uint32_t leading = read_u32(encoded, cursor);
            const std::uint32_t heads = read_u32(encoded, cursor);
            const std::uint32_t alignment = read_u32(encoded, cursor);
            if (dtype > static_cast<std::uint32_t>(DType::FP8_E4M3FN) || leading == 0 ||
                leading > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
                heads == 0 ||
                heads > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
                alignment == 0) {
                throw std::invalid_argument("heterogeneous KV descriptor plane is invalid");
            }
            descriptor.geometry.planes.push_back(
                {.dtype = static_cast<DType>(dtype),
                 .leading_extent = static_cast<std::int32_t>(leading),
                 .head_extent = static_cast<std::int32_t>(heads),
                 .alignment = alignment});
        }
        if (descriptor.retention == KvGroupRetention::SlidingWindow) {
            if (descriptor.window_tokens == 0 ||
                descriptor.window_tokens > descriptor.maximum_context) {
                throw std::invalid_argument("heterogeneous KV descriptor window is invalid");
            }
        } else if (descriptor.window_tokens != 0) {
            throw std::invalid_argument("heterogeneous KV full descriptor has a window");
        }
        for (const KvGroupDescriptor& previous : out) {
            if (previous.group_id == descriptor.group_id) {
                throw std::invalid_argument("heterogeneous KV descriptor id is duplicated");
            }
        }
        out.push_back(std::move(descriptor));
    }
    if (cursor != encoded.size()) {
        throw std::invalid_argument("heterogeneous KV descriptor has trailing bytes");
    }
    return out;
}

struct HeterogeneousKVCache::Impl {
    struct Group {
        KvGroupLayout layout;
        DeviceKVPagePool pages;
        KVExecutionTablePool tables;
        KVExecutionTablePool transaction_tables;
        Tensor device_state;
        PinnedHostBuffer state_shadow;

        Group(DeviceSpan backing, const KvGroupLayout& source)
            : layout(source), pages(backing, layout.pages),
              tables(backing, layout.execution_tables, pages),
              transaction_tables(backing, layout.transaction_tables, pages),
              device_state(layout.device_state.bind(backing)),
              state_shadow(static_cast<std::size_t>(layout.spec.table_rows) * 2U *
                           sizeof(std::int32_t)) {}

        void publish_state(std::int32_t row, std::uint32_t visible_begin,
                           std::uint32_t frontier, cudaStream_t stream) {
            auto* values = static_cast<std::int32_t*>(state_shadow.data());
            values[2 * row] = static_cast<std::int32_t>(visible_begin);
            values[2 * row + 1] = static_cast<std::int32_t>(frontier);
            CUDA_CHECK(cudaMemcpyAsync(device_state.slice(1, row, 1).data, values + 2 * row,
                                       2U * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
        }
    };

    struct RowGroup {
        KVExecutionRowLease execution_row;
        KVExecutionRowLease transaction_row;
        std::vector<Binding> bindings;
        std::uint32_t frontier = 0;
    };

    struct Row {
        bool active = false;
        bool transaction_active = false;
        std::vector<RowGroup> groups;
    };

    std::vector<std::unique_ptr<Group>> groups;
    std::vector<Row> rows;

    std::size_t group_index(std::uint32_t id) const {
        for (std::size_t index = 0; index < groups.size(); ++index) {
            if (groups[index]->layout.spec.group_id == id) { return index; }
        }
        throw std::out_of_range("heterogeneous KV group id is unknown");
    }

    Row& require_row(std::int32_t row) {
        if (row < 0 || row >= static_cast<std::int32_t>(rows.size()) || !rows[row].active) {
            throw std::invalid_argument("heterogeneous KV row is inactive");
        }
        return rows[static_cast<std::size_t>(row)];
    }

    const Row& require_row(std::int32_t row) const {
        return const_cast<Impl*>(this)->require_row(row);
    }
};

struct HeterogeneousKVTransaction::Impl {
    struct PendingGroup {
        std::vector<Binding> pages;
    };
    HeterogeneousKVCache* cache = nullptr;
    std::int32_t row = -1;
    std::uint32_t first = 0;
    std::uint32_t target = 0;
    std::vector<PendingGroup> groups;
};

struct HeterogeneousKVCheckpoint::Impl {
    struct Group {
        std::uint32_t group_id = 0;
        std::uint32_t frontier = 0;
        std::vector<Binding> pages;
    };
    const HeterogeneousKVCache* owner = nullptr;
    std::vector<Group> groups;
};

HeterogeneousKVTransaction::HeterogeneousKVTransaction() noexcept = default;
HeterogeneousKVTransaction::~HeterogeneousKVTransaction() { rollback(); }
HeterogeneousKVTransaction::HeterogeneousKVTransaction(HeterogeneousKVTransaction&&) noexcept =
    default;
HeterogeneousKVTransaction& HeterogeneousKVTransaction::operator=(
    HeterogeneousKVTransaction&& other) noexcept {
    if (this == &other) { return *this; }
    rollback();
    impl_ = std::move(other.impl_);
    return *this;
}
HeterogeneousKVTransaction::HeterogeneousKVTransaction(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
bool HeterogeneousKVTransaction::valid() const noexcept { return impl_ != nullptr; }
std::uint32_t HeterogeneousKVTransaction::first_position() const noexcept {
    return impl_ == nullptr ? 0 : impl_->first;
}
std::uint32_t HeterogeneousKVTransaction::target_frontier() const noexcept {
    return impl_ == nullptr ? 0 : impl_->target;
}
void HeterogeneousKVTransaction::rollback() noexcept {
    if (impl_ != nullptr) {
        HeterogeneousKVCache::Impl& cache = *impl_->cache->impl_;
        if (impl_->row >= 0 && impl_->row < static_cast<std::int32_t>(cache.rows.size())) {
            cache.rows[static_cast<std::size_t>(impl_->row)].transaction_active = false;
        }
    }
    impl_.reset();
}

HeterogeneousKVCheckpoint::HeterogeneousKVCheckpoint() noexcept = default;
HeterogeneousKVCheckpoint::~HeterogeneousKVCheckpoint() = default;
HeterogeneousKVCheckpoint::HeterogeneousKVCheckpoint(HeterogeneousKVCheckpoint&&) noexcept =
    default;
HeterogeneousKVCheckpoint& HeterogeneousKVCheckpoint::operator=(
    HeterogeneousKVCheckpoint&&) noexcept = default;
HeterogeneousKVCheckpoint::HeterogeneousKVCheckpoint(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
bool HeterogeneousKVCheckpoint::valid() const noexcept { return impl_ != nullptr; }
std::uint32_t HeterogeneousKVCheckpoint::frontier() const noexcept {
    return impl_ == nullptr || impl_->groups.empty() ? 0 : impl_->groups.front().frontier;
}

HeterogeneousKVCache::HeterogeneousKVCache(DeviceSpan backing,
                                           const HeterogeneousKVCacheLayout& layout)
    : impl_(std::make_unique<Impl>()) {
    if (layout.groups.empty() || layout.table_rows <= 0) {
        throw std::invalid_argument("heterogeneous KV layout is empty");
    }
    impl_->groups.reserve(layout.groups.size());
    for (const KvGroupLayout& group : layout.groups) {
        impl_->groups.push_back(std::make_unique<Impl::Group>(backing, group));
    }
    impl_->rows.resize(static_cast<std::size_t>(layout.table_rows));
}

HeterogeneousKVCache::~HeterogeneousKVCache() = default;
std::size_t HeterogeneousKVCache::group_count() const noexcept { return impl_->groups.size(); }
std::int32_t HeterogeneousKVCache::row_count() const noexcept {
    return static_cast<std::int32_t>(impl_->rows.size());
}
bool HeterogeneousKVCache::active(std::int32_t row) const {
    if (row < 0 || row >= row_count()) { throw std::out_of_range("heterogeneous KV row invalid"); }
    return impl_->rows[static_cast<std::size_t>(row)].active;
}

void HeterogeneousKVCache::activate(std::int32_t row, cudaStream_t stream) {
    if (row < 0 || row >= row_count()) { throw std::out_of_range("heterogeneous KV row invalid"); }
    Impl::Row& target = impl_->rows[static_cast<std::size_t>(row)];
    if (target.active) { throw std::logic_error("heterogeneous KV row is already active"); }
    std::vector<Impl::RowGroup> groups;
    groups.reserve(impl_->groups.size());
    for (const std::unique_ptr<Impl::Group>& group : impl_->groups) {
        Impl::RowGroup state;
        state.execution_row = group->tables.acquire(row);
        state.transaction_row = group->transaction_tables.acquire(row);
        group->publish_state(row, 0, 0, stream);
        groups.push_back(std::move(state));
    }
    target.groups = std::move(groups);
    target.active = true;
}

void HeterogeneousKVCache::deactivate(std::int32_t row) noexcept {
    if (row < 0 || row >= row_count()) { return; }
    Impl::Row& target = impl_->rows[static_cast<std::size_t>(row)];
    if (!target.active) { return; }
    if (target.transaction_active) { return; }
    target.groups.clear();
    target.active = false;
}

std::uint32_t HeterogeneousKVCache::frontier(std::int32_t row) const {
    const Impl::Row& state = impl_->require_row(row);
    const std::uint32_t value = state.groups.front().frontier;
    for (const Impl::RowGroup& group : state.groups) {
        if (group.frontier != value) {
            throw std::logic_error("heterogeneous KV group frontiers diverged");
        }
    }
    return value;
}

std::uint32_t HeterogeneousKVCache::visible_begin(std::int32_t row,
                                                  std::uint32_t group_id) const {
    const std::size_t index = impl_->group_index(group_id);
    const Impl::Row& state = impl_->require_row(row);
    return visible_begin_for(impl_->groups[index]->layout.spec, state.groups[index].frontier);
}

std::uint32_t HeterogeneousKVCache::resident_pages(std::int32_t row,
                                                   std::uint32_t group_id) const {
    const std::size_t index = impl_->group_index(group_id);
    return static_cast<std::uint32_t>(impl_->require_row(row).groups[index].bindings.size());
}

std::uint32_t HeterogeneousKVCache::table_page_capacity(std::uint32_t group_id) const {
    return impl_->groups[impl_->group_index(group_id)]->layout.table_page_capacity;
}

const KVPageGeometry& HeterogeneousKVCache::geometry(std::uint32_t group_id) const {
    return impl_->groups[impl_->group_index(group_id)]->pages.geometry();
}

std::size_t HeterogeneousKVCache::plane_count(std::uint32_t group_id) const {
    return impl_->groups[impl_->group_index(group_id)]->pages.plane_count();
}

Tensor HeterogeneousKVCache::plane(std::uint32_t group_id, std::size_t plane_index) const {
    return impl_->groups[impl_->group_index(group_id)]->pages.plane(plane_index);
}

KvGroupExecutionView HeterogeneousKVCache::execution_view(std::int32_t row,
                                                          std::uint32_t group_id) const {
    const std::size_t index = impl_->group_index(group_id);
    const Impl::Row& state = impl_->require_row(row);
    const Impl::Group& group = *impl_->groups[index];
    return {.block_table = group.tables.row(state.groups[index].execution_row.handle()),
            .state = group.device_state.slice(1, row, 1).view({2}),
            .group_id = group_id,
            .retention = group.layout.spec.retention,
            .table_page_capacity = group.layout.table_page_capacity};
}

KvGroupBatchExecutionView HeterogeneousKVCache::batch_execution_view(
    std::uint32_t group_id) const {
    const Impl::Group& group = *impl_->groups[impl_->group_index(group_id)];
    return {.block_tables = group.tables.matrix(),
            .states = group.device_state,
            .group_id = group_id,
            .retention = group.layout.spec.retention,
            .table_page_capacity = group.layout.table_page_capacity};
}

KvGroupBatchExecutionView HeterogeneousKVCache::staging_batch_execution_view(
    std::uint32_t group_id) const {
    const Impl::Group& group = *impl_->groups[impl_->group_index(group_id)];
    return {.block_tables = group.transaction_tables.matrix(),
            .states = group.device_state,
            .group_id = group_id,
            .retention = group.layout.spec.retention,
            .table_page_capacity = group.layout.table_page_capacity};
}

KvGroupExecutionView HeterogeneousKVTransaction::execution_view(std::uint32_t group_id) const {
    if (impl_ == nullptr) { throw std::logic_error("heterogeneous KV transaction is empty"); }
    HeterogeneousKVCache::Impl& cache = *impl_->cache->impl_;
    const std::size_t index = cache.group_index(group_id);
    const HeterogeneousKVCache::Impl::Group& group = *cache.groups[index];
    const HeterogeneousKVCache::Impl::RowGroup& row_group =
        cache.require_row(impl_->row).groups[index];
    return {.block_table = group.transaction_tables.row(row_group.transaction_row.handle()),
            .state = group.device_state.slice(1, impl_->row, 1).view({2}),
            .group_id = group_id,
            .retention = group.layout.spec.retention,
            .table_page_capacity = group.layout.table_page_capacity};
}

Tensor HeterogeneousKVTransaction::plane(std::uint32_t group_id,
                                         std::size_t plane_index) const {
    if (impl_ == nullptr) { throw std::logic_error("heterogeneous KV transaction is empty"); }
    return impl_->cache->plane(group_id, plane_index);
}

HeterogeneousKVTransaction HeterogeneousKVCache::begin_transaction(
    std::int32_t row, std::uint32_t first_position, std::uint32_t token_count,
    cudaStream_t stream) {
    if (token_count == 0) { throw std::invalid_argument("heterogeneous KV transaction is empty"); }
    Impl::Row& state = impl_->require_row(row);
    if (state.transaction_active) {
        throw std::logic_error("heterogeneous KV row already has an active transaction");
    }
    if (frontier(row) != first_position) {
        throw std::invalid_argument("heterogeneous KV transaction is not at the frontier");
    }
    const std::uint32_t target =
        checked_add(first_position, token_count, "heterogeneous KV frontier overflow");
    auto transaction = std::make_unique<HeterogeneousKVTransaction::Impl>();
    transaction->cache = this;
    transaction->row = row;
    transaction->first = first_position;
    transaction->target = target;
    transaction->groups.resize(impl_->groups.size());

    std::vector<DeviceKVPageReservationRequest> requests;
    std::vector<std::size_t> request_groups;
    for (std::size_t index = 0; index < impl_->groups.size(); ++index) {
        Impl::Group& group = *impl_->groups[index];
        Impl::RowGroup& row_group = state.groups[index];
        const KvGroupSpec& spec = group.layout.spec;
        if (target > spec.maximum_context || token_count > spec.maximum_transaction_tokens) {
            throw std::invalid_argument("heterogeneous KV transaction exceeds its group bound");
        }
        const std::uint32_t first_block = first_position / spec.geometry.page_tokens;
        const std::uint32_t last_block = (target - 1U) / spec.geometry.page_tokens;
        std::vector<Binding>& pending = transaction->groups[index].pages;
        for (std::uint32_t block = first_block; block <= last_block; ++block) {
            const Binding* existing = find_binding(row_group.bindings, block);
            const bool copy_tail = existing != nullptr && block == first_block &&
                                   first_position % spec.geometry.page_tokens != 0;
            if (existing == nullptr || copy_tail) {
                pending.push_back(Binding{.logical_block = block});
            }
        }
        row_group.bindings.reserve(row_group.bindings.size() + pending.size());
        if (!pending.empty()) {
            requests.push_back({.pool = &group.pages,
                                .pages = static_cast<std::uint32_t>(pending.size())});
            request_groups.push_back(index);
        }
    }
    std::vector<DeviceKVPageReservation> reservations = reserve_device_kv_page_bundle(requests);
    for (std::size_t request = 0; request < reservations.size(); ++request) {
        const std::size_t group_index = request_groups[request];
        Impl::Group& group = *impl_->groups[group_index];
        Impl::RowGroup& row_group = state.groups[group_index];
        for (Binding& binding : transaction->groups[group_index].pages) {
            binding.page = group.pages.materialize_one(reservations[request]);
            const Binding* existing = find_binding(row_group.bindings, binding.logical_block);
            if (existing != nullptr) {
                group.pages.copy_page(existing->page.handle(), binding.page.handle(), stream);
            }
        }
    }
    for (std::size_t index = 0; index < impl_->groups.size(); ++index) {
        Impl::Group& group = *impl_->groups[index];
        Impl::RowGroup& row_group = state.groups[index];
        const Tensor committed = group.tables.row(row_group.execution_row.handle());
        const Tensor staging = group.transaction_tables.row(row_group.transaction_row.handle());
        CUDA_CHECK(cudaMemcpyAsync(staging.data, committed.data, committed.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        const KvGroupSpec& spec = group.layout.spec;
        for (const Binding& binding : transaction->groups[index].pages) {
            const std::uint32_t slot = spec.retention == KvGroupRetention::SlidingWindow
                                           ? binding.logical_block % group.layout.table_page_capacity
                                           : binding.logical_block;
            const DeviceKVPageHandle handle = binding.page.handle();
            group.transaction_tables.publish(row_group.transaction_row.handle(), slot,
                                             std::span<const DeviceKVPageHandle>(&handle, 1),
                                             stream);
        }
    }
    state.transaction_active = true;
    return HeterogeneousKVTransaction(std::move(transaction));
}

void HeterogeneousKVTransaction::commit(cudaStream_t stream) {
    if (impl_ == nullptr) { throw std::logic_error("heterogeneous KV transaction is empty"); }
    HeterogeneousKVCache& cache = *impl_->cache;
    HeterogeneousKVCache::Impl::Row& row = cache.impl_->require_row(impl_->row);
    if (cache.frontier(impl_->row) != impl_->first) {
        throw std::logic_error("heterogeneous KV transaction frontier changed before commit");
    }
    for (std::size_t index = 0; index < cache.impl_->groups.size(); ++index) {
        HeterogeneousKVCache::Impl::Group& group = *cache.impl_->groups[index];
        HeterogeneousKVCache::Impl::RowGroup& row_group = row.groups[index];
        const KvGroupSpec& spec = group.layout.spec;
        const std::uint32_t visible_begin = visible_begin_for(spec, impl_->target);
        const std::uint32_t first_retained_block = visible_begin / spec.geometry.page_tokens;
        if (spec.retention == KvGroupRetention::SlidingWindow) {
            std::erase_if(row_group.bindings, [&](const Binding& binding) {
                return binding.logical_block < first_retained_block;
            });
        }
        for (Binding& binding : impl_->groups[index].pages) {
            const std::uint32_t slot = binding.logical_block % group.layout.table_page_capacity;
            std::erase_if(row_group.bindings, [&](const Binding& current) {
                return current.logical_block == binding.logical_block ||
                       (spec.retention == KvGroupRetention::SlidingWindow &&
                        current.logical_block % group.layout.table_page_capacity == slot);
            });
            row_group.bindings.push_back(std::move(binding));
        }
        std::sort(row_group.bindings.begin(), row_group.bindings.end(),
                  [](const Binding& left, const Binding& right) {
                      return left.logical_block < right.logical_block;
                  });
        row_group.frontier = impl_->target;
        const Tensor committed = group.tables.row(row_group.execution_row.handle());
        const Tensor staging =
            group.transaction_tables.row(row_group.transaction_row.handle());
        CUDA_CHECK(cudaMemcpyAsync(committed.data, staging.data, committed.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        group.publish_state(impl_->row, visible_begin, impl_->target, stream);
    }
    row.transaction_active = false;
    impl_.reset();
}

HeterogeneousKVCheckpoint HeterogeneousKVCache::checkpoint(std::int32_t row,
                                                           cudaStream_t stream) {
    Impl::Row& source = impl_->require_row(row);
    if (source.transaction_active) {
        throw std::logic_error("cannot checkpoint an active heterogeneous KV transaction");
    }
    auto checkpoint = std::make_unique<HeterogeneousKVCheckpoint::Impl>();
    checkpoint->owner = this;
    checkpoint->groups.resize(impl_->groups.size());
    std::vector<DeviceKVPageReservationRequest> requests;
    for (std::size_t index = 0; index < impl_->groups.size(); ++index) {
        auto& destination = checkpoint->groups[index];
        destination.group_id = impl_->groups[index]->layout.spec.group_id;
        destination.frontier = source.groups[index].frontier;
        destination.pages.reserve(source.groups[index].bindings.size());
        if (!source.groups[index].bindings.empty()) {
            requests.push_back({.pool = &impl_->groups[index]->pages,
                                .pages = static_cast<std::uint32_t>(
                                    source.groups[index].bindings.size())});
        }
    }
    std::vector<DeviceKVPageReservation> reservations = reserve_device_kv_page_bundle(requests);
    std::size_t reservation_index = 0;
    for (std::size_t index = 0; index < impl_->groups.size(); ++index) {
        if (source.groups[index].bindings.empty()) { continue; }
        Impl::Group& group = *impl_->groups[index];
        auto& destination = checkpoint->groups[index];
        for (const Binding& binding : source.groups[index].bindings) {
            Binding copy{.logical_block = binding.logical_block,
                         .page = group.pages.materialize_one(reservations[reservation_index])};
            group.pages.copy_page(binding.page.handle(), copy.page.handle(), stream);
            destination.pages.push_back(std::move(copy));
        }
        ++reservation_index;
    }
    return HeterogeneousKVCheckpoint(std::move(checkpoint));
}

void HeterogeneousKVCache::restore(std::int32_t row,
                                   const HeterogeneousKVCheckpoint& checkpoint,
                                   cudaStream_t stream) {
    if (checkpoint.impl_ == nullptr || checkpoint.impl_->owner != this) {
        throw std::invalid_argument("heterogeneous KV checkpoint belongs to another cache");
    }
    Impl::Row& destination = impl_->require_row(row);
    if (destination.transaction_active) {
        throw std::logic_error("cannot restore an active heterogeneous KV transaction");
    }
    std::vector<DeviceKVPageReservationRequest> requests;
    for (std::size_t index = 0; index < impl_->groups.size(); ++index) {
        if (checkpoint.impl_->groups[index].group_id != impl_->groups[index]->layout.spec.group_id) {
            throw std::invalid_argument("heterogeneous KV checkpoint group order changed");
        }
        if (!checkpoint.impl_->groups[index].pages.empty()) {
            requests.push_back({.pool = &impl_->groups[index]->pages,
                                .pages = static_cast<std::uint32_t>(
                                    checkpoint.impl_->groups[index].pages.size())});
        }
    }
    std::vector<DeviceKVPageReservation> reservations = reserve_device_kv_page_bundle(requests);
    std::vector<std::vector<Binding>> replacement(impl_->groups.size());
    std::size_t reservation_index = 0;
    for (std::size_t index = 0; index < impl_->groups.size(); ++index) {
        const auto& source = checkpoint.impl_->groups[index];
        replacement[index].reserve(source.pages.size());
        if (source.pages.empty()) { continue; }
        Impl::Group& group = *impl_->groups[index];
        for (const Binding& binding : source.pages) {
            Binding copy{.logical_block = binding.logical_block,
                         .page = group.pages.materialize_one(reservations[reservation_index])};
            group.pages.copy_page(binding.page.handle(), copy.page.handle(), stream);
            replacement[index].push_back(std::move(copy));
        }
        ++reservation_index;
    }
    for (std::size_t index = 0; index < impl_->groups.size(); ++index) {
        Impl::Group& group = *impl_->groups[index];
        Impl::RowGroup& row_group = destination.groups[index];
        row_group.bindings = std::move(replacement[index]);
        row_group.frontier = checkpoint.impl_->groups[index].frontier;
        const KvGroupSpec& spec = group.layout.spec;
        for (const Binding& binding : row_group.bindings) {
            const std::uint32_t slot = spec.retention == KvGroupRetention::SlidingWindow
                                           ? binding.logical_block % group.layout.table_page_capacity
                                           : binding.logical_block;
            const DeviceKVPageHandle handle = binding.page.handle();
            group.tables.publish(row_group.execution_row.handle(), slot,
                                 std::span<const DeviceKVPageHandle>(&handle, 1), stream);
        }
        group.publish_state(row, visible_begin_for(spec, row_group.frontier), row_group.frontier,
                            stream);
    }
}

} // namespace ninfer
