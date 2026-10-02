#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {

enum class KvGroupRetention : std::uint8_t {
    FullHistory = 0,
    SlidingWindow = 1,
};

struct KvGroupSpec {
    std::uint32_t group_id = 0;
    KvGroupRetention retention = KvGroupRetention::FullHistory;
    std::uint32_t layer_count = 0;
    std::uint32_t maximum_context = 0;
    std::uint32_t window_tokens = 0;
    std::uint32_t maximum_transaction_tokens = 0;
    std::int32_t table_rows = 0;
    std::uint32_t physical_page_groups = 0;
    KVPageGeometry geometry;

    friend bool operator==(const KvGroupSpec&, const KvGroupSpec&) = default;
};

struct KvGroupDescriptor {
    std::uint32_t group_id = 0;
    KvGroupRetention retention = KvGroupRetention::FullHistory;
    std::uint32_t layer_count = 0;
    std::uint32_t maximum_context = 0;
    std::uint32_t window_tokens = 0;
    KVPageGeometry geometry;

    friend bool operator==(const KvGroupDescriptor&, const KvGroupDescriptor&) = default;
};

struct KvGroupLayout {
    KvGroupSpec spec;
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    KVExecutionTableLayout transaction_tables;
    TensorRegion device_state;
    std::uint32_t table_page_capacity = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
};

struct HeterogeneousKVCacheLayout {
    std::vector<KvGroupLayout> groups;
    std::int32_t table_rows = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
};

[[nodiscard]] HeterogeneousKVCacheLayout
plan_heterogeneous_kv_cache(LayoutBuilder& builder, std::span<const KvGroupSpec> groups);

[[nodiscard]] KvGroupDescriptor kv_group_descriptor(const KvGroupSpec& spec);
[[nodiscard]] std::vector<std::byte>
encode_kv_group_descriptors(std::span<const KvGroupSpec> groups);
[[nodiscard]] std::vector<KvGroupDescriptor>
decode_kv_group_descriptors(std::span<const std::byte> encoded);

struct KvGroupExecutionView {
    Tensor block_table;
    Tensor state; // I32 [visible_begin, committed_frontier]
    std::uint32_t group_id = 0;
    KvGroupRetention retention = KvGroupRetention::FullHistory;
    std::uint32_t table_page_capacity = 0;
};

struct KvGroupBatchExecutionView {
    Tensor block_tables;
    Tensor states; // I32 [2, rows]: visible_begin, committed_frontier
    std::uint32_t group_id = 0;
    KvGroupRetention retention = KvGroupRetention::FullHistory;
    std::uint32_t table_page_capacity = 0;
};

// Target-neutral host materialization of one logical heterogeneous-KV frontier. Pages are in
// ascending logical-block order; payload holds one plan_host_kv_page_layout record per block.
// Targets own framing, token identity, compatibility fingerprints, and persistence policy.
struct HeterogeneousKVHostGroupImage {
    std::uint32_t group_id = 0;
    std::uint32_t frontier = 0;
    std::vector<std::uint32_t> logical_blocks;
    std::vector<std::byte> payload;

    friend bool operator==(const HeterogeneousKVHostGroupImage&,
                           const HeterogeneousKVHostGroupImage&) = default;
};

struct HeterogeneousKVHostImage {
    std::vector<HeterogeneousKVHostGroupImage> groups;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;

    friend bool operator==(const HeterogeneousKVHostImage&,
                           const HeterogeneousKVHostImage&) = default;
};

class HeterogeneousKVCache;

class HeterogeneousKVTransaction {
public:
    HeterogeneousKVTransaction() noexcept;
    ~HeterogeneousKVTransaction();
    HeterogeneousKVTransaction(HeterogeneousKVTransaction&&) noexcept;
    HeterogeneousKVTransaction& operator=(HeterogeneousKVTransaction&&) noexcept;
    HeterogeneousKVTransaction(const HeterogeneousKVTransaction&) = delete;
    HeterogeneousKVTransaction& operator=(const HeterogeneousKVTransaction&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t first_position() const noexcept;
    [[nodiscard]] std::uint32_t target_frontier() const noexcept;
    [[nodiscard]] KvGroupExecutionView execution_view(std::uint32_t group_id) const;
    [[nodiscard]] Tensor plane(std::uint32_t group_id, std::size_t plane_index) const;
    // Commit or rollback only after work submitted through the staging view is ordered complete.
    void commit(cudaStream_t stream = nullptr);
    // Publish the first token_count rows and discard the remaining reserved suffix.
    void commit_prefix(std::uint32_t token_count, cudaStream_t stream = nullptr);
    void rollback() noexcept;

private:
    friend class HeterogeneousKVCache;
    struct Impl;
    explicit HeterogeneousKVTransaction(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

class HeterogeneousKVCheckpoint {
public:
    HeterogeneousKVCheckpoint() noexcept;
    ~HeterogeneousKVCheckpoint();
    HeterogeneousKVCheckpoint(HeterogeneousKVCheckpoint&&) noexcept;
    HeterogeneousKVCheckpoint& operator=(HeterogeneousKVCheckpoint&&) noexcept;
    HeterogeneousKVCheckpoint(const HeterogeneousKVCheckpoint&) = delete;
    HeterogeneousKVCheckpoint& operator=(const HeterogeneousKVCheckpoint&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;

private:
    friend class HeterogeneousKVCache;
    struct Impl;
    explicit HeterogeneousKVCheckpoint(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

class HeterogeneousKVCache {
public:
    HeterogeneousKVCache(DeviceSpan backing, const HeterogeneousKVCacheLayout& layout);
    ~HeterogeneousKVCache();
    HeterogeneousKVCache(const HeterogeneousKVCache&) = delete;
    HeterogeneousKVCache& operator=(const HeterogeneousKVCache&) = delete;
    HeterogeneousKVCache(HeterogeneousKVCache&&) = delete;
    HeterogeneousKVCache& operator=(HeterogeneousKVCache&&) = delete;

    [[nodiscard]] std::size_t group_count() const noexcept;
    [[nodiscard]] std::int32_t row_count() const noexcept;
    [[nodiscard]] bool active(std::int32_t row) const;
    void activate(std::int32_t row, cudaStream_t stream = nullptr);
    void deactivate(std::int32_t row) noexcept;

    [[nodiscard]] std::uint32_t frontier(std::int32_t row) const;
    [[nodiscard]] std::uint32_t visible_begin(std::int32_t row,
                                              std::uint32_t group_id) const;
    [[nodiscard]] std::uint32_t resident_pages(std::int32_t row,
                                               std::uint32_t group_id) const;
    [[nodiscard]] std::uint32_t table_page_capacity(std::uint32_t group_id) const;
    [[nodiscard]] const KVPageGeometry& geometry(std::uint32_t group_id) const;
    [[nodiscard]] std::size_t plane_count(std::uint32_t group_id) const;
    [[nodiscard]] Tensor plane(std::uint32_t group_id, std::size_t plane_index) const;
    [[nodiscard]] KvGroupExecutionView execution_view(std::int32_t row,
                                                      std::uint32_t group_id) const;
    [[nodiscard]] KvGroupBatchExecutionView batch_execution_view(
        std::uint32_t group_id) const;
    [[nodiscard]] KvGroupBatchExecutionView staging_batch_execution_view(
        std::uint32_t group_id) const;

    [[nodiscard]] HeterogeneousKVTransaction begin_transaction(
        std::int32_t row, std::uint32_t first_position, std::uint32_t token_count,
        cudaStream_t stream = nullptr);
    [[nodiscard]] HeterogeneousKVCheckpoint checkpoint(std::int32_t row,
                                                       cudaStream_t stream = nullptr);
    // Checkpoint copies and restore publication are ordered on stream; cross-stream use requires
    // the caller's usual event dependency.
    void restore(std::int32_t row, const HeterogeneousKVCheckpoint& checkpoint,
                 cudaStream_t stream = nullptr);
    // Raw copies are ordered on stream. Callers must synchronize before reading an exported
    // payload or releasing an imported payload. Import validates the exact retained logical
    // block set for every group before reserving or changing device state.
    [[nodiscard]] HeterogeneousKVHostImage export_host(std::int32_t row,
                                                       cudaStream_t stream = nullptr) const;
    void import_host(std::int32_t row, const HeterogeneousKVHostImage& image,
                     cudaStream_t stream = nullptr);

private:
    friend class HeterogeneousKVTransaction;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
