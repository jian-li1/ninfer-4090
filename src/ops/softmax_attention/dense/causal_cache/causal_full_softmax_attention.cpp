#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 512;
constexpr std::int32_t kQHeads = 32;
constexpr std::int32_t kKVHeads = 4;
constexpr std::int32_t kGroups = 8;
constexpr std::int32_t kMaximumPageLocalTokens = 64;
constexpr float kScale = 1.0F;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != n3) {
        throw std::invalid_argument("causal_full_softmax_attention: invalid shape for " +
                                    std::string(name));
    }
}

void require_contiguous(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument("causal_full_softmax_attention: " + std::string(name) +
                                    " must be contiguous and non-null");
    }
}

std::uint32_t validate_cache(const PagedKVLayerView& cache) {
    const auto layout = paged_kv_storage_layout(KvCacheStorage::RK4V4E8, kHeadDim);
    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages = cache.block_table.ne[0];
    const std::int64_t capacity =
        static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (cache.storage != KvCacheStorage::RK4V4E8 || cache.head_dim != kHeadDim ||
        cache.num_kv_heads != kKVHeads || physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("causal_full_softmax_attention: invalid cache profile");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kKVHeads,
                  physical_pages, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kKVHeads,
                  physical_pages, "cache v pages");
    require_shape(cache.k_scale_pages, layout.key.scale_leading_extent, kPagedKVPageSize,
                  kKVHeads, physical_pages, "cache k scale pages");
    require_shape(cache.v_scale_pages, layout.value.scale_leading_extent, kPagedKVPageSize,
                  kKVHeads, physical_pages, "cache v scale pages");
    require_shape(cache.block_table, logical_pages, 1, 1, 1, "block table");
    if (cache.k_pages.dtype != DType::U8 || cache.v_pages.dtype != DType::U8 ||
        cache.k_scale_pages.dtype != DType::FP16 ||
        cache.v_scale_pages.dtype != DType::FP16 || cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument("causal_full_softmax_attention: invalid cache dtype");
    }
    require_contiguous(cache.k_pages, "cache k pages");
    require_contiguous(cache.v_pages, "cache v pages");
    require_contiguous(cache.k_scale_pages, "cache k scale pages");
    require_contiguous(cache.v_scale_pages, "cache v scale pages");
    require_contiguous(cache.block_table, "block table");
    return static_cast<std::uint32_t>(capacity);
}

void validate_profile(AttentionHeadGeometry geometry, KvCacheStorage storage,
                      CausalAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
                      std::int32_t max_tokens, const char* operation) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kQHeads || geometry.kv_heads != kKVHeads ||
        storage != KvCacheStorage::RK4V4E8 || min_tokens < 1 || max_tokens < min_tokens ||
        max_tokens > kMaximumPageLocalTokens || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument(std::string(operation) + ": invalid profile or interval");
    }
}

std::uint32_t tensor_core_route_threshold(std::int32_t tokens) {
    return std::max<std::uint32_t>(128, static_cast<std::uint32_t>(tokens) * 16);
}

template <class Allocator>
void allocate_partials(Allocator& workspace, std::int32_t tokens, std::int32_t splits,
                       Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l) {
    partial_acc = workspace.alloc(DType::FP32, {kHeadDim, kQHeads, tokens, splits});
    partial_m = workspace.alloc(DType::FP32, {kQHeads, tokens, splits});
    partial_l = workspace.alloc(DType::FP32, {kQHeads, tokens, splits});
}

} // namespace

std::size_t causal_full_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage,
    CausalAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens) {
    validate_profile(geometry, cache_storage, envelope, min_tokens, max_tokens,
                     "causal_full_softmax_attention workspace");
    if (envelope.max_visible_keys < tensor_core_route_threshold(min_tokens)) return 0;
    const std::int32_t tokens = std::min(max_tokens, 4);
    const std::int32_t splits = detail::causal_full_attention_split_capacity(envelope);
    WorkspaceLayoutBuilder layout;
    Tensor acc;
    Tensor m;
    Tensor l;
    allocate_partials(layout, tokens, splits, acc, m, l);
    return layout.peak_bytes(1);
}

void causal_full_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& positions, AttentionHeadGeometry geometry,
                                   float scale, PagedKVLayerView cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out,
                                   cudaStream_t stream) {
    validate_profile(geometry, cache.storage, envelope, q.ne[2], q.ne[2],
                     "causal_full_softmax_attention");
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16 || positions.dtype != DType::I32) {
        throw std::invalid_argument("causal_full_softmax_attention: invalid input dtype");
    }
    if (!std::isfinite(scale) || std::abs(scale - kScale) > 1.0e-7F) {
        throw std::invalid_argument("causal_full_softmax_attention: scale must be 1.0");
    }
    const std::int32_t tokens = q.ne[2];
    require_shape(q, kHeadDim, kQHeads, tokens, 1, "q");
    require_shape(k, kHeadDim, kKVHeads, tokens, 1, "k");
    require_shape(v, kHeadDim, kKVHeads, tokens, 1, "v");
    require_shape(positions, tokens, 1, 1, 1, "positions");
    require_shape(out, kHeadDim, kQHeads, tokens, 1, "out");
    require_contiguous(q, "q");
    require_contiguous(k, "k");
    require_contiguous(v, "v");
    require_contiguous(positions, "positions");
    require_contiguous(out, "out");
    const std::uint32_t cache_capacity = validate_cache(cache);
    if (envelope.max_visible_keys > cache_capacity) {
        throw std::invalid_argument(
            "causal_full_softmax_attention: execution envelope exceeds cache");
    }

    detail::kv_cache_append_launch(k, v, positions, cache, stream);
    auto workspace_scope = workspace.scope();
    Tensor partial_acc;
    Tensor partial_m;
    Tensor partial_l;
    if (envelope.max_visible_keys < tensor_core_route_threshold(tokens)) {
        detail::causal_full_attention_launch(q, positions, envelope, cache, partial_acc, partial_m,
                                             partial_l, out, stream);
        return;
    }

    // Gemma MTP verifies up to seven tokens at once. Use the decode-width kernel independently
    // for each query so every accepted-prefix column is bitwise identical to ordinary T=1.
    const std::int32_t tile_tokens = tokens <= 7 ? 1 : std::min(tokens, 4);
    const std::int32_t splits = detail::causal_full_attention_split_capacity(envelope);
    allocate_partials(workspace, tile_tokens, splits, partial_acc, partial_m, partial_l);
    for (std::int32_t begin = 0; begin < tokens; begin += tile_tokens) {
        const std::int32_t count = std::min(tile_tokens, tokens - begin);
        const Tensor q_tile = q.slice(2, begin, count);
        const Tensor positions_tile = positions.slice(0, begin, count);
        Tensor out_tile = out.slice(2, begin, count);
        detail::causal_full_attention_launch(q_tile, positions_tile, envelope, cache, partial_acc,
                                             partial_m, partial_l, out_tile, stream);
    }
}

void shared_kv_full_softmax_attention(const Tensor& q,
                                      const Tensor& last_key_positions,
                                      AttentionHeadGeometry geometry, float scale,
                                      const PagedKVLayerView& cache,
                                      CausalAttentionExecutionEnvelope envelope,
                                      WorkspaceArena& workspace, Tensor& out,
                                      cudaStream_t stream) {
    validate_profile(geometry, cache.storage, envelope, 1, 1,
                     "shared_kv_full_softmax_attention");
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        last_key_positions.dtype != DType::I32) {
        throw std::invalid_argument("shared_kv_full_softmax_attention: invalid input dtype");
    }
    if (!std::isfinite(scale) || std::abs(scale - kScale) > 1.0e-7F) {
        throw std::invalid_argument("shared_kv_full_softmax_attention: scale must be 1.0");
    }
    require_shape(q, kHeadDim, kQHeads, 1, 1, "q");
    require_shape(last_key_positions, 1, 1, 1, 1, "last key positions");
    require_shape(out, kHeadDim, kQHeads, 1, 1, "out");
    require_contiguous(q, "q");
    require_contiguous(last_key_positions, "last key positions");
    require_contiguous(out, "out");
    const std::uint32_t cache_capacity = validate_cache(cache);
    if (envelope.max_visible_keys > cache_capacity) {
        throw std::invalid_argument(
            "shared_kv_full_softmax_attention: execution envelope exceeds cache");
    }

    auto workspace_scope = workspace.scope();
    Tensor partial_acc;
    Tensor partial_m;
    Tensor partial_l;
    if (envelope.max_visible_keys >= tensor_core_route_threshold(1)) {
        const std::int32_t splits = detail::causal_full_attention_split_capacity(envelope);
        allocate_partials(workspace, 1, splits, partial_acc, partial_m, partial_l);
    }
    detail::causal_full_attention_launch(q, last_key_positions, envelope, cache,
                                         partial_acc, partial_m, partial_l, out, stream);
}

} // namespace ninfer::ops
