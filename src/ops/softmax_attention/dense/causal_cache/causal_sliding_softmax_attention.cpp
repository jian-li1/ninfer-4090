#include "ninfer/ops/softmax_attention.h"

#include "core/paged_kv_storage.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads = 32;
constexpr std::int32_t kKVHeads = 16;
constexpr std::int32_t kWindow = 1024;
constexpr std::int32_t kRingPages = 17;
constexpr float kScale = 1.0F;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != n3) {
        throw std::invalid_argument("causal_sliding_softmax_attention: invalid shape for " +
                                    std::string(name));
    }
}

void require_contiguous(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument("causal_sliding_softmax_attention: " + std::string(name) +
                                    " must be contiguous and non-null");
    }
}

void validate_cache(const PagedKVLayerView& cache) {
    const auto layout = paged_kv_storage_layout(KvCacheStorage::RK4V4E8, kHeadDim);
    const std::int32_t physical_pages = cache.k_pages.ne[3];
    if (cache.storage != KvCacheStorage::RK4V4E8 || cache.head_dim != kHeadDim ||
        cache.num_kv_heads != kKVHeads || physical_pages <= 0 ||
        cache.v_pages.ne[3] != physical_pages || cache.block_table.ne[0] != kRingPages) {
        throw std::invalid_argument("causal_sliding_softmax_attention: invalid cache profile");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kKVHeads,
                  physical_pages, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kKVHeads,
                  physical_pages, "cache v pages");
    require_shape(cache.k_scale_pages, layout.key.scale_leading_extent, kPagedKVPageSize, kKVHeads,
                  physical_pages, "cache k scale pages");
    require_shape(cache.v_scale_pages, layout.value.scale_leading_extent, kPagedKVPageSize,
                  kKVHeads, physical_pages, "cache v scale pages");
    require_shape(cache.block_table, kRingPages, 1, 1, 1, "block table");
    if (cache.k_pages.dtype != DType::U8 || cache.v_pages.dtype != DType::U8 ||
        cache.k_scale_pages.dtype != DType::FP16 ||
        cache.v_scale_pages.dtype != DType::FP16 || cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument("causal_sliding_softmax_attention: invalid cache dtype");
    }
    require_contiguous(cache.k_pages, "cache k pages");
    require_contiguous(cache.v_pages, "cache v pages");
    require_contiguous(cache.k_scale_pages, "cache k scale pages");
    require_contiguous(cache.v_scale_pages, "cache v scale pages");
    require_contiguous(cache.block_table, "block table");
}

} // namespace

void causal_sliding_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& positions, AttentionHeadGeometry geometry,
                                      std::uint32_t window, float scale, PagedKVLayerView cache,
                                      Tensor& out, cudaStream_t stream) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kQHeads || geometry.kv_heads != kKVHeads || window != kWindow) {
        throw std::invalid_argument("causal_sliding_softmax_attention: unsupported profile");
    }
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16 || positions.dtype != DType::I32) {
        throw std::invalid_argument("causal_sliding_softmax_attention: invalid input dtype");
    }
    if (!std::isfinite(scale) || std::abs(scale - kScale) > 1.0e-7F) {
        throw std::invalid_argument("causal_sliding_softmax_attention: scale must be 1.0");
    }
    const std::int32_t tokens = q.ne[2];
    if (tokens < 1 || tokens > kPagedKVPageSize) {
        throw std::invalid_argument(
            "causal_sliding_softmax_attention: page-local T must be 1..64");
    }
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
    validate_cache(cache);

    // The caller supplies one sequential page-local segment. Publishing it before attention is
    // safe because entering the next modulo slot makes the evicted page older than W=1024 for
    // every query in this segment. Current rows therefore cross the same E8 boundary as history.
    detail::kv_cache_append_sliding_launch(k, v, positions, cache, stream);
    if (tokens <= 7) {
        // Speculative verification must retain ordinary T=1 arithmetic for every possible
        // committed prefix, including the bonus column after all drafts match.
        for (std::int32_t token = 0; token < tokens; ++token) {
            const Tensor q_token = q.slice(2, token, 1);
            const Tensor position = positions.slice(0, token, 1);
            Tensor out_token = out.slice(2, token, 1);
            detail::causal_sliding_attention_launch(
                q_token, position, scale, cache, out_token, stream);
        }
    } else {
        detail::causal_sliding_attention_launch(q, positions, scale, cache, out, stream);
    }
}

void shared_kv_sliding_softmax_attention(const Tensor& q,
                                         const Tensor& last_key_positions,
                                         AttentionHeadGeometry geometry,
                                         std::uint32_t window, float scale,
                                         const PagedKVLayerView& cache, Tensor& out,
                                         cudaStream_t stream) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kQHeads || geometry.kv_heads != kKVHeads ||
        window != kWindow) {
        throw std::invalid_argument("shared_kv_sliding_softmax_attention: unsupported profile");
    }
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        last_key_positions.dtype != DType::I32 || q.ne[2] != 1) {
        throw std::invalid_argument("shared_kv_sliding_softmax_attention: invalid input");
    }
    if (!std::isfinite(scale) || std::abs(scale - kScale) > 1.0e-7F) {
        throw std::invalid_argument("shared_kv_sliding_softmax_attention: scale must be 1.0");
    }
    require_shape(q, kHeadDim, kQHeads, 1, 1, "q");
    require_shape(last_key_positions, 1, 1, 1, 1, "last key positions");
    require_shape(out, kHeadDim, kQHeads, 1, 1, "out");
    require_contiguous(q, "q");
    require_contiguous(last_key_positions, "last key positions");
    require_contiguous(out, "out");
    validate_cache(cache);
    detail::causal_sliding_attention_launch(q, last_key_positions, scale, cache, out, stream);
}

} // namespace ninfer::ops
