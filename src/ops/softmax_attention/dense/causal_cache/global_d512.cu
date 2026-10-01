#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kD = 512;
constexpr int kQHeads = 32;
constexpr int kKVHeads = 4;
constexpr int kGroupSize = 8;
constexpr int kGroups = 8;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kKeyTile = 16;
constexpr int kPackedD = kD / 2;
constexpr int kPrefillQueryTile = 4;
constexpr int kMaximumSplits = 32;
constexpr std::uint32_t kKeysPerSplitTarget = 1024;

static_assert(kWarps == kGroupSize);

struct Gemma4D512H32Kv4 {
    static constexpr int HeadDim = kD;
    static constexpr int QHeads = kQHeads;
    static constexpr int KVHeads = kKVHeads;
    static constexpr int GroupSize = kGroupSize;
    static constexpr int SmallTSplitScale = 1;
    static constexpr int SmallTMaximumSplits = kMaximumSplits;
};

__device__ __forceinline__ std::int8_t unpack_i4(std::uint8_t packed, bool high) {
    const unsigned nibble = high ? packed >> 4 : packed & 0x0fu;
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

template <int QueryTile, bool Split>
__global__ __launch_bounds__(kThreads) void causal_full_d512_kernel(
    const __nv_bfloat16* __restrict__ q, const std::int32_t* __restrict__ positions,
    const std::uint8_t* __restrict__ cache_k, const std::uint8_t* __restrict__ cache_v,
    const __half* __restrict__ scale_k, const __half* __restrict__ scale_v,
    const std::int32_t* __restrict__ block_table, int width, int split_count,
    int split_key_tiles, float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* q_s = reinterpret_cast<__half*>(shared_raw);
    auto* k_s = q_s + QueryTile * kGroupSize * kD;
    auto* v_s = k_s + kKeyTile * kD;

    constexpr unsigned FullMask = 0xffffffffu;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int split = Split ? static_cast<int>(blockIdx.y) : 0;
    const int query_tile = Split ? static_cast<int>(blockIdx.z) : static_cast<int>(blockIdx.y);
    const int query_begin = query_tile * QueryTile;
    const int valid_queries = min(QueryTile, width - query_begin);
    if (kv_head >= kKVHeads || valid_queries <= 0) return;

    const int q_head = kv_head * kGroupSize + warp;
    const int query_head_local = warp;
    for (int query = 0; query < QueryTile; ++query) {
        const bool valid = query < valid_queries;
#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
            const int d0 = group * 64 + lane;
            const int d1 = d0 + 32;
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (valid) {
                const std::int64_t base = static_cast<std::int64_t>(kD) *
                                          (q_head + kQHeads * (query_begin + query));
                x0 = __bfloat162float(q[base + d0]);
                x1 = __bfloat162float(q[base + d1]);
            }
            kv_cache_hadamard64(x0, x1, FullMask);
            const std::int64_t shared_base = static_cast<std::int64_t>(kD) *
                                             (query_head_local + kGroupSize * query);
            q_s[shared_base + d0] = __float2half_rn(x0);
            q_s[shared_base + d1] = __float2half_rn(x1);
        }
    }
    __syncthreads();

    int query_positions[QueryTile];
#pragma unroll
    for (int query = 0; query < QueryTile; ++query) {
        query_positions[query] =
            query < valid_queries ? positions[query_begin + query] : -1;
    }
    const int maximum_visible = query_positions[valid_queries - 1] + 1;
    const int first_key = Split ? split * split_key_tiles * kKeyTile : 0;
    const int split_limit = Split ? min(maximum_visible, first_key + split_key_tiles * kKeyTile)
                                  : maximum_visible;

    float running_m[QueryTile];
    float running_l[QueryTile];
    float accumulator[QueryTile][kD / 32];
#pragma unroll
    for (int query = 0; query < QueryTile; ++query) {
        running_m[query] = -CUDART_INF_F;
        running_l[query] = 0.0f;
#pragma unroll
        for (int item = 0; item < kD / 32; ++item) accumulator[query][item] = 0.0f;
    }

    for (int key_base = first_key; key_base < split_limit; key_base += kKeyTile) {
        const int physical_page = block_table[key_base >> kPagedKVPageShift];
#pragma unroll
        for (int key_local = 0; key_local < kKeyTile; ++key_local) {
            const int key = key_base + key_local;
            const int page_offset = key & kPagedKVPageMask;
            const int group = tid >> 5;
            float ks = 0.0f;
            float vs = 0.0f;
            std::uint8_t kc = 0;
            std::uint8_t vc = 0;
            if (key < split_limit) {
                const std::int64_t scale_offset =
                    paged_kv_element_offset<kGroups, kKVHeads>(physical_page, kv_head,
                                                               page_offset, group);
                ks = __half2float(scale_k[scale_offset]);
                vs = __half2float(scale_v[scale_offset]);
                const std::int64_t code_offset =
                    paged_kv_element_offset<kPackedD, kKVHeads>(physical_page, kv_head,
                                                                page_offset, tid);
                kc = cache_k[code_offset];
                vc = cache_v[code_offset];
            }
            const int d0 = 2 * tid;
            const std::int64_t dst = static_cast<std::int64_t>(key_local) * kD + d0;
            k_s[dst] = __float2half_rn(static_cast<float>(unpack_i4(kc, false)) * ks);
            k_s[dst + 1] = __float2half_rn(static_cast<float>(unpack_i4(kc, true)) * ks);
            v_s[dst] = __float2half_rn(static_cast<float>(unpack_i4(vc, false)) * vs);
            v_s[dst + 1] = __float2half_rn(static_cast<float>(unpack_i4(vc, true)) * vs);
        }
        __syncthreads();

#pragma unroll
        for (int key_local = 0; key_local < kKeyTile; ++key_local) {
            const int key = key_base + key_local;
            if (key >= split_limit) break;
#pragma unroll
            for (int query = 0; query < QueryTile; ++query) {
                if (query >= valid_queries || key > query_positions[query]) continue;
                const std::int64_t q_base = static_cast<std::int64_t>(kD) *
                                            (query_head_local + kGroupSize * query);
                const std::int64_t kv_base = static_cast<std::int64_t>(key_local) * kD;
                float score = 0.0f;
#pragma unroll
                for (int item = 0; item < kD / 32; ++item) {
                    const int d = lane + item * 32;
                    score = __fmaf_rn(__half2float(q_s[q_base + d]),
                                      __half2float(k_s[kv_base + d]), score);
                }
                score = warp_sum(score, FullMask);
                const float next_m = fmaxf(running_m[query], score);
                const float alpha = running_m[query] == -CUDART_INF_F
                                        ? 0.0f
                                        : expf(running_m[query] - next_m);
                const float probability = expf(score - next_m);
                running_l[query] = running_l[query] * alpha + probability;
                running_m[query] = next_m;
#pragma unroll
                for (int item = 0; item < kD / 32; ++item) {
                    const int d = lane + item * 32;
                    accumulator[query][item] =
                        __fmaf_rn(probability, __half2float(v_s[kv_base + d]),
                                  accumulator[query][item] * alpha);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int query = 0; query < QueryTile; ++query) {
        if (query >= valid_queries) continue;
        if constexpr (Split) {
            if (lane == 0) {
                const std::int64_t stat_index = q_head +
                    static_cast<std::int64_t>(kQHeads) *
                        (query_begin + query + static_cast<std::int64_t>(width) * split);
                partial_m[stat_index] = running_m[query];
                partial_l[stat_index] = running_l[query];
            }
#pragma unroll
            for (int item = 0; item < kD / 32; ++item) {
                const int d = lane + item * 32;
                const std::int64_t output_index = d + static_cast<std::int64_t>(kD) *
                    (q_head + static_cast<std::int64_t>(kQHeads) *
                        (query_begin + query + static_cast<std::int64_t>(width) * split));
                partial_acc[output_index] = accumulator[query][item];
            }
        } else {
            const float inverse = 1.0f / running_l[query];
#pragma unroll
            for (int item = 0; item < kD / 32; ++item) {
                const int d = lane + item * 32;
                const std::int64_t output_index = d + static_cast<std::int64_t>(kD) *
                    (q_head + static_cast<std::int64_t>(kQHeads) * (query_begin + query));
                out[output_index] = __float2bfloat16(accumulator[query][item] * inverse);
            }
        }
    }
}

__global__ void causal_full_d512_reduce_kernel(
    const float* __restrict__ partial_acc, const float* __restrict__ partial_m,
    const float* __restrict__ partial_l, int width, int split_count,
    __nv_bfloat16* __restrict__ out) {
    __shared__ float global_m;
    __shared__ float global_l;
    const int q_head = static_cast<int>(blockIdx.x);
    const int query = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        float maximum = -CUDART_INF_F;
        for (int split = 0; split < split_count; ++split) {
            const std::int64_t index = q_head + static_cast<std::int64_t>(kQHeads) *
                (query + static_cast<std::int64_t>(width) * split);
            maximum = fmaxf(maximum, partial_m[index]);
        }
        float denominator = 0.0f;
        for (int split = 0; split < split_count; ++split) {
            const std::int64_t index = q_head + static_cast<std::int64_t>(kQHeads) *
                (query + static_cast<std::int64_t>(width) * split);
            if (partial_l[index] > 0.0f) {
                denominator += partial_l[index] * expf(partial_m[index] - maximum);
            }
        }
        global_m = maximum;
        global_l = denominator;
    }
    __syncthreads();

    for (int d = tid; d < kD; d += blockDim.x) {
        float value = 0.0f;
        for (int split = 0; split < split_count; ++split) {
            const std::int64_t stat_index = q_head + static_cast<std::int64_t>(kQHeads) *
                (query + static_cast<std::int64_t>(width) * split);
            if (partial_l[stat_index] <= 0.0f) continue;
            const std::int64_t acc_index = d + static_cast<std::int64_t>(kD) *
                (q_head + static_cast<std::int64_t>(kQHeads) *
                    (query + static_cast<std::int64_t>(width) * split));
            value += partial_acc[acc_index] * expf(partial_m[stat_index] - global_m);
        }
        const std::int64_t output_index =
            d + static_cast<std::int64_t>(kD) * (q_head + kQHeads * query);
        out[output_index] = __float2bfloat16(value / global_l);
    }
}

template <int QueryTile>
constexpr int shared_bytes() {
    return static_cast<int>(sizeof(__half)) *
           (QueryTile * kGroupSize * kD + 2 * kKeyTile * kD);
}

template <int TokenTile>
void launch_tensor_core_global_d512(
    const Tensor& q, const Tensor& positions, CausalAttentionExecutionEnvelope envelope,
    const PagedKVLayerView& cache, int splits, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, cudaStream_t stream) {
    using Geometry = Gemma4D512H32Kv4;
    constexpr int kTensorCoreWarps = 8;
    constexpr int kTensorCoreKeyBlock = 32;
    constexpr int kDynamicBytes = 4 * kTensorCoreKeyBlock * Geometry::HeadDim;
    static const cudaError_t attribute = cudaFuncSetAttribute(
        causal_attention_small_t_i8_tiled_kernel<
            Geometry, TokenTile, kTensorCoreWarps, 1, kTensorCoreKeyBlock, true, true, true, true,
            true, true, false, false, false, CausalCachedInput>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicBytes);
    CUDA_CHECK(attribute);
    const dim3 grid(Geometry::KVHeads, splits, 1);
    causal_attention_small_t_i8_tiled_kernel<
        Geometry, TokenTile, kTensorCoreWarps, 1, kTensorCoreKeyBlock, true, true, true, true,
        true, true, false, false, false, CausalCachedInput>
        <<<grid, kTensorCoreWarps * 32, kDynamicBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), CausalCachedInput{},
            static_cast<const std::int32_t*>(positions.data),
            static_cast<std::int8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<__half*>(cache.k_scale_pages.data),
            static_cast<__half*>(cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(cache.block_table.data), nullptr, nullptr,
            cache.block_table.ne[0], TokenTile, 0,
            static_cast<std::int32_t>(envelope.max_visible_keys), 1.0f,
            static_cast<float*>(partial_acc.data), static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::int32_t causal_full_attention_split_capacity(CausalAttentionExecutionEnvelope envelope) {
    return std::min(kMaximumSplits,
                    static_cast<int>(div_up(envelope.max_visible_keys,
                                            kKeysPerSplitTarget)));
}

void causal_full_attention_launch(const Tensor& q, const Tensor& positions,
                                  CausalAttentionExecutionEnvelope envelope,
                                  const PagedKVLayerView& cache, Tensor& partial_acc,
                                  Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                  cudaStream_t stream) {
    const int width = q.ne[2];
    const auto* block_table = static_cast<const std::int32_t*>(cache.block_table.data);
    const auto* cache_k = static_cast<const std::uint8_t*>(cache.k_pages.data);
    const auto* cache_v = static_cast<const std::uint8_t*>(cache.v_pages.data);
    const auto* scale_k = static_cast<const __half*>(cache.k_scale_pages.data);
    const auto* scale_v = static_cast<const __half*>(cache.v_scale_pages.data);
    const bool split_route = partial_acc.data != nullptr;

    if (split_route) {
        const int splits = causal_full_attention_split_capacity(envelope);
        switch (width) {
        case 1:
            launch_tensor_core_global_d512<1>(q, positions, envelope, cache, splits, partial_acc,
                                                partial_m, partial_l, stream);
            break;
        case 2:
            launch_tensor_core_global_d512<2>(q, positions, envelope, cache, splits, partial_acc,
                                                partial_m, partial_l, stream);
            break;
        case 3:
            launch_tensor_core_global_d512<3>(q, positions, envelope, cache, splits, partial_acc,
                                                partial_m, partial_l, stream);
            break;
        case 4:
            launch_tensor_core_global_d512<4>(q, positions, envelope, cache, splits, partial_acc,
                                                partial_m, partial_l, stream);
            break;
        }
        const dim3 reduce_grid(kQHeads, static_cast<unsigned>(width));
        causal_full_d512_reduce_kernel<<<reduce_grid, kThreads, 0, stream>>>(
            static_cast<const float*>(partial_acc.data), static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data), width, splits,
            static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    } else {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            causal_full_d512_kernel<kPrefillQueryTile, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes<kPrefillQueryTile>());
        CUDA_CHECK(attribute);
        const dim3 grid(kKVHeads,
                        static_cast<unsigned>(div_up(width, kPrefillQueryTile)));
        causal_full_d512_kernel<kPrefillQueryTile, false>
            <<<grid, kThreads, shared_bytes<kPrefillQueryTile>(), stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int32_t*>(positions.data), cache_k, cache_v, scale_k,
                scale_v, block_table, width, 1, 0, nullptr, nullptr, nullptr,
                static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    }

    kv_cache_inverse_rotate_output_kernel<kQHeads, kD>
        <<<width * kQHeads * kGroups, 32, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data), width, width, 0, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
