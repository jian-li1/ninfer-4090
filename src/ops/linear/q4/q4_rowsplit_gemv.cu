#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Schedule, int Tokens>
__device__ __forceinline__ void q4_gemv_dot_word_async_small_t(
    Q4GemvTileStorage<Schedule>& shared_tiles, int cta_warp,
    const __nv_bfloat16* __restrict__ activation,
    const std::uint8_t* __restrict__ code_row,
    const std::uint8_t* __restrict__ scale_row, int group_end, int lane,
    float (&sums)[Tokens]) {
    static_assert(Schedule::kWarpsPerRow == 1);
    static_assert(Tokens >= 2 && Tokens <= 7);
    constexpr int kGroupsPerTile = Schedule::kGroupsPerWarpTile;
    const int tile_count = (group_end + kGroupsPerTile - 1) / kGroupsPerTile;
#pragma unroll
    for (int token = 0; token < Tokens; ++token) { sums[token] = 0.0F; }
    for (int tile = 0; tile < tile_count; ++tile) {
        const int group_begin = tile * kGroupsPerTile;
        const int active_groups = min(kGroupsPerTile, group_end - group_begin);
        q4_gemv_issue_async_tile<Schedule>(shared_tiles.codes[cta_warp][0],
                                           shared_tiles.scale_pairs[cta_warp][0],
                                           code_row, scale_row, group_begin,
                                           active_groups, lane);
        cp_wait<0>();
        __syncwarp();
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            sums[token] = q4_gemv_consume_word_tile<Schedule>(
                shared_tiles.codes[cta_warp][0], shared_tiles.scale_pairs[cta_warp][0],
                activation + static_cast<std::int64_t>(token) * group_end * 64,
                group_begin, active_groups, lane, sums[token]);
        }
        __syncwarp();
    }
}

template <class Schedule, int Tokens>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kLaunchBoundsMinBlocks)
void q4_rowsplit_gemv_small_t_shared_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    std::int32_t rows, std::int32_t k) {
    static_assert(Schedule::kWarpsPerRow == 1);
    static_assert(Tokens >= 2 && Tokens <= 7);
    __shared__ Q4GemvTileStorage<Schedule> shared_tiles;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int cta_warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta + cta_warp;
    const int groups = k / Q4RowSplitStorage::kGroupK;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * groups * Q4RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * groups * Q4RowSplitStorage::kScaleBytesPerGroup;
    float sums[Tokens];
    q4_gemv_dot_word_async_small_t<Schedule, Tokens>(
        shared_tiles, cta_warp, x, code_row, scale_row, groups, lane, sums);
#pragma unroll
    for (int token = 0; token < Tokens; ++token) { sums[token] = warp_reduce_sum(sums[token]); }
    if (lane == 0) {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            out[static_cast<std::int64_t>(token) * rows + row] =
                __float2bfloat16(sums[token]);
        }
    }
}

template <class Schedule>
void launch_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    const std::size_t dynamic_shared_bytes =
        Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK
            ? static_cast<std::size_t>(k) * sizeof(__nv_bfloat16)
            : 0u;

    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, dynamic_shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), nullptr,
        rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_gemv<Q4GemvR4W1DirectSchedule>(x, w, out, stream);
}

void launch_q4_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_gemv<Q4GemvR1W8DirectSchedule>(x, w, out, stream);
}

template <int Tokens>
void launch_small_t_shared(const Tensor& x, const Weight& w, Tensor& out,
                           cudaStream_t stream) {
    using Schedule = Q4GemvR4W1DirectSchedule;
    const std::int32_t rows = out.ne[0];
    const std::int32_t k = x.ne[0];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_small_t_shared_kernel<Schedule, Tokens><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4_gemv_r4_w1_small_t_shared(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2: return launch_small_t_shared<2>(x, w, out, stream);
    case 3: return launch_small_t_shared<3>(x, w, out, stream);
    case 4: return launch_small_t_shared<4>(x, w, out, stream);
    case 5: return launch_small_t_shared<5>(x, w, out, stream);
    case 6: return launch_small_t_shared<6>(x, w, out, stream);
    case 7: return launch_small_t_shared<7>(x, w, out, stream);
    default: throw std::invalid_argument("Gemma Q4 shared small-T route requires T in [2,7]");
    }
}

} // namespace ninfer::ops::detail
