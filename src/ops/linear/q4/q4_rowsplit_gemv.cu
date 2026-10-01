#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
__device__ __forceinline__ void q4_gemv_dot_word_async_t2(
    Q4GemvTileStorage<Schedule>& shared_tiles, int cta_warp,
    const __nv_bfloat16* __restrict__ activation,
    const std::uint8_t* __restrict__ code_row,
    const std::uint8_t* __restrict__ scale_row, int group_end, int lane,
    float& first, float& second) {
    static_assert(Schedule::kWarpsPerRow == 1);
    constexpr int kGroupsPerTile = Schedule::kGroupsPerWarpTile;
    const int tile_count = (group_end + kGroupsPerTile - 1) / kGroupsPerTile;
    const __nv_bfloat16* second_activation = activation + group_end * 64;
    first = 0.0F;
    second = 0.0F;
    for (int tile = 0; tile < tile_count; ++tile) {
        const int group_begin = tile * kGroupsPerTile;
        const int active_groups = min(kGroupsPerTile, group_end - group_begin);
        q4_gemv_issue_async_tile<Schedule>(shared_tiles.codes[cta_warp][0],
                                           shared_tiles.scale_pairs[cta_warp][0],
                                           code_row, scale_row, group_begin,
                                           active_groups, lane);
        cp_wait<0>();
        __syncwarp();
        first = q4_gemv_consume_word_tile<Schedule>(
            shared_tiles.codes[cta_warp][0], shared_tiles.scale_pairs[cta_warp][0],
            activation, group_begin, active_groups, lane, first);
        second = q4_gemv_consume_word_tile<Schedule>(
            shared_tiles.codes[cta_warp][0], shared_tiles.scale_pairs[cta_warp][0],
            second_activation, group_begin, active_groups, lane, second);
        __syncwarp();
    }
}

template <class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kLaunchBoundsMinBlocks)
void q4_rowsplit_gemv_t2_shared_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    std::int32_t rows, std::int32_t k) {
    static_assert(Schedule::kWarpsPerRow == 1);
    __shared__ Q4GemvTileStorage<Schedule> shared_tiles;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int cta_warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta + cta_warp;
    const int groups = k / Q4RowSplitStorage::kGroupK;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * groups * Q4RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * groups * Q4RowSplitStorage::kScaleBytesPerGroup;
    float first = 0.0F;
    float second = 0.0F;
    q4_gemv_dot_word_async_t2<Schedule>(shared_tiles, cta_warp, x, code_row,
                                        scale_row, groups, lane, first, second);
    first = warp_reduce_sum(first);
    second = warp_reduce_sum(second);
    if (lane == 0) {
        out[row] = __float2bfloat16(first);
        out[static_cast<std::int64_t>(rows) + row] = __float2bfloat16(second);
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

void launch_q4_gemv_r4_w1_t2_shared(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    using Schedule = Q4GemvR4W1DirectSchedule;
    const std::int32_t rows = out.ne[0];
    const std::int32_t k = x.ne[0];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_t2_shared_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
