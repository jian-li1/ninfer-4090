# Gemma 4 Phase 8 global-attention qualification

This record qualifies the public D512, Hq=32, Hkv=4, GQA-8 causal full-attention Op on an
NVIDIA GeForce RTX 4090 (`sm_89`, CUDA 13.1, driver 580.178.04). The production cache is the
Phase 6 RK4V4-E8 persistent paged layout. Timings include cache append, attention, split reduction,
and inverse H64 output rotation.

## Production routes

| Route | Domain | CTA/tile | Workspace | Dispatch |
|---|---|---|---|---|
| shallow SIMT | page-local T=1..64 | 8 warps, Q tile 4, K tile 16 | none | visible keys `< max(128, 16*T)` |
| tensor-core split | page-local T=1..64 | 8 warps, Q tile 1..4, K tile 32, at most 32 splits | FP32 partial O/m/l, reused per four-token tile | visible keys `>= max(128, 16*T)` |

Both routes consume compressed pages directly and perform online causal softmax. Neither builds an
attention matrix or reconstructs a context-sized K/V tensor. The tensor-core route quantizes the
rotated query by 64-element group, uses signed-int8 QK MMA, reconstructs V into an FP16 shared tile,
uses FP16 PV MMA with FP32 accumulation, and shares every K/V head across eight query heads.

The dispatch threshold came from same-binary shallow-context measurements. At T=1, the split route
reduced 128/256/512-token latency from 110.304/207.872/403.712 us to
28.256/40.192/64.544 us. T=16 crossed over at 256 visible keys; the 256-token result changed from
498.688 us to 224.256 us. A 64-token query block remains on SIMT below 1K, where it measured
243.968 us at C=64 and 595.616 us at C=256.

## Correctness

The production Op is compared directly with a naive FP64 oracle. The oracle independently decodes
the stored i4 codes and represented FP16 scales, applies H64 to the public BF16 query, computes
stable causal softmax, accumulates V in FP64, and applies the inverse H64 transform. It does not call
another attention or codec kernel.

The matrix covers total visible lengths 128, 1,024, 4,096, 16,384, and 131,072; T=1,2,3,4,64;
all 32 heads at the route boundaries; representative heads at 16K/128K; nonidentity fragmented page
mapping; page tails; output/cache guards; and current K/V crossing the persistent codec boundary.
Acceptance is relative L2 <= 0.0025 and elementwise absolute error <=
`0.0003 + 0.003*abs(reference)`. Observed relative L2 was 0.000422-0.001592 and maximum absolute
error was at most 0.000225. The final test result was `OK gemma4_global_attention`.

## Performance matrix

Median public-Op latency in microseconds; seven measured launches after three warmups. T=128 and
T=1024 are submitted as two and sixteen page-local calls respectively. Workspace is bounded by
the four-token split tile: 0.50 MiB at 1K, 1.26 MiB at 4K, 4.27 MiB at 16K, and 8.03 MiB from 32K
through 256K.

| KV depth | T=1 | T=2 | T=4 | T=16 | T=128 | T=1024 |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 66.560 | 70.656 | 91.136 | 352.192 | 2,852.768 | 30,275.232 |
| 4K | 95.040 | 100.000 | 126.976 | 493.568 | 3,978.240 | 35,573.761 |
| 16K | 114.656 | 119.808 | 151.552 | 593.920 | 4,730.272 | 39,522.305 |
| 32K | 144.128 | 150.528 | 184.320 | 731.136 | 5,885.984 | 47,394.817 |
| 64K | 269.312 | 280.512 | 335.456 | 1,328.736 | 10,612.448 | 84,935.677 |
| 128K | 503.808 | 520.192 | 614.400 | 2,446.976 | 19,525.473 | 156,421.402 |
| 192K | 739.040 | 755.712 | 893.920 | 3,562.496 | 28,386.305 | 227,109.695 |
| 256K | 968.704 | 994.304 | 1,164.192 | 4,637.696 | 37,127.998 | 296,277.771 |

Deep T=1 decode scales monotonically from 144.128 us at 32K to 968.704 us at 256K. The useful
compressed payload rate rises from 494.7 GB/s at 32K to 588.9 GB/s at 256K. Relative to the initial
SIMT split implementation, T=1 improved from 2,825.216 to 503.808 us at 128K and from 5,613.568 to
968.704 us at 256K (5.61x and 5.79x).

The independent BF16 control is intentionally limited to <=16K because it is a correctness-first
materialized-cache kernel. For the complete public prefill operation, RK4V4-E8 measured 352.384 us
versus 919.392 us at C=1K/T=16, 3,995.584 versus 39,701.889 us at C=4K/T=128, and 37,492.737
versus 516,629.517 us at C=8K/T=1024: 2.61x, 9.94x, and 13.78x faster. Plain INT8 D512 was not
promoted as a target route: it is not a registered Gemma cache geometry and doubles K/V code bytes
relative to the selected RK4V4-E8 layout. This is a deliberate production-scope exclusion, not an
unmeasured claim that INT8 is slower.

## Static resources and profiler limitation

`cuobjdump --dump-resource-usage` reports the tensor-core partial kernels at 254 registers/thread
for T=1/2 and 255 for T=3/4, with zero stack and zero local spill. Static shared memory is 10,304
bytes for T=1/2 and 19,584 bytes for T=3/4; the K/V arena adds 65,536 bytes of opt-in dynamic shared
memory. The reducer uses 36 registers/thread, 8 bytes shared, and no stack/local spill.

Nsight Compute 2025.2.1 attached to the exact 128K production launch, but the host denied hardware
counter access with `ERR_NVGPUCTRPERM`. Consequently achieved occupancy, DRAM counters, L2 hit rate,
and tensor-pipe utilization are unavailable on this machine. Dequantization, softmax/reduction, and
page lookup were instead investigated by source/SASS ownership plus same-route timing: packed-code
load/dequant and QK/PV are fused in the partial kernel; the separate reducer is context-independent;
page lookup occurs once per 64-row page; and the 32K-256K latency/payload series above is stable and
linear. Counter collection remains a reproducible environment limitation, not a Phase 8 correctness
or use blocker.

## Reproduction

```bash
cmake --build build -j --target ninfer_gemma4_global_attention_test ninfer_gemma4_global_attention_bench
./build/tests/ninfer_gemma4_global_attention_test
./build/bench/ninfer_gemma4_global_attention_bench --context 1024,4096,16384,32768,65536,131072,196608,262144 --tokens 1,2,4,16,128,1024 --route optimized --warmup 3 --repeat 7
./build/bench/ninfer_gemma4_global_attention_bench --context 1024,4096,8192 --tokens 16,128,1024 --route both --warmup 3 --repeat 10
cuobjdump --dump-resource-usage build/bench/ninfer_gemma4_global_attention_bench | c++filt
ncu --kernel-name-base demangled --kernel-name regex:causal_attention_small_t_i8_tiled_kernel --launch-count 1 --section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis --section InstructionStats ./build/bench/ninfer_gemma4_global_attention_bench --context 131072 --tokens 1 --route optimized --warmup 0 --repeat 1
```
