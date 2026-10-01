# Gemma 4 Phase 7 sliding-attention qualification

This record qualifies the production local-attention Op for
`gemma4-31b-it/groupwise-int`. Measurements used an NVIDIA GeForce RTX 4090 (sm_89), CUDA 13.1,
the Release build, and commit state immediately before the Phase 7 commit.

## Production route

The public `causal_sliding_softmax_attention` profile is fixed to D256, Q32/KV16, a 1,024-token
causal window, attention scale 1.0, batch one, and one page-local segment of 1-64 tokens. Each call
publishes current K/V directly into the 17-slot circular RK4V4-E8 cache and then attends only the
visible logical window. The E8 K/V decode remains fused into the attention load path; there is no
BF16 cache materialization.

The SM89 attention CTA has 16 warps, a 64x64 query/key tile, 93,696 dynamic shared-memory bytes,
and a 128-register compile cap. The selected E8 specialization compiles to 119 registers/thread
with no stack or local-memory spill. Absolute logical pages are reduced modulo 17 only at the
paged-address boundary, so runtime block-table addresses remain stable for persistent-cache and
future CUDA Graph consumers.

## Numerical gate

`ninfer_gemma4_sliding_attention_test` calls the public Op and independently decodes the packed
E8 cache before computing a naive FP64 softmax-attention oracle. It covers total sequence lengths
1, 2, 127, 128, 1023, 1024, 1025, and 2048; nonidentity physical page mapping; page tails; the
1024/1025 overwrite boundary; repeated ring wraps; guard regions; and attention scale 1.0.

Across the reported boundary rows, relative L2 was 0.002065-0.002377 and maximum absolute error
was at most 0.001790. The maintained reduction gate is relative L2 at most 0.0035, with the shared
gross-error criterion of `0.0008 + 0.008 * max_reference`.

## RTX 4090 performance

Warm eager measurements used three warmups and ten repetitions. `optimized` is the complete public
Op, including E8 append, E8 load/decode, attention, and inverse rotation. `control` reproduces the
Phase 4 correctness-first transient-BF16 local-attention kernel and does not pay cache-publication
cost, making the comparison conservative for the optimized route.

| tokens | calls | optimized median us | control median us | speedup | optimized useful TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 2 | 52.448 | 245.696 | 4.68x | 5.16 |
| 256 | 4 | 117.760 | 930.816 | 7.90x | 9.15 |
| 512 | 8 | 290.784 | 3,607.552 | 12.41x | 14.80 |
| 1,024 | 16 | 803.776 | 14,236.672 | 17.71x | 21.39 |
| 2,048 | 32 | 2,177.024 | 42,585.983 | 19.56x | 23.68 |

One-token decode at a full 1,024-token local window measured 70.432 us median and 70.656 us p95.
That is 3.52 ms if all 50 local layers are summed in isolation. A separate split-reduction decode
kernel is deferred unless whole-model profiling shows this bounded cost is material; at long
context the ten unbounded global-attention layers are the expected attention bottleneck.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build -j --target \
  ninfer_gemma4_sliding_attention_test \
  ninfer_gemma4_sliding_attention_bench
NINFER_OP_REPORT_STATS=1 ./build/tests/ninfer_gemma4_sliding_attention_test
./build/bench/ninfer_gemma4_sliding_attention_bench \
  --tokens 128,256,512,1024,2048 --route both --warmup 3 --repeat 10
cuobjdump --dump-resource-usage build/src/libninfer_ops.a
```
