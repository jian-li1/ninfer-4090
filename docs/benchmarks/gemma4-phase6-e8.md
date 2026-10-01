# Gemma 4 Phase 6 E8 qualification

This record qualifies the production `rk4v4-e8` append codec for the two text-cache geometries of
`gemma4-31b-it/groupwise-int`. Measurements used an NVIDIA GeForce RTX 4090 (sm_89), CUDA 13.1,
the Release build, and commit state immediately before the Phase 6 commit.

## Correctness and numerical gate

`ninfer_gemma4_e8_kv_codec_test` calls the public `kv_cache_append` route and compares every packed
code byte and represented FP16 scale against an independent FP32 host oracle. It covers D256/H16
and D512/H4, both the small-token and page-tiled schedulers, a page crossing, nonidentity physical
page mapping, zero vectors, guard regions, and untouched sentinel storage. The test reported:

| geometry | plane | relative L2 range | cosine range |
|---|---|---:|---:|
| D256/H16 | K (E8 lattice) | 0.203940-0.206549 | 0.979152-0.979717 |
| D256/H16 | V (rotated INT4) | 0.188849-0.189257 | 0.982239-0.982379 |
| D512/H4 | K (E8 lattice) | 0.206276-0.206853 | 0.979107-0.979245 |
| D512/H4 | V (rotated INT4) | 0.188970-0.189368 | 0.982202-0.982262 |

The maintained gates are relative L2 at most 0.22 and cosine at least 0.975. Exact code/scale
parity is mandatory independently of those reconstruction gates.

The deterministic retrieval benchmark uses one D512 query, a controlled cosine-0.25 needle, and
independently generated unit-length distractors. It compares exact and decoded E8 scores:

| context | needle rank exact/E8 | recall@32 | score cosine |
|---:|---:|---:|---:|
| 32,768 | 1 / 1 | 0.8125 | 0.986487 |
| 65,536 | 1 / 1 | 0.7500 | 0.986418 |

The gate requires E8 needle rank at most 3, recall@32 at least 0.65, and score cosine at least
0.975. This is a codec retrieval qualification, not an end-to-end language-model quality claim.

## Production append throughput

Warm CUDA Graph measurements used 10 warmups and 100 repetitions. Useful bandwidth counts the
BF16 K/V inputs plus packed code/scale outputs.

| geometry | tokens | median us | useful GB/s |
|---|---:|---:|---:|
| D256/H16 local | 1 | 3.968 | 5.2 |
| D256/H16 local | 32 | 4.832 | 137.3 |
| D256/H16 local | 128 | 8.192 | 324.0 |
| D256/H16 local | 1024 | 40.960 | 518.4 |
| D512/H4 global | 1 | 3.776 | 2.7 |
| D512/H4 global | 32 | 4.096 | 81.0 |
| D512/H4 global | 128 | 5.952 | 223.0 |
| D512/H4 global | 1024 | 22.272 | 476.7 |

## Physical bytes

`rk4v4-e8` stores one K or V vector as D/2 packed-code bytes plus D/64 FP16 scales. This is 136
bytes at D256 and 272 bytes at D512, or 272 and 544 bytes respectively for K+V per token/head.

For all Gemma text layers, the local window occupies 222,822,400 bytes with 16 resident pages and
236,748,800 bytes at the 17-page partial-boundary maximum. The global group occupies 713,031,680
bytes at 32K and 1,426,063,360 bytes at 64K. Combined local/global payload is therefore
935,854,080-949,780,480 bytes at 32K and 1,648,885,760-1,662,812,160 bytes at 64K, excluding
allocator metadata, block tables, and transaction/checkpoint replicas.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build -j --target \
  ninfer_gemma4_e8_kv_codec_test \
  ninfer_kv_cache_append_bench \
  ninfer_gemma4_e8_retrieval_bench
./build/tests/ninfer_gemma4_e8_kv_codec_test
./build/bench/ninfer_kv_cache_append_bench \
  --mode full --full-geometry gemma-local --kv-dtype rk4v4-e8 \
  --tokens 1,32,128,1024 --context 128 --execution graph --cache warm \
  --warmup 10 --repeat 100
./build/bench/ninfer_kv_cache_append_bench \
  --mode full --full-geometry gemma-global --kv-dtype rk4v4-e8 \
  --tokens 1,32,128,1024 --context 128 --execution graph --cache warm \
  --warmup 10 --repeat 100
./build/bench/ninfer_gemma4_e8_retrieval_bench
```
