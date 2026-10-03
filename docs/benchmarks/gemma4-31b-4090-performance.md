# Gemma 4 31B RTX 4090 performance qualification

This report qualifies the text-only `gemma4-31b-it/groupwise-int` target through the public
`.ninfer` Engine on one NVIDIA GeForce RTX 4090. It records the Phase 16 production profile,
correctness gates, memory residency, same-machine comparison, and remaining performance limits.
It is not a vision result.

## Result

The production profile starts with the full 262,144-token logical and physical KV capacity,
retains 1,353,383,936 bytes after startup without MTP and 868,941,824 bytes with MTP1, and passes
five-value semantic retrieval through a 260K-token prompt. NInfer does not allocate CUDA managed
memory; the qualified runs stayed in physical device memory.

No-spec decode is 6.7% faster than the target-only llama.cpp comparison on the shared 32K
retrieval prompt. MTP1 improves the representative mixed prompt by 29.4% with identical output.
Cold prefill exceeds llama.cpp at approximately 256K by 28.1%, but remains 43.5% slower at 64K
and 39.1% slower at 128K. The specification's cold-prefill goal is therefore not met at 64K or
128K. Nsight Systems attributes the remaining 64K time primarily to the Q4/BF16 linear route and
D512 global attention; `AllowA8` currently selects the same A16 Q4 implementation and is not an
activation-quantized route.

## Provenance

Measurements were collected on 2026-10-02 from the Phase 16 source revision: the commit containing
this report, whose parent is `5efa86f1` (`feat(serve): complete gemma4 public product integration`).
This definition remains exact if the branch is rebased; `git show --format=%H --no-patch` at this
document's revision identifies the measured commit.

| Item | Qualified value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB, `sm_89`, 480 W limit |
| Driver | 580.178.04 |
| CUDA compiler/runtime | 12.9.86 / 12.9 |
| Host compiler | GCC 13.3.0 |
| Build | Release, `sm_89`, 12 build jobs |
| NInfer artifact | `/dev/shm/gemma4_31b_it.ninfer`, 17,485,998,848 bytes |
| Artifact SHA-256 | `11d70c73f940930035516113a4b1cc25c685111692c4e9f2b345b430234ec83a` |
| Target weights | `groupwise-int`, Q4 G64 with FP16 scales |
| KV storage | heterogeneous `rk4v4-e8` |
| llama.cpp | 0.4.0-dev, build 10850, commit `f114f91f9`, GCC 13.3.0 |
| llama.cpp model | `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`, 17,287,668,064 bytes |

The artifact is local and contains the target plus the official assistant. The source checkpoint
revisions and conversion contract are recorded in the
[model](../maintainer/gemma4-31b-it-model.md) and
[artifact](../maintainer/gemma4-31b-it-artifact.md) references. The comparison model uses a
different mixed GGUF weight quantization and Q8 KV, so the comparison is a same-machine product
baseline rather than an isolated engine-only result.

## Production profile and method

The NInfer profile was one request, one resident model, CUDA Graph decode enabled, no prefix reuse,
greedy generation where output correctness was compared, full `--max-ctx 262144`, explicit
262,144-token KV capacity, 1,024-token prefill chunks, and `rk4v4-e8`. Long cold-prefill timings use
one repetition and no warmup because every repetition recomputes the full prefix. Decode at short
context uses three measured repetitions after one warmup. Depth decode uses one cold prefill and
32 generated tokens. The public benchmark reports GPU execution time exposed by the Engine, not
tokenization or process startup.

The deterministic throughput corpus is `bench/fixtures/bench_corpus.ids` (65,536 IDs). The
`--cycle-corpus` option repeats it exactly for deeper throughput prompts. Retrieval uses a separate
synthetic archive with five immutable key/value facts at 5%, 25%, 50%, 75%, and 95% of the prompt.
Success requires the five values to appear in the requested order; plausible free-form output is
not accepted.

Representative commands:

```bash
cmake --build build -j 12

./build/bench/ninfer_bench \
  --weights /dev/shm/gemma4_31b_it.ninfer \
  --cycle-corpus -p 4096,32768,65536,131072,196608,262016 \
  -r 1 --warmup 0 --max-ctx 262144 --prefill-chunk 1024 \
  --kv-dtype rk4v4-e8 --output json --output-file /tmp/gemma4-prefill.json

./build/bench/ninfer_bench \
  --weights /dev/shm/gemma4_31b_it.ninfer \
  --cycle-corpus -pg 262016,32 -r 1 --warmup 0 \
  --max-ctx 262144 --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
  --output json --output-file /tmp/gemma4-decode-262016.json

./build/bench/ninfer_gemma4_31b_it_retrieval \
  /dev/shm/gemma4_31b_it.ninfer --prompt-tokens 261120 \
  --max-context 262144 --prefill-chunk 1024 --generate 32
```

JSON report schema 15 records the complete command, artifact size, target identity, GPU/runtime,
load traffic, configuration, memory summary, individual repetitions, timings, and speculative
counters. `--cycle-corpus` is also recorded so a repeated corpus cannot be mistaken for a unique
262K corpus.

## Memory residency

These are startup reservations from public `Engine::memory_summary()`. Sequence capacity includes
the heterogeneous KV arena and target state; `kv_payload` isolates physical KV bytes.

| Maximum context | Weights | Sequence | Workspace | KV payload | Free after startup |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 16,971,062,784 | 565,413,376 | 301,479,936 | 565,411,840 | 6,982,139,904 |
| 32,768 | 16,971,062,784 | 1,189,319,680 | 312,532,992 | 1,189,314,560 | 6,344,605,696 |
| 65,536 | 16,971,062,784 | 1,902,355,456 | 312,532,992 | 1,902,346,240 | 5,631,574,016 |
| 131,072 | 16,971,062,784 | 3,328,427,008 | 312,532,992 | 3,328,409,600 | 4,205,510,656 |
| 196,608 | 16,971,062,784 | 4,754,498,560 | 312,532,992 | 4,754,472,960 | 2,779,447,296 |
| 262,144 | 16,971,062,784 | 6,180,570,112 | 312,532,992 | 6,180,536,320 | 1,353,383,936 |

With the assistant resident and MTP1 enabled at the full capacity, weights are 17,453,691,904
bytes, workspace is 312,554,496 bytes, and 868,941,824 bytes remain. Startup planning takes the
maximum of the mutually exclusive 1,024-token prefill workspace and seven-column MTP verifier
workspace; it does not reserve per-token logits for the entire prefill chunk.

NInfer has no `cudaMallocManaged`/`cuMemAllocManaged` allocation route. The 262K profiles fit below
the device limit with explicit remaining headroom, and the Nsight capture showed ordinary device
allocations and transfers rather than persistent Unified Memory paging.

## Long-context retrieval

Every depth returned exactly, in order:

```text
orange-telescope violet-compass silver-lantern cobalt-orchid amber-harbor
```

| Prompt tokens | Prefill | Decode (32 tokens) | Missing values |
|---:|---:|---:|---:|
| 32,768 | 25.340 s | 0.900 s | 0 |
| 65,536 | 57.953 s | 0.937 s | 0 |
| 131,072 | 146.788 s | 1.013 s | 0 |
| 196,608 | 267.054 s | 1.085 s | 0 |
| 245,760 | 363.459 s | 1.132 s | 0 |
| 261,120 | 412.128 s | 1.155 s | 0 |

The 32K prompt token IDs were exported once and submitted unchanged to llama.cpp. Both engines
returned the same five values, providing a semantic quality check across the NInfer Q4/RK4V4-E8
and llama.cpp UD-Q4_K_XL/Q8-KV profiles.

## Prefill throughput

| Prompt tokens | NInfer tok/s | NInfer time | llama.cpp tok/s | llama.cpp time | NInfer delta |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,465.8 | 2.794 s | — | — | — |
| 32,768 | 1,318.3 | 24.856 s | — | — | — |
| 65,536 | 1,139.2 | 57.529 s | 2,017.6 | 32.483 s | -43.5% |
| 131,072 | 895.9 | 146.298 s | 1,470.5 | 89.132 s | -39.1% |
| 196,608 | 738.2 | 266.325 s | — | — | — |
| 262,016 | 628.2 | 417.066 s | 490.5 | 534.186 s | +28.1% |

The measured NInfer points exceed the initial 64K/128K/256K aspirational rates of
1,000/700/350 tok/s. They do not satisfy the stronger same-machine llama.cpp goal at 64K and
128K. A 2,048-token chunk reached 1,151.8 tok/s at 64K and 908.9 tok/s at 128K, only 1.1–1.4%
above 1,024, while increasing workspace from about 298 MiB to 583 MiB and reducing startup
headroom by about 491 MiB. The production default therefore remains 1,024.

## Decode and MTP

No-spec depth results use 32 output tokens after the stated cold prompt:

| Prompt depth | Output tok/s | Decode time |
|---:|---:|---:|
| 32,768 | 34.53 | 0.927 s |
| 65,536 | 33.17 | 0.965 s |
| 131,072 | 30.68 | 1.043 s |
| 196,608 | 28.69 | 1.115 s |
| 262,016 | 26.90 | 1.190 s |

Short no-spec decode is 40.46 output tok/s for 128 tokens. On the shared 32K semantic prompt,
llama.cpp target-only decode reports 32.36 output tok/s and NInfer reports 34.53 output tok/s.
The llama.cpp MTP1 route reports 45.78 output tok/s with 13 accepted of 17 drafted tokens.

For the representative mixed prompt, NInfer no-spec produces 64 tokens at 40.5 tok/s. MTP1
produces the identical 64 token IDs at 52.4 tok/s: 39 rounds, 39 drafts, 25 accepted, 64.1%
acceptance, and a 29.4% end-to-end speedup. Widths 1–6 remain supported, but an intentionally
unfavorable 128-token generic-corpus sweep accepted no drafts and slowed monotonically from 32.62
tok/s at width 1 to 13.20 tok/s at width 6. Acceptance is therefore always reported and MTP1 is
the selected production width; speculative throughput is not claimed independently of workload.

The full 262,144-token MTP1 startup profile and graph-prime path also passed. Its unrelated
all-rejected short decode is an admission/graph test, not the positive performance claim.

## llama.cpp comparison configuration

The comparison used the user's local llama.cpp flags: all layers on GPU, flash attention,
`--parallel 1`, 12 CPU threads, 262,144 context, seed 3407, temperature 1.0, top-p 0.95, top-k 64,
presence penalty 1.05, repeat penalty 1.0, thinking enabled, and Q8_0 K/V cache. MTP measurements
added the official assistant with `--spec-type draft-mtp --spec-draft-n-max 1` and all assistant
layers on GPU. Cold prefill disabled prefix reuse. The configured disk/host prompt cache flags
were omitted to avoid writes outside this repository, and `--parallel 1` was made explicit. The
llama.cpp process used its configured `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`; NInfer did not.

Because the tokenizer implementations can differ, the throughput sweep fed each engine its own
fixed numeric corpus construction at the exact reported token count. The semantic comparison fed
the same exported 32,768-token ID array to both engines and used greedy output.

## Profiling and tuning decision

The 65,536-token public-Engine capture was collected with the benchmark's measured CUDA profiler
range and summarized with:

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  /tmp/ninfer-gemma4-phase16/ninfer-pp65536-final.nsys-rep
```

The measured prefill was 1,125.9 tok/s under profiling. GPU kernel time was:

| Route | GPU kernel time |
|---|---:|
| Q4 row-split BF16 tensor-core GEMM | 51.2% |
| D512 global small-T split attention, T6/W12/Bc32 | 22.4% |
| D512 global prompt attention | 12.8% |
| D256 sliding prompt attention | 6.3% |
| D512 tail T4/W8/Bc32 | 2.2% |
| D512 split reduction | 1.4% |

The exact Gemma MLP microbenchmark (`N=43008,K=5376,T=1024`) measured 3,561.5 us and 133.0
TFLOP/s for both A16 and `AllowA8`, confirming that Q4 has no distinct A8 route. A tested D512
T4/Bc16 schedule remained oracle-correct but regressed 64K attention from 3.86 ms to 5.58 ms and
128K from 6.99 ms to 10.29 ms, so it was rejected. Nsight Compute hardware counters were not
available (`ERR_NVGPUCTRPERM`); no counter-based claim is made.

The retained changes widen target execution to 1,024-token chunks, tile cache publication at the
64-token physical page boundary, add the D512 prompt route through 32K visible keys, use measured
T6 verifier/deep-context scheduling, avoid logits on intermediate prefill chunks, allocate exact
workspace, and update prefix hashing incrementally. Further 64K/128K gains require a qualified Q4
activation-quantized path and/or another D512 algorithm, not an unmeasured dispatch threshold.

## Verification

The final campaign builds all targets with `cmake --build build -j 12`. The complete CTest suite
runs serially with the real Gemma artifact and available Qwen3.8 artifact; unavailable optional
model bundles remain explicit skips. In addition to the full suite, the following production-route
checks are material:

- independent FP32/FP64 global-attention oracle across D512 routes;
- compute-sanitizer memcheck of the real Gemma Engine route with zero errors;
- public-Engine wide-chunk equivalence and full-context MTP admission;
- persistent-target smoke with the production codec;
- five-needle retrieval at all six depths;
- public Qwen3.8 real-model prefix test;
- benchmark schema/CLI tests for target-validated chunks, MTP width 6, and cycled corpus.

The Python retrieval wrapper was updated to invoke the public Engine qualifier, but was not run:
the repository's pinned Python 3.11 interpreter was unavailable in this environment, and no
unrelated Python environment was used. The C++ qualifier is the authoritative executed route.
