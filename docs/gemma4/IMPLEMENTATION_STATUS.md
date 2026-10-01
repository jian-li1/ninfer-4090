# Gemma 4 31B implementation status

This is the current execution log for the text-only `gemma4-31b-it/groupwise-int` target on one
RTX 4090. It tracks the phase gates in the implementation specification and is updated in the same
commit that completes each phase.

## Scope

- Target: `google/gemma-4-31B-it-qat-w4a16-ct`
- Optional assistant: `google/gemma-4-31B-it-assistant`
- Runtime artifact: `.ninfer` container v2
- Device target: `sm_89`
- Product route: public `ninfer::Engine`
- Required compatibility: persistent prefix/cache implementation inherited from
  `feat/persist-kv-cache`
- Excluded: optional multimodal/vision extension

## Phase status

| Phase | Status | Evidence or gate |
|---:|---|---|
| 0 | complete | Baseline environment, full test suites, and Qwen3.8 public-Engine smoke recorded |
| 1 | complete | Immutable sources, compact semantic fixtures, real target logits, and assistant MTP1 |
| 2 | complete | Compile-time family/target packages, exact schedule, capabilities, and registry identity |
| 3 | complete | Reproducible `.ninfer` converter, closed binder, and 15.806 GiB target residency |
| 4 | complete | Reference-correct C++/CUDA short-context execution and first-divergence parity |
| 5 | complete | Atomic heterogeneous local/global KV groups qualified through 32K |
| 6 | complete | Exact E8 parity, retrieval gates, production throughput, and physical bytes for D256/D512 |
| 7 | complete | D256 Q32/KV16 E8 ring attention is 4.68-19.56x faster than the reference path |
| 8 | complete | D512 GQA-8 global attention is correct through 128K and stable through 256K |
| 9 | complete | 262K allocation plus exact five-needle 240K full-model semantic retrieval |
| 10 | deferred/conditional | Re-evaluate after graph, MTP, selected draft width, and server residency measurements |
| 11 | complete | One graph spans absolute positions, ring wrap, rollback/restore, and 262K with a 2.13% decode benefit |
| 12 | complete | Official MTP1 with exact greedy parity, atomic prefix commit, graph replay, and positive 4K speedup |
| 13 | pending | MTP2 through MTP6 measurement and selection |
| 14 | pending | Prefix reuse and persistent continuation qualification |
| 15 | pending | Full server/product integration |
| 16 | pending | Performance and regression qualification |
| 17 | pending | Documentation and release packaging |

The optional multimodal/vision extension after Phase 17 is intentionally not scheduled.

## Phase 0 record

- Branch `feat/gemma-4-31b` starts at
  `3a7a0d4ecc250c92dcf22c6f3d074578be8ca2d6` from `feat/persist-kv-cache`.
- Release `sm_89` build succeeded.
- CTest: 121/121 passed; 109 executed and 12 expected skips.
- Python artifact/converter suite: 76 passed and 3 source-gated skips on CPython 3.11.13.
- Qwen3.8 groupwise public-Engine smoke succeeded with MTP3 and INT8 KV.
- Full environment, commands, skip reasons, artifact hash, and smoke metrics are in
  [the baseline record](../benchmarks/gemma4-baseline-environment.md).

## Pinned sources

These revisions are immutable. Small-resource hashes and every fixture-array hash are recorded in
`tests/fixtures/gemma4_31b_it/metadata.json`.

| Source | Revision | Checkpoint SHA-256 |
|---|---|---|
| `google/gemma-4-31B-it-qat-w4a16-ct` | `52f3f65bc7a02d555763bc923bd1d9094898219d` | `1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20` |
| `google/gemma-4-31B-it-assistant` | `627c5ec1458b9086b841a91e0512fd31fd2fbbf1` | `9f80df6099fa1fd7db71220ec9ee864d5ecff769878697dd5763e4285b15a1da` |
| Target tokenizer resources | target revision above | hashes in fixture metadata |
| Hugging Face Transformers | `d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a` | n/a |

## Phase 1 record

- Pinned the public target, assistant, tokenizer, and Transformers implementation to the revisions
  above; both complete checkpoint SHA-256 values match their official linked hashes.
- The compact 108 KiB fixture archive covers every unique text-model operation and the immutable
  assistant layer-to-target K/V mapping.
- The source-gated generator independently decoded and executed all 60 target layers, captured 68
  checkpoint arrays including full-model logits, and ran official assistant MTP1.
- The target INT4/group-32 decoder matched compressed-tensors 0.15.1.a20260521 bit-for-bit on real
  stored weights and scales.
- Pinned Transformers reproduced the assistant feedback vector and all 262,144 MTP1 logits exactly,
  including the real shared target K/V fixture.
- A second complete source-checkpoint generation reproduced all 68 canonical array hashes.
- Tests: 9/9 Phase 1 tests passed with source checkpoints; maintained Python suites passed 84/84
  with four expected skips; the separately packaged eval suite passed 19/19 plus three subtests.
- Qwen3.8 public-Engine regression smoke remained healthy: 125.8 decode tok/s, 99.0 overall tok/s,
  62.5% MTP acceptance, and 5.80 GiB free after startup.

## Phase 2 record

- Added peer `gemma4` family and `gemma4_31b_it` target packages behind
  `NINFER_BUILD_GEMMA4_31B_IT`, enabled by default.
- The immutable target configuration records all target and assistant dimensions, attention
  geometry, RoPE parameters, quantization geometry, target K/V sharing, and layer types.
- The generated target schedule is compile-time validated as 60 layers: 50 sliding and 10 full,
  with full attention exactly at zero-based layers 5, 11, 17, ..., 59.
- Model views represent the target and optional companion assistant without device allocation or
  execution code. Capabilities are text required, MTP artifact-optional, and vision/audio
  unavailable.
- The cold registry identifies the exact synthetic identity
  `gemma4-31b-it/groupwise-int`; existing Qwen identity lookup remains covered.
- Gemma-enabled Engine and configuration-test builds passed, the configuration/registry test
  passed, and an independent Gemma-disabled Engine build passed. No Gemma inference was attempted.

The optional all-target rebuild could not finish because the root filesystem exhausted its final
2.1 GiB while linking unrelated test executables, even after reducing the build to two jobs. The
Phase 0 full-suite result remains the regression baseline; all Phase 2 affected targets passed.

## Phase 3 record

- Added the reproducible container-v2 converter for the exact pinned target and optional assistant.
  It validates checkpoint SHA-256, config, compressed-tensors group-32 semantics, complete source
  classification, learned layer scalars, resources, and every materialized signature before write.
- The target inventory has 662 device tensors and four host resources. The tied output head aliases
  the row-scaled FP8 embedding; Q/K/V and gate/up are row-fused; source INT4 group-32 linears are
  decoded and requantized to the existing Q4G64 persistent format.
- Exactly 356 multimodal source tensors are deliberately omitted. Unknown source tensors, artifact
  extras, missing objects, and malformed tensor signatures fail closed.
- The optional 44-tensor assistant is packaged in W8/FP8/BF16 form and fully validated, but remains
  nonresident until the MTP execution phase.
- Target device residency is 16,971,062,784 bytes (15.806 GiB), below the 17.5 GiB planning
  checkpoint. The assistant package adds 482,629,120 artifact bytes without Phase 3 residency.
- The locally generated target-plus-assistant artifact has 710 objects, 17,485,998,848 file bytes,
  and SHA-256 `11d70c73f940930035516113a4b1cc25c685111692c4e9f2b345b430234ec83a`.
  All 710 per-object hashes were independently verified after reopening the artifact.
- Representative matrices measured maximum relative L2 0.098805 and minimum cosine similarity
  0.995299. The FP8 embedding sample measured relative L2 0.025692 and cosine 0.999670.
- Ten affected Python converter/numeric tests passed. Both affected C++ tests passed, including the
  closed synthetic 15.806 GiB load plan; the C++ reader/binder also accepted the real artifact.

## Phase 4 record

- Added a target-owned correctness-first C++/CUDA execution route for 1–4096 token prefixes. It
  executes the complete 60-layer schedule over the registered `.ninfer` artifact without a Python
  model-inference path or multimodal residency.
- The implementation covers direct-gain RMSNorm, BF16 embedding scaling, full and proportional
  RoPE, unscaled grouped-query causal attention, the global K-as-V rule with distinct K/V norms,
  dense tanh-GELU MLPs, learned layer scalars, tied FP8 output projection, and pre-selection logit
  softcap.
- The route uses transient BF16 current-prefix K/V and owns no persistent cache allocation. It is
  therefore compatible with the inherited persistent-cache contracts while Phase 5 introduces
  heterogeneous physical cache groups.
- The pinned six-token real-checkpoint prompt produced reference greedy token `100`. All 25
  exported substages passed the production-artifact gate: worst relative L2 0.290551, minimum
  cosine 0.959352, and no first divergence. Layer-0 and layer-5 output cosines were 0.999469 and
  0.999482 respectively; final-logit cosine was 0.994011.
- The T=1 path also completed on the real artifact. The Gemma C++ suite passed 3/3, and the
  affected shared embedding and Q4 oracle tests passed 2/2 on the RTX 4090/CUDA 13.1 toolchain.
- The dump comparator reports stage ownership, checksums, max/mean absolute error, relative L2,
  cosine similarity, selected slices, first divergence, and greedy-token equality.

## Phase 5 record

- Added target-neutral heterogeneous KV planning and runtime ownership over the existing physical
  page primitives. Groups own independent geometry/capacity while every sequence has one atomic
  frontier across reserve, partial-tail COW, commit, rollback, checkpoint, and restore.
- Committed and transaction block-table matrices plus `[visible_begin, frontier]` device state have
  Engine-lifetime-stable addresses for CUDA Graph consumers. Append writes use a staging table;
  rejected transactions never publish provisional pages through the committed mapping.
- The exact Gemma target plan maps 50 sliding layers to Hkv=16/D256 and 10 global layers to
  Hkv=4/D512. The local group uses a 17-slot modulo ring for the 1024-token window; the global group
  uses `ceil(max_context/64)` logical pages and an aggregate resident-token budget.
- Versioned group descriptors round-trip retention, layer count, logical limits, page order, and
  every data/scale plane. This is the descriptor primitive required by the later persistent
  continuation phase; existing disk snapshot publication is not broadened yet.
- RTX 4090/CUDA 13.1 device tests passed tail payload COW/reject, 1023→1024→1025 boundaries,
  multiple wraps, destructor rollback across a wrap, wrapped prefix payload restore, append after
  restore, cross-lane restore/isolation, stable device pointers/state, and 32K progression. Local
  residency stayed at 16–17 page groups while global residency grew to 512.
- The existing physical KV suite and Qwen runtime/state/cache regressions remain green. The Phase 4
  prefix oracle remains transient until the optimized local/global attention leaves in Phases 7–8
  consume the new cache views.

## Phase 6 record

- Generalized the production group-64 append geometry from fixed D256 assumptions to exact
  D256/H16 and D512/H4 Gemma registrations. Gemma H16 accepts BF16 or RK4V4-E8 and D512
  accepts RK4V4-E8; unrelated layouts fail closed.
- The public append route matches an independent oracle exactly for every code byte and FP16 scale
  through both scheduling paths, page crossings, nonidentity mappings, zeros, sentinels, and guards.
- K reconstruction measured relative L2 0.203940-0.206853 and cosine 0.979107-0.979717; V
  measured relative L2 0.188849-0.189368 and cosine 0.982202-0.982379.
- Deterministic D512 retrieval retained the rank-1 needle at 32K and 64K, with recall of
  0.8125 and 0.7500 and score cosine of 0.986487 and 0.986418.
- At T=1024, warm CUDA Graph append measured 518.4 GB/s for local D256/H16 and 476.7 GB/s
  for global D512/H4 on RTX 4090. Detailed commands and results are in
  [the Phase 6 E8 qualification](../benchmarks/gemma4-phase6-e8.md).
- Packed K+V payload is 272 bytes per D256 head-token and 544 bytes per D512 head-token. All 50
  local layers consume 222,822,400-236,748,800 bytes across the 16-17 resident-page range;
  combined local/global payload is 935,854,080-949,780,480 bytes at 32K and
  1,648,885,760-1,662,812,160 bytes at 64K.

## Phase 7 record

- Added the exact D256 Q32/KV16, scale-1.0 public local-attention Op over the Phase 5 stable
  17-slot circular mapping. Each page-local call publishes current K/V through RK4V4-E8, decodes
  E8 directly in the attention load path, and restricts reads to the causal 1,024-token window.
- The independent FP64 oracle covers lengths 1, 2, 127, 128, 1023, 1024, 1025, and 2048,
  fragmented physical mapping, page tails, and repeated ring wraps. Relative L2 remained
  0.002065-0.002377 with maximum absolute error at most 0.001790.
- On RTX 4090, the complete public Op measured 52.448 us at T=128 and 2,177.024 us at T=2048,
  including E8 append and inverse rotation. This is 4.68x and 19.56x faster respectively than the
  Phase 4 transient-BF16 control, with intermediate speedups increasing monotonically.
- The SM89 E8 kernel retains the tuned 16-warp/64x64 schedule and 93,696-byte dynamic shared
  arena. It compiles at 119 registers/thread with no spill. Full-window T=1 measured 70.432 us; a
  separate decode kernel remains conditional on later whole-model attribution.
- Shared causal-attention, KV append, and exact Gemma E8 codec regressions remain green. Commands,
  the complete timing table, and interpretation are in
  [the Phase 7 qualification](../benchmarks/gemma4-phase7-sliding-attention.md).

## Phase 8 record

- Added the exact public D512 Q32/KV4, scale-1.0 causal full-attention Op over the Phase 5 stable global page mapping and Phase 6 RK4V4-E8 codec. It publishes current K/V once and consumes compressed cache pages directly with no context-sized materialization.
- The shallow SIMT online-softmax route and signed-int8 QK/FP16 PV tensor-core split route use the measured `max(128, 16*T)` visible-key crossover. Page-local prefill is tiled into at most four queries and reuses bounded FP32 partial workspace.
- The independent FP64 oracle covers 128, 1K, 4K, 16K, and 128K visible rows; T=1/2/3/4/64; fragmented pages; tails; and guards. Relative L2 was 0.000422-0.001592 and maximum absolute error was at most 0.000225.
- On RTX 4090, T=1 measured 144.128 us at 32K, 503.808 us at 128K, and 968.704 us at 256K. Compressed payload throughput reached 588.9 GB/s and the deep route was 5.79x faster than the provisional SIMT split kernel at 256K.
- Public prefill beat the BF16 control by 2.61x at C=1K/T=16, 9.94x at C=4K/T=128, and 13.78x at C=8K/T=1024. Static resource inspection found 254-255 registers/thread, 10,304-19,584 static plus 65,536 dynamic shared bytes, and no stack/local spill.
- Nsight Compute counter collection is blocked by host `ERR_NVGPUCTRPERM`; the exact attempted command and the complete correctness, timing, route, resource, and limitation record are in [the Phase 8 qualification](../benchmarks/gemma4-phase8-global-attention.md).

## Phase 9 record

- Added a target-owned streaming Program route that processes page-local chunks through all 60 layers while retaining history in the Phase 5 heterogeneous cache. One transaction publishes every layer at a common frontier; absolute positions drive full and proportional RoPE.
- The real 15.806 GiB target plus complete 262,144-token RK4V4-E8 cache and bounded workspaces starts with 1,785,397,248 bytes (1.66 GiB) free. Exact KV payload is 5,971,640,320 bytes and metadata is 32,920 bytes.
- A real-model 260,000-token cold prefill completed in 793,626.625 ms with no host paging. One isolated 60-layer decode at that frontier took 32.562 ms and committed frontier 260,001.
- Deterministic E8 retrieval passed at 240K and 260K: both needles ranked first; 260K recall@32 was 0.71875 and score cosine was 0.986491.
- The full 31B model retrieved all five requested semantic values exactly from token positions 12K, 60K, 120K, 180K, and 228K in one 240,000-token prompt. Prefill took 707,340.688 ms; peak GPU use was 22,378 MiB.
- The maintained real-artifact test preserves five-token BF16-reference greedy parity, varied three-token generation invariance, and 65-token output invariance across chunk schedules. The exact 262K layout/memory contract is covered without requiring the artifact.
- Ordinary RK4V4-E8 exceeds the preferred 512 MiB reserve by 3.4x, so Phase 10 shared-global-latent cache is not triggered by Phase 9 capacity. Commands and the complete memory/execution record are in [the Phase 9 qualification](../benchmarks/gemma4-phase9-262k.md).

## Phase 11 record

- Captured the complete target T=1 decode with explicit device input-token and absolute-position
  parameters. RoPE no longer captures the host frontier, and the graph never interprets committed
  cache state as a staging target.
- Local and global attention retain stable staging-table addresses while each transaction updates
  their contents. Reservation is outside the graph; commit or rollback remains one atomic
  heterogeneous-cache operation after replay.
- Real-artifact eager/graph output matches across ordinary decode, a page crossing, the local-ring
  wrap at position 1,088, and the maintained 4K semantic prompt. Device tests additionally cover
  rollback, restored-prefix replay, and an 8,193-token global frontier with one executable.
- On the 4K semantic prompt, 63 graph replays took 1,789.035 ms versus 1,828.036 ms eager: 28.397
  versus 29.016 ms per replay, a 2.13% whole-model latency reduction. Generated IDs matched
  exactly and capture count remained one.
- The complete 262K target-only profile retains 1,774,911,488 bytes free after graph upload. The
  isolated graph device allocation is 8,388,608 bytes, the largest existing temporary remains
  33,554,432 bytes, and capture adds no model workspace residency. Commands and detailed memory
  accounting are in [the Phase 11 qualification](../benchmarks/gemma4-phase11-cuda-graphs.md).

## Phase 12 record

- The optional 482,629,120-byte official assistant is materialized only for MTP. Its four Q-only
  layers consume target layer 58 sliding K/V and target layer 59 full K/V without owning a cache.
- A two-token target verifier uses one heterogeneous transaction. Accepted proposals commit both
  tokens; rejected proposals commit the one-token prefix atomically across both cache groups.
- Explicit assistant W8 and target T=2 Q4 registrations passed independent decoded-weight oracles.
  Shared sliding/global attention passed independent packed-cache oracles through key 131,071.
- The pinned first draft is token `236743`, matching Transformers. A 64-token real-model check
  matched ordinary greedy output exactly with 24 accepts and 15 rejects; eager and graphed MTP1
  also matched each other.
- A separate 8,192-token prefill produced the same sixteen greedy IDs with ordinary and MTP1
  execution. All 15 deep-context proposals were rejected without changing the cache frontier.
- On the Phase 11 4K corpus, graphed MTP1 took 1,717.122 ms for 64 tokens versus 1,828.164 ms for
  eager target decode and 1,789.035 ms for the target graph. The 49 proposals accepted 15 tokens.
- The complete 262K target, assistant, cache, workspaces, and graph retain 1,294,663,680 bytes free.
  Commands, exact residency, and qualification details are in
  [the Phase 12 record](../benchmarks/gemma4-phase12-mtp1.md).

## Open blockers and limitations

- The generated artifact remains local under `/dev/shm`; publication is a later product phase.
- The correctness route is not yet published through `ninfer::Engine`; product integration remains
  Phase 15, after draft-width selection and persistent continuation qualification.
- Full-checkpoint fixture regeneration requires restaging the two immutable source checkpoints;
  they are intentionally not committed to this repository.
- The repository-documented Python 3.11 interpreter path is absent on this machine. The isolated
  reference environment is reproducible through `uv` and the pinned requirements under
  `tools/reference/gemma4_31b_it/`.

## Reproduction

Run the commands in [the Phase 0 baseline record](../benchmarks/gemma4-baseline-environment.md).
After staging the pinned sources, reproduce Phase 1 with:

```bash
/tmp/ninfer-gemma4-reference/bin/python tools/reference/gemma4_31b_it/generate.py --source-dir /dev/shm/ninfer-gemma4-sources --output-dir tests/fixtures/gemma4_31b_it
/tmp/ninfer-gemma4-reference/bin/python tools/reference/gemma4_31b_it/generate_checkpoint.py --source-dir /dev/shm/ninfer-gemma4-sources --output-dir tests/fixtures/gemma4_31b_it --row-chunk 2048
NINFER_GEMMA4_SOURCE_DIR=/dev/shm/ninfer-gemma4-sources /tmp/ninfer-gemma4-reference/bin/python -m pytest -q tests/reference
```
