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
| 3 | pending | `.ninfer` converter and exact binder |
| 4 | pending | Reference-correct short-context execution |
| 5 | pending | Heterogeneous physical KV groups |
| 6 | pending | E8 codecs for D256 and D512 |
| 7 | pending | Optimized D256 sliding attention |
| 8 | pending | Optimized D512 global attention |
| 9 | pending | 262,144-token target-only execution |
| 10 | conditional | Shared global K/V latent cache; implement only if Phase 9/MTP/graph memory evidence triggers it |
| 11 | pending | CUDA Graph decode |
| 12 | pending | Official Gemma MTP1 |
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

## Open blockers and limitations

- The generated artifact remains local under `/dev/shm`; publication is a later product phase.
- Phase 4 still needs executable Gemma Ops and short-context numerical parity before the artifact
  can serve inference.
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
