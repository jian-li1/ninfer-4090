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
| 1 | pending | Pinned source revisions and semantic fixtures |
| 2 | pending | Compile-time Gemma family and target skeleton |
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

Phase 1 will replace these placeholders with immutable revisions and fixture hashes after inspecting
the authoritative Google artifacts and the exact Transformers implementation:

| Source | Revision |
|---|---|
| Target checkpoint | pending |
| Assistant checkpoint | pending |
| Tokenizer resources | pending |
| Transformers Gemma 4 implementation | pending |

## Open blockers and limitations

- The target and assistant checkpoints are not yet present locally; Phase 1 must pin and acquire
  only the metadata/resources needed for fixtures before any runtime semantics are frozen.
- No Gemma artifact exists yet, so Gemma memory and performance measurements begin in later phases.
- The repository-documented Python 3.11 interpreter path is absent on this machine. The isolated
  Phase 0 environment is reproducible through `uv`; Phase 1 will commit exact environment
  instructions for the reference tools.

## Reproduction

Run the commands in [the Phase 0 baseline record](../benchmarks/gemma4-baseline-environment.md).
Phase-specific commands will be added here as their gates are completed.
