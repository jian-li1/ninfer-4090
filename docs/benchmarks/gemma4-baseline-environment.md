# Gemma 4 31B implementation baseline

This record freezes the repository, machine, and existing Qwen3.8 behavior before the
Gemma 4 31B work begins. It is the Phase 0 baseline for
`gemma4-31b-it/groupwise-int`; it is not a Gemma performance claim.

## Repository baseline

| Field | Value |
|---|---|
| Parent branch | `feat/persist-kv-cache` |
| Feature branch | `feat/gemma-4-31b` |
| Baseline commit | `3a7a0d4ecc250c92dcf22c6f3d074578be8ca2d6` |
| Build directory | `build-sm89` |
| Build type | `Release` |
| CUDA architecture | `89` |
| Artifact container | version 2 |

The feature branch was created directly from the named parent with a clean worktree. The
implementation specification is `/home/jianli/GEMMA4_31B_NINFER_4090_IMPLEMENTATION_SPEC.md`,
dated 2026-09-30. Its optional multimodal/vision extension is excluded from this project.

## Machine and toolchain

Captured on 2026-09-30 in `America/Los_Angeles`:

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9 |
| GPU memory | 24,564 MiB total; 24,021 MiB free while idle |
| Driver | 580.178.04 |
| Driver-reported CUDA compatibility | 13.0 |
| CUDA compiler | 12.9.86 |
| Host compiler | GCC 13.3.0 |
| CMake | 3.28.3 |
| GPU power limit | 480 W current/requested, 530 W maximum |
| Maximum SM clock | 3165 MHz |
| Idle temperature during capture | 38 C |

The repository's documented Python interpreter
`/home/neroued/miniconda3/envs/py311/bin/python` was absent. Baseline Python tests therefore used
an isolated `uv` environment at `/tmp/ninfer-gemma-py311` with CPython 3.11.13, pytest 9.1.1,
NumPy 2.4.6, safetensors 0.8.0, and CPU PyTorch 2.14.1. The system Python is 3.12.3 and did not
contain the test dependencies.

## Baseline verification

### Build

```bash
cmake -S . -B build-sm89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DBUILD_TESTING=ON \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-sm89 -j
```

Result: configuration and all build targets completed successfully.

### C++ and CUDA tests

```bash
ctest --test-dir build-sm89 --output-on-failure -j1
```

Result: 121/121 tests passed in 281.33 seconds. Of these, 109 ran and 12 were expected skips:
seven real-artifact target tests without their opt-in environment variables, four unsupported
SM89 NVFP4/K8V4 routes, and the SM89-inapplicable NVFP4-A4 Linear test. The E8 codec, paged KV,
sliding attention, softmax attention, speculative transaction, public API, serving-schema, and
persistent prompt-cache tests all passed.

### Python artifact and converter tests

```bash
/tmp/ninfer-gemma-py311/bin/python -m pytest \
  tests/artifact tests/convert \
  tests/test_bench_matrix.py tests/test_serve_corpus.py
```

Result: 76 passed and 3 source-checkpoint-dependent tests skipped in 2.39 seconds.

## Qwen3.8 regression smoke

The smoke used the explicit local artifact `models/qwen3_8_27b.ninfer`:

| Field | Value |
|---|---|
| Size | 18,210,531,328 bytes |
| SHA-256 | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| Runtime identity | `qwen3.8-27b/groupwise-int` |

```bash
./build-sm89/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode in two sentences." \
  --max-context 8192 --max-new 32 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --greedy --no-thinking
```

The public Engine route loaded and generated successfully. The fixed short smoke is for regression
comparison rather than stable performance publication:

| Metric | Result |
|---|---:|
| Weight materialization | 16.7 GiB in 11.1 s |
| Engine startup | 12.2 s |
| Prompt / generated tokens | 22 / 32 |
| Text prefill | 76.7 ms, 286.9 tok/s |
| Decode | 246 ms, 126.1 tok/s |
| Overall model throughput | 99.2 tok/s |
| MTP acceptance | 20/32, 62.5% |
| GPU sequence reservation | 448.2 MiB |
| Runtime reservation | 686.8 MiB |
| Free after startup | 5.80 GiB |

Output began: “Prefill is the initial phase where the model processes the entire input prompt in
parallel … Decode is the subsequent …” and stopped at the requested output-token limit.

This smoke is the Phase 0 Qwen functional/performance reference. Later generic cache, Engine, or Op
changes must at minimum preserve successful public-Engine generation and the relevant regression
suites; performance comparisons require the same artifact, command, and machine conditions.
