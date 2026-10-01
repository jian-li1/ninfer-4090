# Gemma 4 31B MTP1 qualification

Phase 12 qualifies the official `google/gemma-4-31B-it-assistant` at pinned revision
`627c5ec1458b9086b841a91e0512fd31fd2fbbf1` with the target-plus-assistant artifact documented in
the Gemma implementation status. Measurements used an RTX 4090, CUDA 13.1, and the `sm_89`
release build.

## Delivered route

The optional 482,629,120-byte assistant is materialized only when MTP is requested. Its four
Q-only layers consume fixed-position target K/V: assistant layers 0-2 read target layer 58's
sliding group and layer 3 reads target layer 59's global group. The assistant owns no K/V cache.
Its six W8 matrix shapes and tied FP8 proposal head have explicit T=1 registrations.

Each MTP1 round proposes one token, reserves a two-token target transaction, and verifies
`[current, draft]`. Acceptance commits both tokens; rejection commits only the current token with
`commit_prefix(1)`. The heterogeneous cache advances the local and global groups atomically in
both cases. Target T=2 Q4 verification stages each weight tile once but preserves the T=1 FMA and
reduction order independently for each column. Full attention evaluates the two queries through
the same one-query arithmetic, and shared-K/V assistant attention never appends to target state.

The target verifier is captured as one stable CUDA Graph. Reservation and prefix commit remain
outside the graph; assistant proposal is eager because its 1.43 ms measured cost did not justify a
second graph in this phase.

## Numerical and state qualification

- Six assistant W8 shapes compare against independently decoded FP64 weight oracles.
- Target T=2 Q4 shapes compare against independent FP64 oracles at every registered matrix shape.
- Shared sliding and global attention compare against independent packed-cache oracles, including
  global keys 127, 4095, and 131071.
- The pinned six-token checkpoint's first proposal is token `236743`, matching pinned Transformers.
- A 64-token real-model run produced exactly the ordinary greedy sequence while exercising 39
  proposals: 24 accepted and 15 rejected. Eager and graphed MTP1 outputs were identical.
- At a deep 8,192-token repeated-token prefill, ordinary and MTP1 both generated sixteen token
  `100` values exactly. MTP proposed 15 tokens and rejected all 15 without changing the target
  sequence or either cache group's frontier.
- Cache tests cover invalid prefix commits, same-page rejection, and cross-page suffix release.

The focused maintained suite was:

```text
cmake --build build -j 12 --target \
  ninfer_gemma4_31b_it_persistent ninfer_gemma4_31b_it_artifact_bindings_test \
  ninfer_gemma4_31b_it_config_test ninfer_heterogeneous_kv_cache_test \
  ninfer_linear_q4_a16_test ninfer_linear_w8_a16_test \
  ninfer_gemma4_sliding_attention_test ninfer_gemma4_global_attention_test \
  ninfer_gemma4_31b_it_long_context
NINFER_GEMMA4_ARTIFACT=/dev/shm/gemma4_31b_it.ninfer \
  ctest --test-dir build --output-on-failure -j 1 \
  -R '^(ninfer_heterogeneous_kv_cache_test|ninfer_gemma4_31b_it_config_test|ninfer_gemma4_31b_it_artifact_bindings_test|ninfer_gemma4_sliding_attention_test|ninfer_gemma4_global_attention_test|ninfer_linear_q4_a16_test|ninfer_linear_w8_a16_test)$'
NINFER_GEMMA4_ARTIFACT=/dev/shm/gemma4_31b_it.ninfer \
  build/tests/ninfer_gemma4_31b_it_persistent
```

All focused checks passed.

## Whole-model result

The 4K semantic corpus used in Phase 11 generated 64 tokens. All modes produced the exact same 64
token IDs.

| Route | Decode (ms) | MTP proposals | Accepted | Change vs eager target |
|---|---:|---:|---:|---:|
| Target eager | 1,828.164 | 0 | 0 | baseline |
| Target graph | 1,789.035 | 0 | 0 | 2.14% faster |
| MTP1 eager | 1,745.434 | 49 | 15 | 4.52% faster |
| MTP1 graph | 1,717.122 | 49 | 15 | 6.07% faster |

The graphed MTP1 route spent 70.334 ms in proposal and 1,646.362 ms in target verification. It was
1.62% faster than eager MTP1 and 4.02% faster than the Phase 11 target graph. This is the positive
corpus speedup required for MTP1 admission; it is not a claim about every prompt.

## 262K residency

With maximum context 262,144, MTP1, and its graph enabled, the process materialized
16,971,062,784 target-weight bytes, 482,629,120 assistant-weight bytes, 5,971,640,320 cache bytes,
32,920 cache-metadata bytes, 37,765,120 bounded workspace bytes, 10,752 MTP workspace bytes, and
an 8 MiB graph allocation. It retained 1,294,663,680 bytes free after graph creation and completed
a proposal/rejection round. MTP therefore fits together with the selected full RK4V4-E8 cache and
does not require the deferred Phase 10 latent-cache variant.
