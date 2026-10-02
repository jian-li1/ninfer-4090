# Gemma 4 Phase 14 continuation qualification

Phase 14 qualifies heterogeneous prefix persistence for `gemma4-31b-it/groupwise-int` on one
NVIDIA GeForce RTX 4090 (`sm_89`, CUDA 13.1). The real checkpoint artifact contains
16,971,062,784 device-weight bytes. Vision was not loaded.

## State and compatibility contract

The version-1 `NIG4CNT1` envelope records an exact token ledger, execution frontier, target model
binding, immutable registered-checkpoint fingerprint, encoded heterogeneous-KV descriptors, byte
count, and checksum. The endpoint stores both the 50-layer sliding group and 10-layer global
group. Each older edit anchor stores only its sliding group; its global pages are reconstructed as
an exact prefix of the endpoint, avoiding one global-history copy per anchor.

Decode validates all framing and compatibility metadata before device allocation. Maintained
negative tests reject a different model binding, artifact fingerprint, KV plane geometry, version,
checksum, byte count, and truncated payload. Raw cache-import tests also reject stale sliding
logical blocks and a truncated global payload before publication.

The selected checkpoint frontier is imported atomically across both groups. The incoming ledger
must exactly match through that frontier. Exact replay selects the endpoint; an edit selects the
deepest earlier compatible anchor; an edit before every anchor starts cold. Sliding pages retain
absolute logical-block identities, so an anchor after the 1,024-token ring wrap restores the
correct local interval rather than stale modulo slots.

## Real-model correctness

The maintained artifact test uses a varied 1,153-token prompt, maximum context 2,048, 64-token
chunks, a CUDA Graph decode, and one anchor at frontier 1,089. It saves the endpoint at frontier
1,152, leaving the final prompt token pending.

- Exact restore reuses 1,152 tokens, computes one prompt token, and produces the same four greedy
  output IDs and final frontier as the original execution.
- Changing token 1,100 invalidates the endpoint but preserves the 1,089-token anchor. Restore
  computes exactly 64 prompt tokens and produces the same four IDs and final frontier as a cold
  edited-prompt run.
- The anchor is beyond the local-window wrap, directly covering the stale-ring regression.

The maintained test completed in 58.27 seconds and all focused framing/raw-transfer tests passed.

## Restart, footprint, and latency

Two separate benchmark processes used a repeated-token 1,153-token prompt with the same context,
chunk, anchor, graph, and four-token generation settings. The first wrote the snapshot to disk;
the second read and restored it. Both generated `100,100,100,100` and ended at frontier 1,156.

| Measurement | Save process | Restore process |
|---|---:|---:|
| Cold/computed prompt tokens | 1,153 | 1 |
| Reused prompt tokens | 0 | 1,152 |
| Snapshot bytes | 484,647,692 | - |
| KV payload bytes in snapshot | 484,638,720 | - |
| Anchors | 1 | - |
| Snapshot capture + encode | 828.019 ms | - |
| Validate + decode + import | - | 499.060 ms |
| Model prefill interval | 2,534.297 ms | 517.287 ms |

The file is 462.20 MiB. Its single global payload serves both endpoint and anchor; the remaining
payload is the two required sliding-window images. Restore plus the remaining model prefill takes
1,016.347 ms versus 2,534.297 ms for cold model prefill in the separate save process. This is a
single-run restart qualification, not a production throughput claim.

## Reproduction

```bash
cmake --build build -j 12 --target \
  ninfer_heterogeneous_kv_cache_test ninfer_gemma4_continuation_test \
  ninfer_gemma4_31b_it_persistent ninfer_gemma4_31b_it_long_context
ctest --test-dir build --output-on-failure \
  -R '^(ninfer_heterogeneous_kv_cache_test|ninfer_gemma4_continuation_test)$'
NINFER_GEMMA4_ARTIFACT=/dev/shm/gemma4_31b_it.ninfer \
  ctest --test-dir build --output-on-failure -R '^ninfer_gemma4_31b_it_persistent$'
build/bench/ninfer_gemma4_31b_it_long_context /dev/shm/gemma4_31b_it.ninfer \
  --max-context 2048 --prefill 1153 --chunk 64 --token 2 --generate 4 --cuda-graph \
  --anchor 1089 --save-continuation /tmp/ninfer_gemma4_phase14.continuation
build/bench/ninfer_gemma4_31b_it_long_context /dev/shm/gemma4_31b_it.ninfer \
  --max-context 2048 --prefill 1153 --chunk 64 --token 2 --generate 4 --cuda-graph \
  --restore-continuation /tmp/ninfer_gemma4_phase14.continuation
```
