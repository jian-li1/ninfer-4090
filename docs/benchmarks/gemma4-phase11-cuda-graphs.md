# Gemma 4 Phase 11 CUDA Graph qualification

Phase 11 captures the complete single-token target decode for
`gemma4-31b-it/groupwise-int` on one NVIDIA GeForce RTX 4090 (`sm_89`, CUDA 13.1,
driver 580.178.04). Prefill remains eager and page-local. Decode uses one graph executable for
all absolute positions, including local-ring reuse and global-context growth.

## Execution and transaction design

The graph reads the input token and absolute logical position from fixed-address device parameter
buffers. RoPE therefore does not capture a host position value. Both local and global attention
read the transaction-owned staging block tables at their Engine-lifetime-stable addresses; each
transaction refreshes the table contents before replay. The graph does not read the committed
`[visible_begin, frontier]` state as a transaction target.

Only model kernels are captured. Page reservation and table preparation happen before replay;
atomic transaction commit or rollback happens after replay. Rejected work is synchronized before
its provisional leases are released. The capture uses the maximum full-attention envelope, so
ordinary sequence growth, 64-token page crossings, and the 1,024-token local-ring wrap do not
change graph topology or trigger recapture. Activation and attention scratch retain their existing
caller-owned allocations.

## Correctness and state qualification

- The real artifact produced identical 64-token greedy output on the maintained 4,096-token
  five-needle semantic prompt with graph execution enabled and disabled.
- The maintained real-artifact test compares eager and graph generation for a varied prompt, a
  64-token page crossing, and a decode at absolute position 1,088 that reuses local ring slot zero.
  Every generated token and final frontier matches; the wrap run reports one capture and two
  replays.
- A rejected real-model graph replay produces the same greedy token when repeated and committed,
  while the rejected replay leaves the common heterogeneous frontier unchanged.
- The device cache test captures both staging table descriptors and an explicit logical-position
  buffer. One executable is replayed through normal commit, rollback at local-ring wrap, append
  after a restored wrapped prefix, and an 8,193-token global frontier. It also verifies that the
  transaction view continues to expose the old committed state rather than silently publishing a
  staging frontier.

## End-to-end decode measurement

The before/after runs use the same binary, real artifact, 4,096-token semantic input, 64 generated
tokens, maximum context 4,160, 64-token prefill chunks, and greedy decoding. Decode timing covers
the 63 model executions after the first token returned by prefill.

| Route | Captures / replays | Decode total (ms) | Decode per replay (ms) | Generated IDs |
|---|---:|---:|---:|---|
| Eager | 0 / 0 | 1,828.036 | 29.016 | reference |
| CUDA Graph | 1 / 63 | 1,789.035 | 28.397 | exact match |

CUDA Graph replay reduces end-to-end decode latency by 2.13% for this workload. This is a
whole-model result rather than an Op microbenchmark. No recapture occurred while the frontier grew.

## 262K memory admission

The capacity comparison uses the complete 262,144-token target-only profile, the real 15.806 GiB
target, RK4V4-E8 local/global cache, one-token prefill, and one single-token decode. Both routes use
the same maximum-context attention envelope.

| Resource/checkpoint | Eager bytes | Graph bytes |
|---|---:|---:|
| Target weights | 16,971,062,784 | 16,971,062,784 |
| KV payload | 5,971,640,320 | 5,971,640,320 |
| KV metadata | 32,920 | 32,920 |
| Existing activation + attention workspace | 41,975,808 | 41,975,808 |
| Largest temporary allocation | 33,554,432 | 33,554,432 |
| Measured graph device allocation after warmup | 0 | 8,388,608 |
| Free after workspace allocation | 1,785,397,248 | 1,785,397,248 |
| Free after capture/upload | n/a | 1,774,911,488 |

Capture introduces no second persistent model workspace. The 8 MiB figure isolates graph upload
after the prefill warmup; the 10 MiB difference from the pre-execution free-memory checkpoint also
contains 2 MiB of first-execution CUDA runtime residency. The graph profile retains 1.653 GiB free
and therefore preserves the 262K fit gate.

## Reproduction

```bash
cmake --build build --target ninfer_gemma4_31b_it_persistent ninfer_gemma4_31b_it_long_context ninfer_heterogeneous_kv_cache_test -j 12
./build/tests/ninfer_heterogeneous_kv_cache_test
NINFER_GEMMA4_ARTIFACT=/dev/shm/gemma4_31b_it.ninfer ./build/tests/ninfer_gemma4_31b_it_persistent
./build/bench/ninfer_gemma4_31b_it_long_context /dev/shm/gemma4_31b_it.ninfer --max-context 4160 --tokens-file /tmp/gemma4_niah_e8_4k_tokens.txt --chunk 64 --generate 64
./build/bench/ninfer_gemma4_31b_it_long_context /dev/shm/gemma4_31b_it.ninfer --max-context 4160 --tokens-file /tmp/gemma4_niah_e8_4k_tokens.txt --chunk 64 --generate 64 --cuda-graph
./build/bench/ninfer_gemma4_31b_it_long_context /dev/shm/gemma4_31b_it.ninfer --max-context 262144 --prefill 1 --chunk 64 --token 2 --generate 2
./build/bench/ninfer_gemma4_31b_it_long_context /dev/shm/gemma4_31b_it.ninfer --max-context 262144 --prefill 1 --chunk 64 --token 2 --generate 2 --cuda-graph
```
