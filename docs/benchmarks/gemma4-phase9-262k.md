# Gemma 4 Phase 9 262K target-only qualification

Phase 9 establishes native 262,144-token allocation and target-only execution without MTP on one
NVIDIA GeForce RTX 4090 (`sm_89`, CUDA 13.1, driver 580.178.04). The run uses the real
15.806 GiB target artifact, 64-token page-local prefill chunks, the heterogeneous persistent cache,
RK4V4-E8 local/global K/V, and the Phase 7/8 production attention Ops.

## Execution design

Each chunk is processed through all 60 layers before the next chunk starts. Every layer retains its
history in its own persistent cache plane, so this schedule is causally equivalent to full-prefix
execution but retains only one activation chunk. One heterogeneous transaction reserves local and
global pages for a chunk; all 60 layers write through its staging page tables; commit atomically
publishes the common frontier. Absolute positions feed Gemma's full/proportional RoPE.

The allocation order is deliberate and fragmentation-resistant: one materialized-weight arena, one
contiguous heterogeneous-KV backing allocation, one 32 MiB activation workspace, and one bounded
global-attention workspace. There is no sequence-sized hidden tensor, host KV spill, unified-memory
fallback, or context-sized decoded K/V/attention materialization.

## Memory admission

The 262,144-capacity startup and one-token real-model execution succeeded.

| Resource/checkpoint | Bytes | GiB |
|---|---:|---:|
| Materialized target weights | 16,971,062,784 | 15.8056 |
| Heterogeneous KV payload | 5,971,640,320 | 5.5615 |
| KV metadata | 32,920 | 0.0000 |
| Activation + attention workspace | 41,975,808 | 0.0391 |
| Free after weights | 7,802,126,336 | 7.2663 |
| Free after KV allocation | 1,829,437,440 | 1.7038 |
| Free after all workspaces | 1,785,397,248 | 1.6628 |

The final 1.66 GiB slack is 3.4x the preferred 512 MiB reserve. Ordinary RK4V4-E8 therefore fits;
Phase 10 shared-global-latent cache is not triggered by Phase 9 capacity. The exact layout contract
is maintained in `ninfer_heterogeneous_kv_cache_test`: local table/physical pages are 17/19,
global table/physical pages are 4096/4098, and payload/metadata bytes match the table above.

## Correctness and retrieval

- The five-token persistent real-artifact route matches the Phase 4 BF16 full-prefix reference
  greedy token exactly (`100`).
- A 65-token persistent run produces the same greedy result with 64+1 and 32+32+1 chunk schedules,
  directly covering cache publication and absolute-RoPE behavior across a page boundary.
- A varied six-token prompt generates the same three greedy tokens with 6-token and 3-token
  prefill chunks. The independent Q4 Op oracle includes Gemma's exact T=1 N=16,384/K=5,376
  projection shape, preventing the decode dispatcher from silently omitting its final four G64
  groups.
- Deterministic D512 RK4V4-E8 retrieval retained the needle at rank 1 at both 240,000 and 260,000
  rows. At 240K, recall@32 was 0.6875 and score cosine was 0.986528. At 260K, recall@32 was
  0.71875 and score cosine was 0.986491.
- The full 31B model retrieved five semantic values exactly from a 240,000-token rendered prompt.
  The facts began at token positions 12,000, 60,000, 120,000, 180,000, and 228,000. Greedy output
  began `orange-telescope violet-compass silver-lantern cobalt-orchid amber-harbor`, with no
  expected value missing.

The persistent BF16-vs-compressed greedy result is intentionally not required to remain identical
after long cache accumulation; codec quality is governed by the independent retrieval criterion.
Chunk schedules, which observe the same codec boundary, must remain invariant.

### Full-model 240K semantic gate

The maintained NIAH harness constructs one deterministic chat-template prompt with five distinct
key/value facts at 5%, 25%, 50%, 75%, and 95%, then asks for all five values in order. It uses the
staged target tokenizer, greedy generation, and the same target-only persistent route as the 262K
capacity run.

| Field | Result |
|---|---|
| Qualification source commit | `66e3fb12` |
| Artifact SHA-256 | `11d70c73f940930035516113a4b1cc25c685111692c4e9f2b345b430234ec83a` |
| Prompt / generated tokens | 240,000 / 32 |
| Cache | local/global RK4V4-E8 |
| Prefill / generation | 707,340.688 ms / 1,143.804 ms |
| Whole harness elapsed | 710.482 s |
| Peak GPU memory used | 22,378 MiB |
| Free after cache / workspaces | 1,829,437,440 / 1,785,397,248 bytes |
| Result | all five values present exactly |

The generated token IDs were:

```text
28975,236772,203504,11512,39261,236772,24938,10173,236772,71043,56896,
236772,142108,67706,236772,6234,4431,106,107,101,1,107,1,107,1,107,236783,
107,1,107,236783,107
```

The same harness also passed a 4,096-token smoke before the long run. The original semantic
gibberish was traced to the Q4 T=1 dispatcher selecting a kernel statically specialized for 80 G64
groups at Gemma's 84-group K=5,376 projection shape. The corrected dynamic-bound route and its
independent decoded-weight oracle are part of the qualification source commit history. The E8
cache itself therefore remains the production profile; the Phase 10 shared-latent experiment did
not improve the real memory/performance tradeoff and was removed.

## Cold prefill and deep decode

The complete real model cold-prefilled 260,000 repeated valid text tokens in 793,626.625 ms
(13m 13.6s), then executed one isolated token through all 60 layers at that frontier in 32.562 ms.
The final committed frontier was 260,001 and the decode greedy token was 64,463. All weights, cache
pages, and workspaces remained resident on the device for the entire run. The repeated-token input
is deliberate: it makes this a deterministic execution/resource gate while the separate randomized
retrieval test probes long-context codec discrimination.

## Reproduction

```bash
cmake --build build -j 12 --target ninfer_gemma4_31b_it_long_context ninfer_gemma4_31b_it_persistent ninfer_heterogeneous_kv_cache_test ninfer_gemma4_e8_retrieval_bench
NINFER_GEMMA4_ARTIFACT=/dev/shm/gemma4_31b_it.ninfer ctest --test-dir build --output-on-failure -R 'ninfer_(heterogeneous_kv_cache_test|gemma4_31b_it_reference|gemma4_31b_it_persistent)$'
./build/bench/ninfer_gemma4_e8_retrieval_bench
./build/bench/ninfer_gemma4_31b_it_long_context /dev/shm/gemma4_31b_it.ninfer --max-context 262144 --prefill 1 --chunk 64 --token 2 --no-decode
./build/bench/ninfer_gemma4_31b_it_long_context /dev/shm/gemma4_31b_it.ninfer --max-context 262144 --prefill 260000 --chunk 64 --token 2
/tmp/ninfer-gemma4-niah/bin/python tools/bench/gemma4_niah.py --executable build/bench/ninfer_gemma4_31b_it_long_context --artifact /dev/shm/gemma4_31b_it.ninfer --tokenizer /dev/shm/ninfer-gemma4-sources/target/tokenizer.json --prompt-tokens 240000 --generate 32 --tokens-out /tmp/gemma4_niah_e8_240k_tokens.txt --report /tmp/gemma4_niah_e8_240k.json
```
