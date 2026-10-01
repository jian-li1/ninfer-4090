# Gemma 4 31B MTP width qualification

Phase 13 selects one draft as the production speculative width for
`google/gemma-4-31B-it-assistant` revision
`627c5ec1458b9086b841a91e0512fd31fd2fbbf1`. Measurements used the target-plus-assistant
artifact, an RTX 4090, CUDA 13.1, and the `sm_89` release build. Vision was not loaded.

## Correctness boundary

All proposals in a round use the same target-cache position. Draft zero consumes the target
hidden state; subsequent drafts consume the preceding assistant postprojection. The assistant
never appends K/V. The target verifies `[current, draft...]` in one heterogeneous transaction and
atomically commits the accepted prefix plus the first target correction.

Verifier Q4 supports T=2 through T=7 while preserving each T=1 column's FMA and reduction order.
Full and sliding attention likewise preserve ordinary T=1 arithmetic for every verifier column.
The latter requirement was material: a 64-token gate exposed a later-column sliding-attention
mismatch that the original five-token smoke missed. The maintained test now compares a seven-row
verifier call bit-for-bit with seven ordinary T=1 calls after populating the cache. Real-artifact
MTP1 through MTP6 runs all produce exactly the ordinary 64-token greedy sequence and frontier.

The pinned Transformers fixture independently executes six fixed-position feedback steps and
records the assistant draft IDs, feedback states, and top-32 proposal logits. The source BF16
sequence is `236743, 236770, 236888, 236888, 236888, 236888`.

## Width sweep

The code, prose, and mixed inputs are the pinned Gemma chat template applied to these prompts:

- code: implement and explain a production C++20 lock-free bounded MPMC queue;
- prose: explain changes in maritime navigation from the fifteenth through nineteenth centuries;
- mixed: review a JSON-to-batched-GPU streaming service, then provide corrected C++ pseudocode
  and an operational checklist.

They contain 53, 49, and 56 prompt tokens respectively. Each run generated 64 tokens with
`--max-context 8192 --chunk 64 --cuda-graph`. The table reports one selection run. Widths 2-6
were captured during selection; after selection they remain supported by the eager verifier while
only production width 1 owns a maintained graph.

| Corpus | Route | Decode (ms) | Rounds | Accepted by draft position | Head (ms) | Round p50/p95 (ms) | Graph |
|---|---|---:|---:|---|---:|---:|---:|
| code | target | 1,557.971 | 63 | - | - | - | 8 MiB |
| code | MTP1 | 1,328.808 | 43 | 21 | 22.766 | 30.922 / 31.084 | 8 MiB |
| code | MTP2 | 1,329.641 | 37 | 20 / 7 | 38.695 | 36.068 / 36.352 | 8 MiB |
| code | MTP3 | 1,698.780 | 37 | 18 / 6 / 3 | - | - | 10 MiB |
| code | MTP4 | 2,000.046 | 36 | total 28 | - | - | 10 MiB |
| code | MTP5 | 2,271.547 | 35 | total 29 | - | - | 12 MiB |
| code | MTP6 | 2,580.048 | 35 | total 29 | - | - | 12 MiB |
| prose | target | 1,556.045 | 63 | - | - | - | 8 MiB |
| prose | MTP1 | 1,233.172 | 40 | 24 | 21.143 | 30.873 / 31.019 | 8 MiB |
| prose | MTP2 | 1,291.294 | 36 | total 28 | - | - | 8 MiB |
| prose | MTP3 | 1,519.128 | 33 | total 31 | - | - | 10 MiB |
| prose | MTP4 | 1,782.541 | 32 | total 32 | - | - | 10 MiB |
| prose | MTP5 | 2,091.599 | 32 | total 32 | - | - | 12 MiB |
| prose | MTP6 | 2,380.955 | 32 | total 32 | - | - | 12 MiB |
| mixed | target | 1,557.763 | 63 | - | - | - | 8 MiB |
| mixed | MTP1 | 1,266.062 | 41 | 23 | 21.687 | 30.900 / 31.047 | 8 MiB |
| mixed | MTP2 | 1,297.234 | 37 | total 27 | - | - | 8 MiB |
| mixed | MTP3 | 1,679.093 | 37 | total 27 | - | - | 10 MiB |
| mixed | MTP4 | 1,979.960 | 36 | total 28 | - | - | 10 MiB |
| mixed | MTP5 | 2,318.805 | 36 | total 28 | - | - | 12 MiB |
| mixed | MTP6 | 2,569.627 | 35 | total 29 | - | - | 12 MiB |

MTP1 improves code, prose, and mixed decode by 14.71%, 20.75%, and 18.72%. Its mean is
1,276.014 ms against 1,557.260 ms for the target graph, an 18.06% improvement. MTP2 is second at
1,306.056 ms. The width-1 proposal head costs about 0.52 ms per proposal and 40-44% of assistant
time, but wider feedback steps do not recover enough target work to offset their added proposal
and T>2 verification cost.

Two additional mixed repeats measured target/MTP1 pairs of 1,554.608/1,228.953 ms and
1,554.391/1,231.527 ms. Together with the selection run, the benefit is 18.72-20.95%, with the
same 23 accepted proposals and exact output on every MTP1 repeat.

## Deep context and residency

The corrected exact verifier changes the earlier Phase 12 4K result. On that 4K semantic input,
MTP1 takes 1,874.372 ms for 64 tokens versus 1,789.035 ms for the target graph: 4.77% slower.
Widths 2-6 take 2,021.854, 2,628.215, 3,213.920, 3,787.874, and 4,333.255 ms. All produce the
same target IDs. At an 8,192-token repeated-token context, target and MTP1 produce the same sixteen
token-100 outputs; the target takes 422.058 ms and MTP1 564.624 ms because all 15 proposals are
rejected. This establishes that MTP is workload-sensitive and should remain opt-in.

At maximum context 262,144, selected MTP1 plus its graph materializes 16,971,062,784 target-weight
bytes, 482,629,120 assistant-weight bytes, 5,971,640,320 cache bytes, 32,920 cache-metadata bytes,
37,765,120 bounded-workspace bytes, 21,504 MTP-state bytes, and an 8 MiB graph allocation. It
retains 1,294,663,680 bytes free and completes a proposal/rejection round.

## Decision

Production Gemma MTP uses one draft. It is the fastest width on every short semantic corpus,
retains a reproducible mixed-workload benefit, has the smallest graph, fits the complete 262K
profile, and preserves ordinary output. Widths 2-6 remain eager-only diagnostic capabilities
implemented by the same loop and verifier; they add no production graph variants.

Reproduce a selected-width run with:

```text
build/bench/ninfer_gemma4_31b_it_long_context ARTIFACT \
  --max-context 8192 --tokens-file TOKENS --chunk 64 --generate 64 --mtp 1 --cuda-graph
```
