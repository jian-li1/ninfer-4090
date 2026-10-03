# Gemma 4 MTP execution

This document is the implementation authority for the official Gemma 4 31B assistant in NInfer.
Model dimensions and mathematical source semantics remain defined by
[Gemma 4 31B IT text model](gemma4-31b-it-model.md); generic speculative publication remains a
runtime Engine contract.

## Startup contract

MTP is optional and fixed when the Engine starts:

```text
backend = mtp
draft_tokens = 1..6
production selection = 1
```

An artifact containing only the target is valid for ordinary generation. Selecting MTP requires
all 44 assistant objects and fails closed if any object, shape, format, or binding is absent. The
assistant adds 482,629,120 resident weight bytes. It has its own tied 1,024-wide embedding/output
head; the generic `--lm-head-draft` selector is not required for Gemma.

## Assistant schedule

The assistant has width 1,024, MLP width 8,192, and four layers: three sliding followed by one
full-attention layer. It owns Q and output projections but no K/V projections or cache.

The target publishes two immutable views for one proposal round:

- target layer 58 supplies the last complete sliding K/V state to assistant layers 0–2;
- target layer 59 supplies the last complete global K/V state to assistant layer 3.

For each draft, the assistant combines the target token embedding with the target hidden state,
preprojects 10,752→1,024, runs its four layers against those shared views, postprojects
1,024→5,376, and applies its tied head. The assistant position remains `input_length - 1` for every
proposal in the round. It never appends to either target KV group.

## Verification and publication

After proposing K tokens, the target evaluates `[current_token, draft_0, ..., draft_K-1]` in one
T=K+1 heterogeneous transaction. The verifier compares target and draft tokens in order and
publishes the accepted prefix followed by the target correction. A rejected tail is never committed.

The T=2..7 Q4, sliding-attention, and full-attention paths are qualified against repeated ordinary
T=1 execution. That exact-column gate is stronger than final-text agreement: each verifier column,
greedy token, accepted count, and cache frontier must match. It prevents a batched verifier from
changing the target distribution merely because speculation is enabled.

MTP1 owns one captured CUDA Graph. Reservation and commit stay outside capture; stable transaction
tables and device position inputs are graph parameters. Widths 2–6 use the same eager verifier and
do not multiply graph residency. MTP state stores the target hidden tail only; the assistant owns
no persistent KV allocation.

## Selection and performance

Widths 1–6 are functional, but acceptance is workload-dependent. The maintained production
selection is one draft because it gives the best latency/acceptance balance and the smallest graph
surface.

On the representative mixed prompt, no-spec generation produces 64 tokens at 40.5 tok/s. MTP1
produces identical token IDs at 52.4 tok/s with 25 accepted of 39 drafts (64.1% acceptance), a
29.4% end-to-end speedup. On an intentionally unfavorable generic corpus, all widths can reject
every draft and become slower. Product metrics therefore publish rounds, drafted tokens, accepted
tokens, per-position acceptance, and fallback steps; a speculative tok/s number without acceptance
is not a quality or performance claim.

The complete 262,144-token MTP1 profile reserves:

| Component | Bytes |
|---|---:|
| Target + assistant weights | 17,453,691,904 |
| Sequence/KV capacity | 6,180,570,112 |
| Unified workspace | 312,554,496 |
| Available after startup | 868,941,824 |

Workspace planning takes the maximum of mutually exclusive 1,024-token prefill and seven-column
verifier shapes. It does not allocate 1,024 columns of logits merely because MTP is enabled.

## Public use

CLI:

```bash
./build/apps/ninfer models/gemma4_31b_it.ninfer \
  --prompt "Explain speculative verification." \
  --max-context 8192 --max-new 256 \
  --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 1
```

Server:

```bash
./build/apps/ninfer-serve models/gemma4_31b_it.ninfer \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 1
```

Greedy MTP output must equal no-spec output. Under sampling, each accepted token still follows the
target verifier and the same Engine sampler contract; assistant probabilities are proposals only.
Continuation identity includes the speculative configuration, so a target-only snapshot cannot be
restored into an MTP Engine or vice versa.

## Qualification

The maintained evidence includes MTP1 deterministic parity, sampled fixed-seed parity, widths
1–6 later-column feedback, accepted and rejected transactions, graph/eager equality, full-context
admission, public Engine/CLI/server routes, and continuation rejection across incompatible startup
profiles. See the [MTP1](../benchmarks/gemma4-phase12-mtp1.md),
[width selection](../benchmarks/gemma4-phase13-mtp-widths.md), and
[production performance](../benchmarks/gemma4-31b-4090-performance.md) records.
