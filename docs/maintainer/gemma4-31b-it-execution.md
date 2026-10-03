# Gemma 4 31B IT text execution

This document defines the production C++/CUDA schedule for the registered text-only
`gemma4-31b-it/groupwise-int` `.ninfer` target. CLI, serving, causal scoring, benchmarks, and
continuations all enter the public `ninfer::Engine`. There is no Python or target-private product
inference route, and the target does not acquire or materialize multimodal inputs.

## Mathematical schedule

The route applies the BF16 embedding scale `73.5`, then the exact 50 sliding / 10 full layer
schedule. Every layer uses direct learned RMS gains and epsilon `1e-6`. Sliding attention uses
32 query heads, 16 KV heads, D256, a 1024-token causal window, and full D256 RoPE at theta 10,000.
Full attention uses 32 query heads, 4 KV heads, D512, a causal full prefix, and only 64 active
rotary pairs at theta 1,000,000. Attention scores are not divided by square root D.

Full layers obtain both K and V from the single K projection. K receives its learned per-head RMS
gain while V receives scale-free RMS normalization. Attention probabilities are formed in FP32,
cast to BF16 before the V product, and the dense MLP uses tanh-approximate GELU. Each layer applies
post-attention and post-feedforward normalization in the source ordering, followed by its learned
scalar. The final vocabulary projection reuses the FP8 embedding object and applies
`30 * tanh(logit / 30)` before greedy selection.

## Persistent production schedule

One Program instance owns one target binding, its startup-fixed lanes, one heterogeneous cache,
one phase-reused workspace, and its CUDA Graph instances. It shares no mutable state or device
allocation with another Program.

The model admits 1–2,048 tokens per execution chunk; production prefill uses 1,024. Q4 projections
and MLPs run across the wide chunk, while each layer publishes K/V and evaluates attention
sequentially in 64-token physical-page tiles. This preserves causal visibility, local-ring eviction,
partial-tail copy-on-write, and the common local/global frontier. Intermediate chunks omit logits
and sampling; only the final prompt chunk computes the next-token distribution.

Sliding D256/H32/KV16 attention consumes the 1,024-token ring directly. Global D512/H32/KV4 uses
the prompt kernel through 32K visible keys, then split T6 tensor-core tiles and FP32 reduction at
deeper prefixes. T=1 decode and T=2..7 verifier columns use separately qualified schedules.
Absolute positions drive both RoPE regimes and cache publication.

Startup derives activation capacity from the same allocation recipe used by execution and takes
the maximum of mutually exclusive prefill, final-logit, and MTP-verifier shapes. At 262,144 tokens,
target-only residency is 16,971,062,784 weight bytes, 6,180,570,112 sequence bytes, and
312,532,992 workspace bytes, leaving 1,353,383,936 bytes after startup on the qualified RTX 4090.
Requests perform no project-owned device allocation or workspace growth.

The group geometry, RK4V4-E8 planes, transaction invariants, and continuation encoding are defined
in [Gemma 4 heterogeneous KV layout](gemma4-kv-layout.md).

## Persistent decode graph

T=1 decode may execute eagerly or through one CUDA Graph executable. Input token, absolute
position, row, and stable transaction tables are graph-safe device parameters. QKV preparation
reads the explicit position, preserving full and proportional RoPE across page boundaries and ring
reuse.

Each cache transaction prepares stable local/global staging tables before graph replay. Attention
kernels use those tables and the explicit position; they do not infer the transaction target from
the committed state tensor. Replay completes before the caller commits or rolls back both groups.
The graph uses the maximum registered full-attention envelope, so global-prefix growth and local
ring wrap preserve one topology and require no update or recapture. Prefill and parity dumps remain
eager.

MTP1 owns a separate verifier graph; widths 2–6 use the common eager path. Exact graph/eager token
and frontier equality is maintained for ordinary and speculative execution. Historical graph
qualification is recorded in
[the Phase 11 record](../benchmarks/gemma4-phase11-cuda-graphs.md); current whole-profile memory is
in the [production performance report](../benchmarks/gemma4-31b-4090-performance.md).

## Persistent continuation

The target continuation envelope stores the exact prompt-token ledger, committed execution
frontier, immutable target-checkpoint fingerprint, version, byte count, checksum, and encoded
descriptors for both KV groups. The endpoint owns the complete sliding and global host images.
Older edit anchors own only their sliding image; their global history is a prefix of the endpoint
global image and is reconstructed without storing a second copy. Decode rejects a different model
binding, checkpoint fingerprint, KV layout, token domain, version, extent, or checksum before it
reserves device pages.

Restore first exact-matches the endpoint and then the deepest compatible anchor against the
incoming token ledger. It imports both groups at one common frontier before normal page-local
prefill resumes. Thus an unchanged prompt executes only its pending final token, while an edit
after an anchor executes only the changed suffix. In particular, the sliding image records
absolute logical blocks rather than ring slots, so restoring an anchor beyond position 1,024
cannot reuse stale local KV. Core owns only logical-order host page transfer; this framing,
identity, anchor selection, and global-prefix reconstruction remain Gemma-family policy.

The long-context diagnostic exposes explicit `--save-continuation PATH`,
`--restore-continuation PATH`, and repeatable `--anchor N` controls for cross-process
qualification. The public Engine, server slot files, and transparent persistent cache consume the
same target semantics. Exact endpoint, edited-anchor, restart, validation, latency, and footprint
evidence is recorded in
[the Phase 14 qualification](../benchmarks/gemma4-phase14-continuation.md).

## MTP execution

MTP1 optionally materializes the official four-layer assistant. Assistant layers 0-2 read the
fixed-position sliding K/V published by target layer 58; layer 3 reads target layer 59's full K/V.
The assistant has no cache and its attention calls never append to target state.

One through six fixed-position proposals are supported. Draft zero consumes the target hidden
state and later drafts consume the prior assistant postprojection. A target T=2..7 execution
verifies `[current, draft...]` in one heterogeneous transaction, which commits the accepted prefix
plus the target correction. Q4, full attention, and sliding attention preserve ordinary T=1
arithmetic for every verifier column, so batching cannot alter greedy output.

Production selects one draft and captures only that verifier as a CUDA Graph; reservation and
commit remain outside capture. Widths 2–6 use the shared eager verifier and add no graph variants.
The complete 262K MTP1 profile holds 17,453,691,904 weight bytes and leaves 868,941,824 bytes after
startup. Short and long real-model checks match ordinary greedy output exactly on accepted and
rejected paths. Proposal ownership, selection evidence, and public commands are defined in
[Gemma 4 MTP execution](gemma4-mtp.md).

## Qualification and first divergence

Build `ninfer_gemma4_31b_it_reference`, then run the pinned six-token checkpoint prompt:

```text
build/tests/ninfer_gemma4_31b_it_reference ARTIFACT DUMP_DIR \
  9259 236764 147224 236743 236812 236888
```

Compare it with the source-gated fixture:

```text
/tmp/ninfer-gemma4-reference/bin/python \
  tools/reference/gemma4_31b_it/compare_ninfer_dump.py \
  --fixture tests/fixtures/gemma4_31b_it/checkpoint.npz --dump DUMP_DIR
```

The report records layer/type/substage, reference and NInfer checksums, max and mean absolute
error, relative L2, cosine similarity, representative slices, the first failing substage, and
greedy-token agreement. The production Q4/FP8 artifact gate is relative L2 at most 0.35, cosine at
least 0.95 for every exported stage, and an identical greedy token. These thresholds cover
accumulated requantization drift; standalone semantic fixtures continue to use their narrower Op
tolerances.
