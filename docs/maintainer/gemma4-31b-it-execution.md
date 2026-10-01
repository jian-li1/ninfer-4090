# Gemma 4 31B IT text execution

The Phase 4 correctness route is a target-owned C++/CUDA implementation over the registered
`gemma4-31b-it/groupwise-int` `.ninfer` artifact. It accepts 1–4096 token IDs and executes the
complete 60-layer text model. It does not expose a Python inference route and does not acquire or
materialize multimodal inputs.

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

## Short-context state and persistent-cache compatibility

The Phase 4 correctness route still materializes each layer's current prefix K/V in transient BF16
storage and reruns the bounded prefix. Phase 5 added the persistent representation used by the
optimized execution route: a 50-layer D256/H16 modulo-17 sliding group and a 10-layer D512/H4
full-history group. Both use the target-neutral grouped transaction layer over the same physical
page primitives as the inherited Qwen persistent cache. The mathematical prefix route remains the
oracle until the Phase 7/8 attention leaves consume these views.

Group transactions expose stable staging block tables, copy a partial tail before mutation, and
atomically advance both groups. Reject leaves committed mappings untouched. Checkpoints clone both
group payloads and can restore a wrapped local position into either lane before appending. At 32K,
the local group remains 16–17 resident page groups while the global group reaches 512. The current
BF16 one-lane/2048-token-transaction plan reserves 5,315,870,720 payload bytes; Phase 6 replaces
those planes with the selected E8 profile before deep-context production execution.

At 4096 tokens the conservative transient workspace estimate is below 1.5 GiB. Together with the
15.806 GiB resident target weights, the Phase 4 route remains within a 24 GiB RTX 4090 planning
envelope. This is a correctness capacity statement, not a performance result.

## Qualification and first divergence

Build `ninfer_gemma4_31b_it_reference`, then run the pinned six-token checkpoint prompt:

```text
build/tests/ninfer_gemma4_31b_it_reference ARTIFACT DUMP_DIR \
  9259 236764 147224 236743 236812 236888
```

Compare it with the source-gated fixture:

```text
uv run --python 3.11 --with numpy python \
  tools/reference/gemma4_31b_it/compare_ninfer_dump.py \
  --fixture tests/fixtures/gemma4_31b_it/checkpoint.npz --dump DUMP_DIR
```

The report records layer/type/substage, reference and NInfer checksums, max and mean absolute
error, relative L2, cosine similarity, representative slices, the first failing substage, and
greedy-token agreement. The production Q4/FP8 artifact gate is relative L2 at most 0.35, cosine at
least 0.95 for every exported stage, and an identical greedy token. These thresholds cover
accumulated requantization drift; standalone semantic fixtures continue to use their narrower Op
tolerances.
