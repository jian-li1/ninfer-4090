# Gemma 4 31B IT text model

This is the semantic authority for the planned text-only target
`gemma4-31b-it/groupwise-int`. It records the exact upstream sources and mathematical
boundaries that conversion, Ops, the family Program, and the assistant must preserve. Vision,
audio, and video are outside the current product scope.

## Pinned authority

| Resource | Immutable revision | Checkpoint SHA-256 | Bytes |
|---|---|---:|---:|
| `google/gemma-4-31B-it-qat-w4a16-ct` | `52f3f65bc7a02d555763bc923bd1d9094898219d` | `1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20` | 23,265,352,448 |
| `google/gemma-4-31B-it-assistant` | `627c5ec1458b9086b841a91e0512fd31fd2fbbf1` | `9f80df6099fa1fd7db71220ec9ee864d5ecff769878697dd5763e4285b15a1da` | 939,042,560 |
| Hugging Face Transformers | `d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a` | n/a | n/a |

The target and assistant repositories are public and ungated at these revisions. The committed
fixture metadata records hashes for `config.json`, generation configuration, tokenizer resources,
chat template, and processor configuration. The reference environment uses Python 3.11, PyTorch
2.14.1, NumPy 2.4.6, tokenizers 0.22.2, safetensors 0.8.0, and the Transformers commit above.

## Text architecture

The target has a 262,144-token vocabulary, hidden width 5,376, MLP width 21,504, 60 dense decoder
layers, tied input/output embeddings, and a maximum position of 262,144. Layers repeat five
`sliding_attention` layers followed by one `full_attention` layer. All linears are bias-free. The
activation is GELU with the tanh approximation. RMSNorm epsilon is `1e-6`.

| Property | Sliding attention | Full attention |
|---|---:|---:|
| Query heads | 32 | 32 |
| K/V heads | 16 | 4 |
| Head dimension | 256 | 512 |
| Window | 1,024 | full causal context |
| RoPE theta | 10,000 | 1,000,000 |
| Rotary fraction | 1.0 | 0.25 |
| K/V projection | distinct K and V | one K projection supplies both sources |

The upstream attention score multiplier is exactly `1.0`; do not introduce the usual
`1/sqrt(head_dim)` factor. Scores are softmaxed in FP32 and cast back to the query dtype. The model
does not apply a text-attention logit cap. Final vocabulary logits are capped as
`30 * tanh(logit / 30)`.

### Embedding and normalization boundaries

Embedding lookup is multiplied by `sqrt(5376)`, with the scalar cast to the embedding weight dtype
before multiplication. The authoritative BF16 scalar is captured in `semantic.npz`.

RMSNorm converts its BF16 input to FP32, computes
`x * pow(mean(x^2) + 1e-6, -0.5)`, optionally multiplies an FP32 learned scale, and casts to the
input dtype. Values use a learned scale for query, key, decoder, and final norms. The value-head
normalization is scale-less.

Each decoder layer performs:

1. input RMSNorm;
2. self-attention;
3. post-attention RMSNorm;
4. first residual add;
5. pre-feedforward RMSNorm;
6. `down(gelu_tanh(gate(x)) * up(x))`;
7. post-feedforward RMSNorm;
8. second residual add.

There are no per-layer embeddings or MoE blocks in this target. Each completed layer is multiplied
by its learned BF16 scalar; conversion retains all 60 scalars as explicit artifact tensors.

### RoPE

Sliding attention uses ordinary full-width RoPE. For full attention, only 25% of the 512 channels
rotate. The proportional initializer creates 64 active inverse-frequency values using denominator
512, then appends 192 zeros. Concatenating the 256 frequencies with themselves creates 512-wide
cosine/sine vectors. The zero-frequency lanes therefore have cosine one and sine zero and remain
unchanged. Positions and trigonometry are evaluated in FP32 before casting to the input dtype.

### Full-attention K equals V

Full layers omit `v_proj`. The K projection result is separately passed through learned K RMSNorm
and RoPE for the key path, and through scale-less RMSNorm without RoPE for the value path. Sharing
the projected source does not mean the stored key and value vectors are equal.

## Assistant/MTP contract

The official assistant has width 1,024, MLP width 8,192, four layers ordered sliding, sliding,
sliding, full, and an assistant embedding tied to its own 1,024-wide LM head. Draft-step input uses
the target's separate 5,376-wide embedding. The assistant has no K/V projections or K/V norms:
every assistant layer consumes target shared K/V states. Target layer 58 publishes the last
complete sliding K/V state, and target layer 59 publishes the last complete full K/V state.
Assistant layers 0-2 use the
former; layer 3 uses the latter.

For the first assisted round the target runs normally. For each proposed token, the assistant:

1. selects the target hidden state associated with the last accepted token;
2. looks up that token in the tied target embedding;
3. concatenates `[target_embedding, target_hidden]`, width 10,752;
4. preprojects to width 1,024;
5. runs the four-layer assistant using the two shared target K/V views;
6. postprojects to width 5,376;
7. independently applies the assistant's tied 1,024-wide LM head to the assistant body output and
   samples the next draft token.

The assistant position ID remains `input_length - 1` throughout one draft proposal loop. The
default upstream proposal budget is six; NInfer qualifies one token first and tunes two through six
only after MTP1 parity.

## Quantized checkpoint

The target uses compressed-tensors symmetric signed INT4, group size 32, for text `Linear`
weights. Input and output activations are not quantized. The LM head is ignored by compression and
is tied to the embedding. Conversion must decode the stored packed values and stored per-group
scales independently before applying NInfer's artifact recipe; it must not infer scales from
already-quantized values or repack at runtime.

## Executable fixtures

[`tools/reference/gemma4_31b_it`](../../tools/reference/gemma4_31b_it/README.md) owns the generator.
[`tests/fixtures/gemma4_31b_it`](../../tests/fixtures/gemma4_31b_it/) stores both fixture sets.
Every array has a canonical dtype/shape/content hash in `metadata.json`. The fixtures expose all
inputs and intermediate boundaries needed to localize parity failures rather than relying on final
text. The committed checkpoint fixture contains selected results, not source weights. Regeneration
is source-gated because the two checkpoints exceed 24 GB; CI exercises the compact semantic set.
