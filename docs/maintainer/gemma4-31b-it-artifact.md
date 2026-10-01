# Gemma 4 31B IT artifact

This document is the storage authority for the registered text-only artifact identity
`gemma4-31b-it/groupwise-int`. The converter writes NInfer container v2 and never places
Hugging Face tensors, runtime repacking, vision weights, or unclassified source tensors on the
product load path.

## Reproducible conversion

The immutable sources and checkpoint hashes are recorded in
[Gemma 4 31B IT text model](gemma4-31b-it-model.md). Generate the complete target-plus-assistant
artifact with Python 3.11:

```bash
python -m tools.convert.gemma4_31b_it.convert \
  --model /path/to/gemma-4-31B-it-qat-w4a16-ct \
  --assistant /path/to/gemma-4-31B-it-assistant \
  --out /dev/shm/gemma4_31b_it.ninfer \
  --device cuda
```

Omit `--assistant` for a target-only artifact. The converter verifies the complete checkpoint
SHA-256 before writing, validates the exact config and source inventory, and refuses any
unclassified text tensor. It emits `<artifact>.conversion.json`; its `objects` array is the exact
expanded per-object table, including source names, runtime name, shape, format, layout, transform,
payload bytes, and payload SHA-256. That generated table is authoritative for checksums because a
checksum necessarily belongs to one artifact instance rather than to this source-level contract.

## Closed object inventory

Names below use `L = 0..59`, with full-attention layers `L = 5, 11, ..., 59`; `A = 0..3`, with
assistant layer 3 full. Every patterned row expands once per applicable layer. `N/K` shapes are in
the runtime linear convention. Direct tensors use `contiguous-le-v1`; grouped tensors use
`row-split-k128-v1`; row-scaled FP8 uses `row-scale-v1`.

| Source tensor(s) | Source shape/storage | Semantic role | Runtime object | Runtime shape/format | Transform, tying, bytes |
|---|---|---|---|---|---|
| `tokenizer.json` | raw | tokenizer | `frontend/tokenizer.json` | `raw-bytes-v1` | verbatim; source bytes |
| `tokenizer_config.json` | raw | tokenizer config | `frontend/tokenizer_config.json` | `raw-bytes-v1` | verbatim; source bytes |
| `chat_template.jinja` | raw | chat template | `frontend/chat_template.jinja` | `raw-bytes-v1` | verbatim; source bytes |
| `generation_config.json` | raw | generation defaults | `frontend/generation_config.json` | `raw-bytes-v1` | verbatim; source bytes |
| `model.language_model.embed_tokens.weight` | `[262144,5376] BF16` | embedding and tied head | `text/token_embedding` | `[262144,5376] FP8_E4M3FN_ROW_BF16S` | row-scale profile `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1`; one physical object; 1,409,810,432 bytes |
| `...layers.L.input_layernorm.weight` | `[5376] BF16` | input norm | `text/layers/L/input_norm` | `[5376] BF16` | verbatim; 10,752 bytes |
| sliding `q_proj`, `k_proj`, `v_proj` packed/scales/shapes | `[8192,5376]`, `[4096,5376]`, `[4096,5376]`; signed symmetric INT4 group 32 | Q/K/V input | `text/layers/L/attention/input_projection` | `[16384,5376] Q4G64_F16S` | decode, row fuse, requantize; 46,792,704 bytes |
| full `q_proj`, `k_proj` packed/scales/shapes | `[16384,5376]`, `[2048,5376]`; signed symmetric INT4 group 32 | Q/K input; projected K also supplies raw V | `text/layers/L/attention/input_projection` | `[18432,5376] Q4G64_F16S` | decode, row fuse, requantize; 52,641,792 bytes |
| `...layers.L.self_attn.q_norm.weight` | `[256]` sliding / `[512]` full BF16 | query norm | `text/layers/L/attention/query_norm` | same shape BF16 | verbatim; 512 / 1,024 bytes |
| `...layers.L.self_attn.k_norm.weight` | `[256]` sliding / `[512]` full BF16 | key norm | `text/layers/L/attention/key_norm` | same shape BF16 | verbatim; 512 / 1,024 bytes |
| sliding `...o_proj` packed/scales/shape | `[5376,8192]` INT4 group 32 | attention output | `text/layers/L/attention/output` | `[5376,8192] Q4G64_F16S` | decode and requantize; 23,396,352 bytes |
| full `...o_proj` packed/scales/shape | `[5376,16384]` INT4 group 32 | attention output | `text/layers/L/attention/output` | `[5376,16384] Q4G64_F16S` | decode and requantize; 46,792,704 bytes |
| `...post_attention_layernorm.weight` | `[5376] BF16` | post-attention norm | `text/layers/L/post_attention_norm` | `[5376] BF16` | verbatim; 10,752 bytes |
| `...pre_feedforward_layernorm.weight` | `[5376] BF16` | pre-FFN norm | `text/layers/L/pre_feedforward_norm` | `[5376] BF16` | verbatim; 10,752 bytes |
| `...mlp.gate_proj` + `up_proj` packed/scales/shapes | two `[21504,5376]` INT4 group 32 | gated FFN input | `text/layers/L/mlp/gate_up` | `[43008,5376] Q4G64_F16S` | decode, row fuse, requantize; 122,830,848 bytes |
| `...mlp.down_proj` packed/scales/shape | `[5376,21504]` INT4 group 32 | FFN output | `text/layers/L/mlp/down` | `[5376,21504] Q4G64_F16S` | decode and requantize; 61,415,424 bytes |
| `...post_feedforward_layernorm.weight` | `[5376] BF16` | post-FFN norm | `text/layers/L/post_feedforward_norm` | `[5376] BF16` | verbatim; 10,752 bytes |
| `...layers.L.layer_scalar` | `[1] BF16` | learned completed-layer scale | `text/layers/L/layer_scalar` | `[1] BF16` | verbatim; 2 bytes |
| `model.language_model.norm.weight` | `[5376] BF16` | final norm | `text/final_norm` | `[5376] BF16` | verbatim; 10,752 bytes |

The target inventory is therefore exactly 662 tensors. `lm_head.weight` is not duplicated: config
and source identity establish tying, and the runtime output-head view aliases
`text/token_embedding`.

### Optional assistant inventory

| Source tensor(s) | Source shape/storage | Runtime object | Runtime shape/format | Transform, tying |
|---|---|---|---|---|
| `model.embed_tokens.weight` | `[262144,1024] BF16` | `assistant/token_embedding` | `[262144,1024] FP8_E4M3FN_ROW_BF16S` | row-scale profile; assistant head aliases this object |
| `pre_projection.weight` | `[1024,10752] BF16` | `assistant/input_projection` | `[1024,10752] W8G32_F16S` | quantize |
| `model.layers.A.input_layernorm.weight` | `[1024] BF16` | `assistant/layers/A/input_norm` | `[1024] BF16` | verbatim |
| `...self_attn.q_proj.weight` | `[8192,1024]` sliding / `[16384,1024]` full BF16 | `assistant/layers/A/attention/query` | same shape W8G32_F16S | quantize |
| `...self_attn.q_norm.weight` | `[256]` sliding / `[512]` full BF16 | `assistant/layers/A/attention/query_norm` | same shape BF16 | verbatim |
| `...self_attn.o_proj.weight` | `[1024,8192]` sliding / `[1024,16384]` full BF16 | `assistant/layers/A/attention/output` | same shape W8G32_F16S | quantize |
| remaining three assistant decoder norm roles | `[1024] BF16` each | corresponding assistant norm object | `[1024] BF16` | verbatim |
| `...mlp.gate_proj.weight` + `up_proj.weight` | two `[8192,1024] BF16` | `assistant/layers/A/mlp/gate_up` | `[16384,1024] W8G32_F16S` | row fuse and quantize |
| `...mlp.down_proj.weight` | `[1024,8192] BF16` | `assistant/layers/A/mlp/down` | same shape W8G32_F16S | quantize |
| `model.layers.A.layer_scalar` | `[1] BF16` | `assistant/layers/A/layer_scalar` | `[1] BF16` | verbatim |
| `model.norm.weight` | `[1024] BF16` | `assistant/final_norm` | `[1024] BF16` | verbatim |
| `post_projection.weight` | `[5376,1024] BF16` | `assistant/output_projection` | `[5376,1024] W8G32_F16S` | quantize |

The optional assistant adds exactly 44 tensors. Ordinary target execution validates them without
materialization. MTP execution requests the closed assistant binding plan and materializes all 44;
a target-only artifact therefore remains valid for ordinary execution and fails closed for MTP.

## Omissions and memory checkpoint

The pinned source classifies 2,009 target tensors as 1,652 text inputs, 356 deliberately omitted
vision/image-projection tensors, and one tied `lm_head` alias. No processor configuration or
multimodal payload is stored. Unknown or extra artifact objects are rejected by `Binder::finish`.

The target load plan is 16,971,062,784 bytes (15.806 GiB), including object alignment. The
optional assistant package occupies 482,629,120 artifact bytes and is resident only when MTP is
requested. A complete locally generated target-plus-assistant artifact measured 17,485,998,848
bytes and had
SHA-256 `11d70c73f940930035516113a4b1cc25c685111692c4e9f2b345b430234ec83a`.

Representative source-dequantized versus NInfer-dequantized comparisons recorded maximum relative
L2 0.098805 and minimum cosine similarity 0.995299 across sliding/full attention and MLP matrices.
The first 256 embedding rows measured relative L2 0.025692 and cosine 0.999670. The complete values
and per-object checksums are in the generated conversion report.
