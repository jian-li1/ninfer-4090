# Gemma 4 31B reference fixtures

This tool is the executable mathematical authority for the text-only Gemma 4 target. The fast
fixture set is independent of NInfer and exercises BF16 embedding scaling, scaled and scale-less
RMSNorm, local and proportional partial RoPE, local and full Q/K/V transforms, unit-scaled grouped
query attention, GELU-tanh MLP, both decoder-layer kinds, final logit soft-capping, and the MTP1
concatenation/projection contract.

Create an isolated Python 3.11 environment and install the exact dependencies:

```bash
uv venv --python 3.11 /tmp/ninfer-gemma4-reference
uv pip install --python /tmp/ninfer-gemma4-reference \
  -r tools/reference/gemma4_31b_it/requirements.txt
```

Stage the immutable Hugging Face resources under one directory with `target/` and `assistant/`
children. The revisions and every required small-resource SHA-256 are embedded in `generate.py`.
The generator rejects missing or changed resources before tokenizing:

```bash
/tmp/ninfer-gemma4-reference/bin/python \
  tools/reference/gemma4_31b_it/generate.py \
  --source-dir /dev/shm/ninfer-gemma4-sources \
  --output-dir tests/fixtures/gemma4_31b_it
```

Omit `--source-dir` to regenerate only `semantic.npz` and its metadata in CI. Compare arrays and
their canonical hashes from `metadata.json`; the ZIP container timestamp is not a semantic value.
Full-checkpoint integration fixtures are source-gated because the target checkpoint is 23.27 GB.
Place the two verified `model.safetensors` files beside the small resources and run:

```bash
/tmp/ninfer-gemma4-reference/bin/python \
  tools/reference/gemma4_31b_it/generate_checkpoint.py \
  --source-dir /dev/shm/ninfer-gemma4-sources \
  --output-dir tests/fixtures/gemma4_31b_it
```

The checkpoint generator verifies both complete SHA-256 hashes, independently decodes the signed
INT4/group-32 target weights from their stored scales, and streams one output-row block at a time.
It emits an actual 60-layer short-prompt forward, selected local/full layer substages, final logits,
the target layer-58/layer-59 shared K/V views, and one official assistant proposal. It is an oracle
