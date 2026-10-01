#!/usr/bin/env python3
"""Generate source-gated fixtures from the official Gemma target and assistant.

This deliberately streams one packed target matrix at a time.  It is a readable
reference path, not an inference implementation or a production conversion route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

from generate import (
    ASSISTANT_REPOSITORY,
    ASSISTANT_REVISION,
    TARGET_REPOSITORY,
    TARGET_REVISION,
    TRANSFORMERS_REVISION,
    array_hash,
    sha256,
    verify_resources,
)


TARGET_CHECKPOINT_SHA256 = "1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20"
ASSISTANT_CHECKPOINT_SHA256 = "9f80df6099fa1fd7db71220ec9ee864d5ecff769878697dd5763e4285b15a1da"
PROMPT = "Hello, Gemma 4!"
TOKEN_IDS = [9259, 236764, 147224, 236743, 236812, 236888]


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16)


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor | None, eps: float = 1.0e-6) -> torch.Tensor:
    output = hidden.float() * torch.pow(hidden.float().pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        output = output * weight.float()
    return output.to(hidden.dtype)


def rope_frequencies(
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
    partial_rotary_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_pairs = int(partial_rotary_factor * head_dim // 2)
    active = 1.0 / (theta ** (torch.arange(0, 2 * active_pairs, 2).float() / head_dim))
    inverse_frequency = torch.cat((active, torch.zeros(head_dim // 2 - active_pairs)))
    frequency = positions.float()[:, None] * inverse_frequency[None, :]
    embedding = torch.cat((frequency, frequency), dim=-1)
    return embedding.cos().to(torch.bfloat16), embedding.sin().to(torch.bfloat16)


def apply_rope(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cosine + rotated * sine


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sliding_window: int | None,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    score = torch.matmul(query, key.transpose(-1, -2)).float()
    if causal:
        queries = torch.arange(query.shape[-2])[:, None]
        keys = torch.arange(key.shape[-2])[None, :]
        valid = keys <= queries
        if sliding_window is not None:
            valid &= (queries - keys) < sliding_window
        score.masked_fill_(~valid[None, None, :, :], -torch.inf)
    probability = torch.softmax(score, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probability, value).transpose(1, 2).contiguous()
    return output, probability


class TargetWeights:
    def __init__(self, checkpoint: Path, row_chunk: int):
        self.handle = safe_open(checkpoint, framework="pt", device="cpu")
        self.row_chunk = row_chunk

    def tensor(self, name: str) -> torch.Tensor:
        return self.handle.get_tensor(name)

    def linear(self, prefix: str, hidden: torch.Tensor) -> torch.Tensor:
        packed = self.tensor(prefix + ".weight_packed")
        scale = self.tensor(prefix + ".weight_scale")
        shape = tuple(int(value) for value in self.tensor(prefix + ".weight_shape").tolist())
        if packed.shape[0] != shape[0] or scale.shape != (shape[0], shape[1] // 32):
            raise RuntimeError(f"invalid packed geometry for {prefix}")

        outputs = []
        for first in range(0, shape[0], self.row_chunk):
            last = min(first + self.row_chunk, shape[0])
            packed_rows = packed[first:last]
            codes = torch.empty((last - first, packed_rows.shape[1] * 8), dtype=torch.int8)
            for nibble in range(8):
                unsigned = (packed_rows >> (4 * nibble)) & 0xF
                codes[:, nibble::8] = (unsigned - 8).to(torch.int8)
            codes = codes[:, : shape[1]]
            weight = (
                codes.reshape(last - first, -1, 32).to(scale.dtype)
                * scale[first:last, :, None]
            ).reshape(last - first, shape[1])
            outputs.append(F.linear(hidden, weight))
        return torch.cat(outputs, dim=-1)


def target_qkv(
    weights: TargetWeights,
    hidden: torch.Tensor,
    layer: int,
    positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer}.self_attn"
    full = layer % 6 == 5
    head_dim = 512 if full else 256
    kv_heads = 4 if full else 16
    q = weights.linear(prefix + ".q_proj", hidden).reshape(1, hidden.shape[1], 32, head_dim)
    k_projected = weights.linear(prefix + ".k_proj", hidden).reshape(
        1, hidden.shape[1], kv_heads, head_dim
    )
    v_projected = (
        k_projected
        if full
        else weights.linear(prefix + ".v_proj", hidden).reshape(1, hidden.shape[1], kv_heads, head_dim)
    )
    q = rms_norm(q, weights.tensor(prefix + ".q_norm.weight"))
    k = rms_norm(k_projected, weights.tensor(prefix + ".k_norm.weight"))
    v = rms_norm(v_projected, None)
    cosine, sine = rope_frequencies(
        positions,
        head_dim,
        1_000_000.0 if full else 10_000.0,
        0.25 if full else 1.0,
    )
    q = apply_rope(q, cosine[None, :, None, :], sine[None, :, None, :]).transpose(1, 2)
    k = apply_rope(k, cosine[None, :, None, :], sine[None, :, None, :]).transpose(1, 2)
    return {
        "q": q,
        "k": k,
        "v": v.transpose(1, 2),
        "k_projected": k_projected,
        "v_projected": v_projected,
    }


def run_target(checkpoint: Path, row_chunk: int) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    weights = TargetWeights(checkpoint, row_chunk)
    token_ids = torch.tensor([TOKEN_IDS], dtype=torch.int64)
    embedding_weight = weights.tensor("model.language_model.embed_tokens.weight")
    embedding_scale = torch.tensor(math.sqrt(5376.0)).to(embedding_weight.dtype)
    hidden = embedding_weight[token_ids] * embedding_scale
    positions = torch.arange(token_ids.shape[1], dtype=torch.int64)
    result: dict[str, torch.Tensor] = {
        "target_token_ids": token_ids,
        "target_embeddings": hidden,
        "target_embedding_scale": embedding_scale.reshape(1),
    }
    shared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for layer in range(60):
        layer_prefix = f"model.language_model.layers.{layer}"
        residual = hidden
        input_normalized = rms_norm(hidden, weights.tensor(layer_prefix + ".input_layernorm.weight"))
        qkv = target_qkv(weights, input_normalized, layer, positions)
        full = layer % 6 == 5
        attended, probabilities = attention(
            qkv["q"], qkv["k"], qkv["v"], None if full else 1024, True
        )
        projected = weights.linear(
            layer_prefix + ".self_attn.o_proj", attended.reshape(1, hidden.shape[1], -1)
        )
        post_attention = rms_norm(
            projected, weights.tensor(layer_prefix + ".post_attention_layernorm.weight")
        )
        attention_residual = residual + post_attention

        pre_feedforward = rms_norm(
            attention_residual, weights.tensor(layer_prefix + ".pre_feedforward_layernorm.weight")
        )
        gate = weights.linear(layer_prefix + ".mlp.gate_proj", pre_feedforward)
        up = weights.linear(layer_prefix + ".mlp.up_proj", pre_feedforward)
        product = F.gelu(gate, approximate="tanh") * up
        feedforward = weights.linear(layer_prefix + ".mlp.down_proj", product)
        post_feedforward = rms_norm(
            feedforward, weights.tensor(layer_prefix + ".post_feedforward_layernorm.weight")
        )
        hidden = (attention_residual + post_feedforward) * weights.tensor(layer_prefix + ".layer_scalar")
        if (layer + 1) % 5 == 0:
            print(f"target layers: {layer + 1}/60", file=sys.stderr, flush=True)

        if layer in (0, 5):
            name = f"target_layer{layer}"
            result.update(
                {
                    name + "_input_normalized": input_normalized,
                    name + "_q": qkv["q"],
                    name + "_k": qkv["k"],
                    name + "_v": qkv["v"],
                    name + "_k_projected": qkv["k_projected"],
                    name + "_v_projected": qkv["v_projected"],
                    name + "_attention_probabilities": probabilities,
                    name + "_attention_output": attended,
                    name + "_attention_projection": projected,
                    name + "_attention_residual": attention_residual,
                    name + "_pre_feedforward": pre_feedforward,
                    name + "_mlp_gate": gate,
                    name + "_mlp_up": up,
                    name + "_mlp_product": product,
                    name + "_mlp_output": feedforward,
                    name + "_output": hidden,
                }
            )
        if layer == 58:
            shared["sliding_attention"] = (qkv["k"], qkv["v"])
        if layer == 59:
            shared["full_attention"] = (qkv["k"], qkv["v"])

    final_hidden = rms_norm(hidden, weights.tensor("model.language_model.norm.weight"))
    logits_unsoftened = F.linear(final_hidden[:, -1, :], embedding_weight)
    logits = 30.0 * torch.tanh(logits_unsoftened / 30.0)
    top_values, top_ids = torch.topk(logits.float(), k=32, dim=-1)
    result.update(
        {
            "target_final_hidden": final_hidden,
            "target_last_logits": logits,
            "target_top32_ids": top_ids,
            "target_top32_logits": top_values,
            "target_greedy_next_token": top_ids[:, :1],
            "target_shared_sliding_k": shared["sliding_attention"][0],
            "target_shared_sliding_v": shared["sliding_attention"][1],
            "target_shared_full_k": shared["full_attention"][0],
            "target_shared_full_v": shared["full_attention"][1],
        }
    )
    return result, shared


class DenseWeights:
    def __init__(self, checkpoint: Path):
        self.handle = safe_open(checkpoint, framework="pt", device="cpu")

    def tensor(self, name: str) -> torch.Tensor:
        return self.handle.get_tensor(name)

    def linear(self, name: str, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.tensor(name))


def run_assistant(
    checkpoint: Path,
    target_result: dict[str, torch.Tensor],
    shared: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    weights = DenseWeights(checkpoint)
    last_token_id = target_result["target_token_ids"][:, -1:]
    target_embedding = target_result["target_embeddings"][:, -1, :]
    target_hidden = target_result["target_final_hidden"][:, -1, :]
    concatenated = torch.cat((target_embedding, target_hidden), dim=-1)
    hidden = weights.linear("pre_projection.weight", concatenated)[:, None, :]
    position = torch.tensor([[len(TOKEN_IDS) - 1]], dtype=torch.int64)
    result = {
        "assistant_last_token_id": last_token_id,
        "assistant_position_id": position,
        "assistant_target_embedding": target_embedding,
        "assistant_target_hidden": target_hidden,
        "assistant_concat": concatenated,
        "assistant_preprojection": hidden,
    }

    for layer in range(4):
        prefix = f"model.layers.{layer}"
        full = layer == 3
        head_dim = 512 if full else 256
        residual = hidden
        normalized = rms_norm(hidden, weights.tensor(prefix + ".input_layernorm.weight"))
        query = weights.linear(prefix + ".self_attn.q_proj.weight", normalized).reshape(
            1, 1, 32, head_dim
        )
        query = rms_norm(query, weights.tensor(prefix + ".self_attn.q_norm.weight"))
        cosine, sine = rope_frequencies(
            position.reshape(-1), head_dim, 1_000_000.0 if full else 10_000.0, 0.25 if full else 1.0
        )
        query = apply_rope(query, cosine[None, :, None, :], sine[None, :, None, :]).transpose(1, 2)
        key, value = shared["full_attention" if full else "sliding_attention"]
        attended, probability = attention(query, key, value, None, False)
        projected = weights.linear(
            prefix + ".self_attn.o_proj.weight", attended.reshape(1, 1, -1)
        )
        hidden = residual + rms_norm(
            projected, weights.tensor(prefix + ".post_attention_layernorm.weight")
        )
        residual = hidden
        normalized = rms_norm(hidden, weights.tensor(prefix + ".pre_feedforward_layernorm.weight"))
        gate = weights.linear(prefix + ".mlp.gate_proj.weight", normalized)
        up = weights.linear(prefix + ".mlp.up_proj.weight", normalized)
        product = F.gelu(gate, approximate="tanh") * up
        feedforward = weights.linear(prefix + ".mlp.down_proj.weight", product)
        hidden = (
            residual
            + rms_norm(feedforward, weights.tensor(prefix + ".post_feedforward_layernorm.weight"))
        ) * weights.tensor(prefix + ".layer_scalar")
        result.update(
            {
                f"assistant_layer{layer}_query": query,
                f"assistant_layer{layer}_attention_probabilities": probability,
                f"assistant_layer{layer}_output": hidden,
            }
        )

    body_output = rms_norm(hidden, weights.tensor("model.norm.weight"))
    feedback = weights.linear("post_projection.weight", body_output)
    logits = F.linear(body_output, weights.tensor("model.embed_tokens.weight"))
    top_values, top_ids = torch.topk(logits.float(), k=32, dim=-1)
    result.update(
        {
            "assistant_body_output": body_output,
            "assistant_feedback": feedback,
            "assistant_logits": logits,
            "assistant_top32_ids": top_ids,
            "assistant_top32_logits": top_values,
            "assistant_draft_token": top_ids[..., :1],
        }
    )
    return result


def generate(source_dir: Path, output_dir: Path, row_chunk: int) -> None:
    verify_resources(source_dir)
    target_path = source_dir / "target" / "model.safetensors"
    assistant_path = source_dir / "assistant" / "model.safetensors"
    for path, expected in (
        (target_path, TARGET_CHECKPOINT_SHA256),
        (assistant_path, ASSISTANT_CHECKPOINT_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(12)
    with torch.inference_mode():
        target_result, shared = run_target(target_path, row_chunk)
        assistant_result = run_assistant(assistant_path, target_result, shared)
    tensors = {**target_result, **assistant_result}
    arrays = {
        name: value.detach().cpu().to(torch.float32).numpy()
        if value.dtype == torch.bfloat16
        else value.detach().cpu().numpy()
        for name, value in sorted(tensors.items())
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "checkpoint.npz", **arrays)
    metadata = {
        "schema": "ninfer.gemma4.checkpoint-fixtures.v1",
        "scope": "text-only target full forward and official assistant MTP1",
        "prompt": PROMPT,
        "token_ids": TOKEN_IDS,
        "target": {
            "repository": TARGET_REPOSITORY,
            "revision": TARGET_REVISION,
            "checkpoint_sha256": TARGET_CHECKPOINT_SHA256,
        },
        "assistant": {
            "repository": ASSISTANT_REPOSITORY,
            "revision": ASSISTANT_REVISION,
            "checkpoint_sha256": ASSISTANT_CHECKPOINT_SHA256,
        },
        "transformers_revision": TRANSFORMERS_REVISION,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": array_hash(value),
            }
            for name, value in arrays.items()
        },
    }
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-chunk", type=int, default=512)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args(sys.argv[1:])
    generate(arguments.source_dir, arguments.output_dir, arguments.row_chunk)
