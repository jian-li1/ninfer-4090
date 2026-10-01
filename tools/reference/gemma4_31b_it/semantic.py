"""Independent Gemma 4 text-path reference math used to build small fixtures.

The functions in this file intentionally spell out the public mathematical
boundaries.  They do not import the NInfer implementation or Transformers.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def bf16(value: torch.Tensor) -> torch.Tensor:
    """Round to BF16 and return FP32 so NumPy can persist the values."""
    return value.to(torch.bfloat16).to(torch.float32)


def deterministic_tensor(shape: tuple[int, ...], phase: float, scale: float = 0.125) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.float32)
    value = torch.sin(index * 0.173 + phase) + 0.5 * torch.cos(index * 0.071 - phase)
    return bf16((value * scale).reshape(shape))


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor | None, eps: float = 1.0e-6) -> torch.Tensor:
    hidden_bf16 = bf16(hidden)
    mean_squared = hidden_bf16.float().pow(2).mean(dim=-1, keepdim=True) + eps
    output = hidden_bf16.float() * torch.pow(mean_squared, -0.5)
    if weight is not None:
        output = output * weight.float()
    return bf16(output)


def rope_frequencies(
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
    partial_rotary_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotary_pairs = int(partial_rotary_factor * head_dim // 2)
    active = 1.0 / (
        theta ** (torch.arange(0, 2 * rotary_pairs, 2, dtype=torch.float32) / head_dim)
    )
    inactive = torch.zeros(head_dim // 2 - rotary_pairs, dtype=torch.float32)
    inverse_frequency = torch.cat((active, inactive))
    frequency = positions.float()[:, None] * inverse_frequency[None, :]
    embedding = torch.cat((frequency, frequency), dim=-1)
    return inverse_frequency, bf16(torch.cos(embedding)), bf16(torch.sin(embedding))


def apply_rope(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return bf16(bf16(value) * cosine + bf16(rotated) * sine)


def linear(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return bf16(torch.matmul(bf16(hidden), bf16(weight).transpose(-1, -2)))


def gelu_tanh(hidden: torch.Tensor) -> torch.Tensor:
    return bf16(F.gelu(bf16(hidden), approximate="tanh"))


def mlp(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = linear(hidden, gate_weight)
    up = linear(hidden, up_weight)
    product = bf16(gelu_tanh(gate) * up)
    return gate, up, product, linear(product, down_weight)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sliding_window: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma 4 eager attention with unit score scaling and a causal mask."""
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(bf16(query), bf16(key).transpose(-1, -2)).float()
    query_count = query.shape[-2]
    key_count = key.shape[-2]
    q_index = torch.arange(query_count)[:, None]
    k_index = torch.arange(key_count)[None, :]
    valid = k_index <= q_index
    if sliding_window is not None:
        valid &= (q_index - k_index) < sliding_window
    scores = scores.masked_fill(~valid[None, None, :, :], -torch.inf)
    probabilities = bf16(torch.softmax(scores, dim=-1, dtype=torch.float32))
    output = bf16(torch.matmul(probabilities, bf16(value))).transpose(1, 2).contiguous()
    return output, probabilities


def qkv_transform(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    theta: float,
    partial_rotary_factor: float,
) -> dict[str, torch.Tensor]:
    sequence = hidden.shape[-2]
    q_projected = linear(hidden, q_weight).reshape(1, sequence, q_heads, head_dim)
    k_projected = linear(hidden, k_weight).reshape(1, sequence, kv_heads, head_dim)
    v_projected = k_projected if v_weight is None else linear(hidden, v_weight).reshape(
        1, sequence, kv_heads, head_dim
    )
    q_normalized = rms_norm(q_projected, q_norm_weight)
    k_normalized = rms_norm(k_projected, k_norm_weight)
    v_normalized = rms_norm(v_projected, None)
    inverse_frequency, cosine, sine = rope_frequencies(
        positions, head_dim, theta, partial_rotary_factor
    )
    q_rotary = apply_rope(q_normalized, cosine[None, :, None, :], sine[None, :, None, :])
    k_rotary = apply_rope(k_normalized, cosine[None, :, None, :], sine[None, :, None, :])
    return {
        "q_projected": q_projected,
        "k_projected": k_projected,
        "v_projected": v_projected,
        "q_normalized": q_normalized,
        "k_normalized": k_normalized,
        "v_normalized": v_normalized,
        "inverse_frequency": inverse_frequency,
        "cosine": cosine,
        "sine": sine,
        "query": q_rotary.transpose(1, 2),
        "key": k_rotary.transpose(1, 2),
        "value": v_normalized.transpose(1, 2),
    }


def decoder_layer(
    hidden: torch.Tensor,
    prefix: str,
    head_dim: int,
    theta: float,
    partial_rotary_factor: float,
    sliding_window: int | None,
    attention_k_eq_v: bool,
) -> dict[str, torch.Tensor]:
    """Small deterministic layer with the production Gemma 4 operation order."""
    hidden_size = hidden.shape[-1]
    q_heads = 2
    kv_heads = 1
    positions = torch.arange(hidden.shape[-2], dtype=torch.int64)
    weights = {
        "input_norm": bf16(1.0 + deterministic_tensor((hidden_size,), 0.1, 0.03)),
        "q_norm": bf16(1.0 + deterministic_tensor((head_dim,), 0.2, 0.03)),
        "k_norm": bf16(1.0 + deterministic_tensor((head_dim,), 0.3, 0.03)),
        "q": deterministic_tensor((q_heads * head_dim, hidden_size), 0.4),
        "k": deterministic_tensor((kv_heads * head_dim, hidden_size), 0.5),
        "v": deterministic_tensor((kv_heads * head_dim, hidden_size), 0.6),
        "o": deterministic_tensor((hidden_size, q_heads * head_dim), 0.7),
        "post_attn_norm": bf16(1.0 + deterministic_tensor((hidden_size,), 0.8, 0.03)),
        "pre_ffn_norm": bf16(1.0 + deterministic_tensor((hidden_size,), 0.9, 0.03)),
        "gate": deterministic_tensor((32, hidden_size), 1.0),
        "up": deterministic_tensor((32, hidden_size), 1.1),
        "down": deterministic_tensor((hidden_size, 32), 1.2),
        "post_ffn_norm": bf16(1.0 + deterministic_tensor((hidden_size,), 1.3, 0.03)),
    }
    input_normalized = rms_norm(hidden, weights["input_norm"])
    transformed = qkv_transform(
        input_normalized,
        weights["q"],
        weights["k"],
        None if attention_k_eq_v else weights["v"],
        weights["q_norm"],
        weights["k_norm"],
        positions,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        partial_rotary_factor,
    )
    attended, probabilities = attention(
        transformed["query"], transformed["key"], transformed["value"], sliding_window
    )
    attention_projection = linear(attended.reshape(1, hidden.shape[-2], -1), weights["o"])
    post_attention = rms_norm(attention_projection, weights["post_attn_norm"])
    attention_residual = bf16(hidden + post_attention)
    pre_feedforward = rms_norm(attention_residual, weights["pre_ffn_norm"])
    gate, up, product, feedforward = mlp(
        pre_feedforward, weights["gate"], weights["up"], weights["down"]
    )
    post_feedforward = rms_norm(feedforward, weights["post_ffn_norm"])
    output = bf16(attention_residual + post_feedforward)
    result = {
        f"{prefix}_input": hidden,
        f"{prefix}_input_normalized": input_normalized,
        f"{prefix}_attention_probabilities": probabilities,
        f"{prefix}_attention_output": attended,
        f"{prefix}_attention_projection": attention_projection,
        f"{prefix}_post_attention": post_attention,
        f"{prefix}_attention_residual": attention_residual,
        f"{prefix}_pre_feedforward": pre_feedforward,
        f"{prefix}_mlp_gate": gate,
        f"{prefix}_mlp_up": up,
        f"{prefix}_mlp_product": product,
        f"{prefix}_mlp_output": feedforward,
        f"{prefix}_post_feedforward": post_feedforward,
        f"{prefix}_output": output,
    }
    result.update({f"{prefix}_{name}": value for name, value in transformed.items()})
    return result


def generate_semantic_fixtures() -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}

    norm_input = deterministic_tensor((2, 16), 0.15)
    norm_weight = bf16(1.0 + deterministic_tensor((16,), 0.35, 0.04))
    result.update(
        {
            "rmsnorm_input": norm_input,
            "rmsnorm_weight": norm_weight,
            "rmsnorm_scaled": rms_norm(norm_input, norm_weight),
            "rmsnorm_unscaled": rms_norm(norm_input, None),
        }
    )

    positions = torch.tensor([0, 1, 17, 1023, 262143], dtype=torch.int64)
    local_input = deterministic_tensor((positions.numel(), 256), 0.45)
    local_inv, local_cos, local_sin = rope_frequencies(positions, 256, 10_000.0)
    global_input = deterministic_tensor((positions.numel(), 512), 0.55)
    global_inv, global_cos, global_sin = rope_frequencies(positions, 512, 1_000_000.0, 0.25)
    result.update(
        {
            "rope_positions": positions,
            "local_rope_input": local_input,
            "local_rope_inverse_frequency": local_inv,
            "local_rope_cosine": local_cos,
            "local_rope_sine": local_sin,
            "local_rope_output": apply_rope(local_input, local_cos, local_sin),
            "global_rope_input": global_input,
            "global_rope_inverse_frequency": global_inv,
            "global_rope_cosine": global_cos,
            "global_rope_sine": global_sin,
            "global_rope_output": apply_rope(global_input, global_cos, global_sin),
        }
    )

    hidden = deterministic_tensor((1, 4, 16), 0.75)
    local_layer = decoder_layer(hidden, "sliding", 8, 10_000.0, 1.0, 3, False)
    result.update(local_layer)
    full_layer = decoder_layer(local_layer["sliding_output"], "full", 16, 1_000_000.0, 0.25, None, True)
    result.update(full_layer)

    final_weight = bf16(1.0 + deterministic_tensor((16,), 1.45, 0.04))
    final_hidden = rms_norm(full_layer["full_output"], final_weight)
    embedding_weight = deterministic_tensor((32, 16), 1.55)
    embedding_ids = torch.tensor([2, 7, 15, 31], dtype=torch.int64)
    embedding_scale = bf16(torch.tensor(math.sqrt(5376.0))).item()
    embeddings = bf16(embedding_weight[embedding_ids] * embedding_scale)
    logits_unsoftened = linear(final_hidden, embedding_weight)
    logits = bf16(30.0 * torch.tanh(logits_unsoftened.float() / 30.0))
    result.update(
        {
            "embedding_ids": embedding_ids,
            "embedding_weight": embedding_weight,
            "embedding_scale_bf16": torch.tensor([embedding_scale], dtype=torch.float32),
            "embeddings": embeddings,
            "final_norm_weight": final_weight,
            "final_hidden": final_hidden,
            "logits_unsoftened": logits_unsoftened,
            "logits": logits,
            "greedy_next_token": torch.argmax(logits[:, -1, :], dim=-1),
        }
    )

    # MTP1 contract: [target token embedding, accepted target hidden] -> pre-project
    # -> assistant body -> assistant LM head plus post-project feedback.
    mtp_token = torch.tensor([7], dtype=torch.int64)
    mtp_embedding = embeddings[1:2]
    mtp_target_hidden = final_hidden[:, -1, :]
    mtp_concat = torch.cat((mtp_embedding, mtp_target_hidden), dim=-1)
    mtp_pre_weight = deterministic_tensor((16, 32), 1.65)
    mtp_body_input = linear(mtp_concat, mtp_pre_weight)
    mtp_body = decoder_layer(mtp_body_input[:, None, :], "mtp", 8, 10_000.0, 1.0, None, False)[
        "mtp_output"
    ][:, 0, :]
    mtp_post_weight = deterministic_tensor((16, 16), 1.75)
    mtp_feedback = linear(mtp_body, mtp_post_weight)
    # The assistant LM head is tied to its own 1,024-wide embedding table.
    mtp_lm_weight = deterministic_tensor((32, 16), 1.85)
    mtp_logits = linear(mtp_body, mtp_lm_weight)
    result.update(
        {
            "mtp_last_token_id": mtp_token,
            "mtp_position_id": torch.tensor([3], dtype=torch.int64),
            "mtp_target_embedding": mtp_embedding,
            "mtp_target_hidden": mtp_target_hidden,
            "mtp_concat": mtp_concat,
            "mtp_body_input": mtp_body_input,
            "mtp_body_output": mtp_body,
            "mtp_feedback": mtp_feedback,
            "mtp_lm_weight": mtp_lm_weight,
            "mtp_logits": mtp_logits,
            "mtp_draft_token": torch.argmax(mtp_logits, dim=-1),
        }
    )
    return result
