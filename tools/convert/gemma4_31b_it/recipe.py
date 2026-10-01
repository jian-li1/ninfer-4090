"""Exact source mapping and materialization for the registered Gemma checkpoint."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from tools.artifact.container import TensorSpec
from tools.convert.common.safetensors import ShardReader, TensorMetadata
from . import inventory


TARGET_REPOSITORY = "google/gemma-4-31B-it-qat-w4a16-ct"
TARGET_REVISION = "52f3f65bc7a02d555763bc923bd1d9094898219d"
TARGET_CHECKPOINT_SHA256 = "1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20"
ASSISTANT_REPOSITORY = "google/gemma-4-31B-it-assistant"
ASSISTANT_REVISION = "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"
ASSISTANT_CHECKPOINT_SHA256 = "9f80df6099fa1fd7db71220ec9ee864d5ecff769878697dd5763e4285b15a1da"


@dataclass(frozen=True, slots=True)
class SourcePreflight:
    tensor_count: int
    dtype_counts: Mapping[str, int]
    used_tensors: tuple[str, ...]
    omitted_multimodal_tensors: tuple[str, ...]
    tied_alias_tensors: tuple[str, ...]


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require(scope: str, actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    for name, value in expected.items():
        if actual.get(name) != value:
            raise ValueError(f"{scope}.{name}={actual.get(name)!r}; expected {value!r}")


def validate_target_config(config: Mapping[str, object]) -> dict[str, object]:
    _require("config", config, {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "dtype": "bfloat16",
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "hidden_act": "gelu_pytorch_tanh",
    })
    text = config.get("text_config")
    quant = config.get("quantization_config")
    if not isinstance(text, Mapping) or not isinstance(quant, Mapping):
        raise ValueError("config must contain text_config and quantization_config objects")
    _require("text_config", text, {
        "num_hidden_layers": 60,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "vocab_size": 262144,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "head_dim": 256,
        "sliding_window": 1024,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_k_eq_v": True,
    })
    expected_layers = ["full_attention" if i in inventory.FULL_LAYERS else "sliding_attention" for i in range(60)]
    if text.get("layer_types") != expected_layers:
        raise ValueError("text_config.layer_types does not match the registered 50/10 schedule")
    groups = quant.get("config_groups")
    if quant.get("format") != "pack-quantized" or quant.get("quant_method") != "compressed-tensors" or not isinstance(groups, Mapping):
        raise ValueError("quantization_config is not compressed-tensors pack-quantized")
    group = groups.get("group_0")
    weights = group.get("weights") if isinstance(group, Mapping) else None
    if not isinstance(weights, Mapping):
        raise ValueError("quantization_config.config_groups.group_0.weights is missing")
    _require("quantization weights", weights, {
        "dynamic": False, "group_size": 32, "num_bits": 4,
        "strategy": "group", "symmetric": True, "type": "int",
    })
    return {
        "layers": 60, "sliding_layers": 50, "full_layers": 10,
        "hidden": 5376, "intermediate": 21504, "vocabulary": 262144,
        "source_quantization": "signed symmetric INT4, group 32, no zero point",
    }


def validate_assistant_config(config: Mapping[str, object]) -> dict[str, object]:
    _require("assistant config", config, {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "tie_word_embeddings": True,
    })
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError("assistant config.text_config is missing")
    _require("assistant text_config", text, {
        "hidden_size": 1024, "intermediate_size": 8192,
        "num_hidden_layers": 4, "vocab_size": 262144,
        "tie_word_embeddings": True,
    })
    return {"layers": 4, "hidden": 1024, "intermediate": 8192, "vocabulary": 262144}


_DIRECT_SUFFIX = {
    "input_norm": "input_layernorm.weight",
    "post_attention_norm": "post_attention_layernorm.weight",
    "pre_feedforward_norm": "pre_feedforward_layernorm.weight",
    "post_feedforward_norm": "post_feedforward_layernorm.weight",
    "attention/query_norm": "self_attn.q_norm.weight",
    "attention/key_norm": "self_attn.k_norm.weight",
    "layer_scalar": "layer_scalar",
}


def target_source_names(runtime_name: str) -> tuple[str, ...]:
    if runtime_name == "text/token_embedding":
        return ("model.language_model.embed_tokens.weight",)
    if runtime_name == "text/final_norm":
        return ("model.language_model.norm.weight",)
    parts = runtime_name.split("/")
    layer = int(parts[2])
    role = "/".join(parts[3:])
    base = f"model.language_model.layers.{layer}."
    if role in _DIRECT_SUFFIX:
        return (base + _DIRECT_SUFFIX[role],)
    if role == "attention/input_projection":
        projections = ["q_proj", "k_proj"]
        if layer not in inventory.FULL_LAYERS:
            projections.append("v_proj")
        return tuple(base + f"self_attn.{name}.weight" for name in projections)
    if role == "attention/output":
        return (base + "self_attn.o_proj.weight",)
    if role == "mlp/gate_up":
        return (base + "mlp.gate_proj.weight", base + "mlp.up_proj.weight")
    if role == "mlp/down":
        return (base + "mlp.down_proj.weight",)
    raise KeyError(runtime_name)


def assistant_source_names(runtime_name: str) -> tuple[str, ...]:
    if runtime_name == "assistant/token_embedding":
        return ("model.embed_tokens.weight",)
    if runtime_name == "assistant/input_projection":
        return ("pre_projection.weight",)
    if runtime_name == "assistant/output_projection":
        return ("post_projection.weight",)
    if runtime_name == "assistant/final_norm":
        return ("model.norm.weight",)
    parts = runtime_name.split("/")
    layer = int(parts[2])
    role = "/".join(parts[3:])
    base = f"model.layers.{layer}."
    direct = {key: value for key, value in _DIRECT_SUFFIX.items() if key != "attention/key_norm"}
    if role in direct:
        return (base + direct[role],)
    if role == "attention/query":
        return (base + "self_attn.q_proj.weight",)
    if role == "attention/output":
        return (base + "self_attn.o_proj.weight",)
    if role == "mlp/gate_up":
        return (base + "mlp.gate_proj.weight", base + "mlp.up_proj.weight")
    if role == "mlp/down":
        return (base + "mlp.down_proj.weight",)
    raise KeyError(runtime_name)


def _expanded_target_sources() -> set[str]:
    names: set[str] = set()
    for spec in inventory.TARGET_TENSOR_SPECS:
        for logical in target_source_names(spec.name):
            if spec.format == inventory.Q4:
                names.update(logical + suffix for suffix in ("_packed", "_scale", "_shape"))
            else:
                names.add(logical)
    return names


def _metadata(reader: ShardReader) -> dict[str, TensorMetadata]:
    return reader.metadata(reader.names)


def preflight_target(reader: ShardReader) -> SourcePreflight:
    metadata = _metadata(reader)
    used = _expanded_target_sources()
    missing = sorted(used - set(metadata))
    if missing:
        raise ValueError(f"target checkpoint is missing {missing[0]}")
    tied = {"lm_head.weight"}
    omitted = {name for name in metadata if name.startswith("model.vision_tower.") or name.startswith("model.embed_vision.")}
    unexpected = set(metadata) - used - tied - omitted
    if unexpected:
        raise ValueError(f"unclassified target source tensor: {sorted(unexpected)[0]}")
    return SourcePreflight(len(metadata), dict(Counter(item.dtype for item in metadata.values())), tuple(sorted(used)), tuple(sorted(omitted)), tuple(sorted(tied)))


def preflight_assistant(reader: ShardReader) -> SourcePreflight:
    metadata = _metadata(reader)
    used: set[str] = set()
    for spec in inventory.ASSISTANT_TENSOR_SPECS:
        used.update(assistant_source_names(spec.name))
    missing = sorted(used - set(metadata))
    unexpected = set(metadata) - used
    if missing:
        raise ValueError(f"assistant checkpoint is missing {missing[0]}")
    if unexpected:
        raise ValueError(f"unclassified assistant source tensor: {sorted(unexpected)[0]}")
    return SourcePreflight(len(metadata), dict(Counter(item.dtype for item in metadata.values())), tuple(sorted(used)), (), ())


def decode_compressed_linear(reader: ShardReader, logical_name: str, device: torch.device) -> torch.Tensor:
    packed = reader.get(logical_name + "_packed")
    scales = reader.get(logical_name + "_scale")
    shape_word = reader.get(logical_name + "_shape")
    if packed.dtype != torch.int32 or scales.dtype != torch.bfloat16 or shape_word.dtype != torch.int64:
        raise ValueError(f"{logical_name}: invalid compressed-tensors dtypes")
    shape = tuple(int(value) for value in shape_word.tolist())
    if len(shape) != 2 or tuple(packed.shape) != (shape[0], (shape[1] + 7) // 8) or tuple(scales.shape) != (shape[0], (shape[1] + 31) // 32):
        raise ValueError(f"{logical_name}: inconsistent compressed-tensors shapes")
    words = packed.to(device=device)
    shifts = torch.arange(8, dtype=torch.int32, device=device) * 4
    codes = ((words.unsqueeze(-1) >> shifts) & 0xF).sub_(8).to(torch.int8).reshape(shape[0], -1)[:, :shape[1]]
    grouped = codes.reshape(shape[0], -1, 32).to(torch.float32)
    scale32 = scales.to(device=device, dtype=torch.float32)
    return (grouped * scale32.unsqueeze(-1)).reshape(shape)


def materialize_target(spec: TensorSpec, reader: ShardReader, device: torch.device) -> torch.Tensor:
    sources = target_source_names(spec.name)
    if spec.format == inventory.Q4:
        tensors = [decode_compressed_linear(reader, name, device) for name in sources]
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
    tensor = reader.get(sources[0])
    if tuple(tensor.shape) != spec.shape or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{sources[0]}: source signature does not match {spec.name}")
    return tensor


def materialize_assistant(spec: TensorSpec, reader: ShardReader, device: torch.device) -> torch.Tensor:
    sources = assistant_source_names(spec.name)
    tensors = [reader.get(name) for name in sources]
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError(f"{spec.name}: assistant source must be BF16")
    tensor = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}")
    return tensor.to(device=device) if spec.format == inventory.W8 else tensor


__all__ = [
    "ASSISTANT_CHECKPOINT_SHA256", "ASSISTANT_REPOSITORY", "ASSISTANT_REVISION",
    "SourcePreflight", "TARGET_CHECKPOINT_SHA256", "TARGET_REPOSITORY", "TARGET_REVISION",
    "assistant_source_names", "decode_compressed_linear", "load_json", "materialize_assistant",
    "materialize_target", "preflight_assistant", "preflight_target", "target_source_names",
    "validate_assistant_config", "validate_target_config",
]
