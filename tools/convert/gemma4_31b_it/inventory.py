"""Closed text-only persistent inventory for Gemma 4 31B IT."""

from __future__ import annotations

from typing import Mapping

from tools.artifact.container import ResourceSpec, TensorSpec, plan_objects


MODEL_ID = "gemma4-31b-it"
WEIGHTS_ID = "groupwise-int"
LAYERS = 60
FULL_LAYERS = tuple(range(5, LAYERS, 6))
SLIDING_LAYERS = tuple(layer for layer in range(LAYERS) if layer not in FULL_LAYERS)
RESOURCE_NAMES = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",
    "frontend/generation_config.json",
)

BF16 = "BF16"
Q4 = "Q4G64_F16S"
W8 = "W8G32_F16S"
FP8_ROW = "FP8_E4M3FN_ROW_BF16S"
CONTIGUOUS = "contiguous-le-v1"
ROW_SPLIT = "row-split-k128-v1"
ROW_SCALE = "row-scale-v1"


def tensor(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    layout = CONTIGUOUS if numeric_format == BF16 else ROW_SCALE if numeric_format == FP8_ROW else ROW_SPLIT
    return TensorSpec(name, shape, numeric_format, layout)


def _target_specs() -> tuple[TensorSpec, ...]:
    specs = [tensor("text/token_embedding", (262144, 5376), FP8_ROW)]
    for layer in range(LAYERS):
        prefix = f"text/layers/{layer}/"
        full = layer in FULL_LAYERS
        input_rows = 18432 if full else 16384
        output_columns = 16384 if full else 8192
        norm_width = 512 if full else 256
        specs.extend(
            (
                tensor(prefix + "input_norm", (5376,), BF16),
                tensor(prefix + "attention/input_projection", (input_rows, 5376), Q4),
                tensor(prefix + "attention/query_norm", (norm_width,), BF16),
                tensor(prefix + "attention/key_norm", (norm_width,), BF16),
                tensor(prefix + "attention/output", (5376, output_columns), Q4),
                tensor(prefix + "post_attention_norm", (5376,), BF16),
                tensor(prefix + "pre_feedforward_norm", (5376,), BF16),
                tensor(prefix + "mlp/gate_up", (43008, 5376), Q4),
                tensor(prefix + "mlp/down", (5376, 21504), Q4),
                tensor(prefix + "post_feedforward_norm", (5376,), BF16),
                tensor(prefix + "layer_scalar", (1,), BF16),
            )
        )
    specs.append(tensor("text/final_norm", (5376,), BF16))
    return tuple(specs)


def _assistant_specs() -> tuple[TensorSpec, ...]:
    specs = [
        tensor("assistant/token_embedding", (262144, 1024), FP8_ROW),
        tensor("assistant/input_projection", (1024, 10752), W8),
    ]
    for layer in range(4):
        prefix = f"assistant/layers/{layer}/"
        full = layer == 3
        rows = 16384 if full else 8192
        norm_width = 512 if full else 256
        specs.extend(
            (
                tensor(prefix + "input_norm", (1024,), BF16),
                tensor(prefix + "attention/query", (rows, 1024), W8),
                tensor(prefix + "attention/query_norm", (norm_width,), BF16),
                tensor(prefix + "attention/output", (1024, rows), W8),
                tensor(prefix + "post_attention_norm", (1024,), BF16),
                tensor(prefix + "pre_feedforward_norm", (1024,), BF16),
                tensor(prefix + "mlp/gate_up", (16384, 1024), W8),
                tensor(prefix + "mlp/down", (1024, 8192), W8),
                tensor(prefix + "post_feedforward_norm", (1024,), BF16),
                tensor(prefix + "layer_scalar", (1,), BF16),
            )
        )
    specs.extend(
        (
            tensor("assistant/final_norm", (1024,), BF16),
            tensor("assistant/output_projection", (5376, 1024), W8),
        )
    )
    return tuple(specs)


TARGET_TENSOR_SPECS = _target_specs()
ASSISTANT_TENSOR_SPECS = _assistant_specs()


def object_specs(resources: Mapping[str, bytes], include_assistant: bool) -> tuple[ResourceSpec | TensorSpec, ...]:
    missing = [name for name in RESOURCE_NAMES if name not in resources]
    if missing:
        raise ValueError(f"missing frontend resources: {', '.join(missing)}")
    resource_specs = tuple(ResourceSpec(name, "raw-bytes-v1", len(resources[name])) for name in RESOURCE_NAMES)
    tensors = TARGET_TENSOR_SPECS + (ASSISTANT_TENSOR_SPECS if include_assistant else ())
    return resource_specs + tensors


def resident_weight_bytes(include_assistant: bool = False) -> int:
    specs = TARGET_TENSOR_SPECS + (ASSISTANT_TENSOR_SPECS if include_assistant else ())
    objects = plan_objects(specs)
    return objects[-1].offset + objects[-1].bytes


__all__ = [
    "ASSISTANT_TENSOR_SPECS", "FULL_LAYERS", "LAYERS", "MODEL_ID", "RESOURCE_NAMES",
    "SLIDING_LAYERS", "TARGET_TENSOR_SPECS", "WEIGHTS_ID", "object_specs", "resident_weight_bytes",
]
