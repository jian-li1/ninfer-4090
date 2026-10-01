#!/usr/bin/env python3
"""Compare a Phase 4 C++ parity dump with the pinned Gemma checkpoint fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


STAGES = (
    "target_embeddings",
    "target_layer0_input_normalized",
    "target_layer0_q",
    "target_layer0_k",
    "target_layer0_v",
    "target_layer0_attention_output",
    "target_layer0_attention_projection",
    "target_layer0_attention_residual",
    "target_layer0_pre_feedforward",
    "target_layer0_mlp_product",
    "target_layer0_mlp_output",
    "target_layer0_output",
    "target_layer5_input_normalized",
    "target_layer5_q",
    "target_layer5_k",
    "target_layer5_v",
    "target_layer5_attention_output",
    "target_layer5_attention_projection",
    "target_layer5_attention_residual",
    "target_layer5_pre_feedforward",
    "target_layer5_mlp_product",
    "target_layer5_mlp_output",
    "target_layer5_output",
    "target_final_hidden",
    "target_last_logits",
)


def checksum(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()


def load_dump(directory: Path, name: str, reference: np.ndarray) -> np.ndarray:
    fp32 = directory / f"{name}.f32"
    if fp32.exists():
        result = np.fromfile(fp32, dtype="<f4")
    else:
        raw = np.fromfile(directory / f"{name}.bf16", dtype="<u2")
        result = (raw.astype(np.uint32) << 16).view(np.float32)
    if name.endswith(("_q", "_k", "_v")):
        _, heads, tokens, width = reference.shape
        result = result.reshape(tokens, heads, width).transpose(1, 0, 2)[None]
    else:
        result = result.reshape(reference.shape)
    return result


def layer_and_substage(name: str) -> tuple[int | None, str, str]:
    if name.startswith("target_layer"):
        suffix = name.removeprefix("target_layer")
        digits = suffix.split("_", 1)[0]
        layer = int(digits)
        stage = suffix[len(digits) + 1 :]
        return layer, "full" if (layer + 1) % 6 == 0 else "sliding", stage
    return None, "model", name.removeprefix("target_")


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    expected = reference.astype(np.float64).ravel()
    observed = actual.astype(np.float64).ravel()
    delta = observed - expected
    expected_norm = float(np.linalg.norm(expected))
    observed_norm = float(np.linalg.norm(observed))
    denominator = max(expected_norm * observed_norm, np.finfo(np.float64).tiny)
    return {
        "reference_checksum": checksum(reference),
        "ninfer_checksum": checksum(actual),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(expected_norm, np.finfo(np.float64).tiny)),
        "cosine_similarity": float(np.dot(expected, observed) / denominator),
        "reference_slice": expected[:8].tolist(),
        "ninfer_slice": observed[:8].tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.35)
    parser.add_argument("--minimum-cosine", type=float, default=0.95)
    args = parser.parse_args()

    fixture = np.load(args.fixture)
    reports = []
    first_divergence = None
    for name in STAGES:
        reference = fixture[name]
        actual = load_dump(args.dump, name, reference)
        report = {"name": name}
        layer, layer_type, substage = layer_and_substage(name)
        report.update({"layer": layer, "layer_type": layer_type, "substage": substage})
        report.update(metrics(reference, actual))
        report["passes"] = (
            report["relative_l2"] <= args.maximum_relative_l2
            and report["cosine_similarity"] >= args.minimum_cosine
        )
        if not report["passes"] and first_divergence is None:
            first_divergence = {"layer": layer, "layer_type": layer_type, "substage": substage}
        reports.append(report)

    logits = load_dump(args.dump, "target_last_logits", fixture["target_last_logits"])
    actual_token = int(np.argmax(logits, axis=-1)[0])
    reference_token = int(fixture["target_greedy_next_token"].reshape(-1)[0])
    output = {
        "schema": "ninfer.gemma4.phase4-parity.v1",
        "fixture": str(args.fixture),
        "dump": str(args.dump),
        "thresholds": {
            "maximum_relative_l2": args.maximum_relative_l2,
            "minimum_cosine_similarity": args.minimum_cosine,
        },
        "greedy_token": actual_token,
        "reference_greedy_token": reference_token,
        "greedy_token_matches": actual_token == reference_token,
        "first_divergence": first_divergence,
        "stages": reports,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if first_divergence is None and actual_token == reference_token else 1


if __name__ == "__main__":
    raise SystemExit(main())
