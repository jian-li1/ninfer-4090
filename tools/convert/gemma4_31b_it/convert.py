"""Convert the pinned Gemma 4 31B IT checkpoint into a text-only NInfer v2 artifact.

Canonical invocation::

    python -m tools.convert.gemma4_31b_it.convert \
      --model /path/to/gemma-4-31B-it-qat-w4a16-ct \
      --assistant /path/to/gemma-4-31B-it-assistant \
      --out /dev/shm/gemma4_31b_it.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable, Iterator, Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter, TensorObject
from tools.artifact.layouts import encode_direct, encode_row_split
from tools.convert.common.fp8_rows import ENCODER_PROFILE, iter_reader_payload, quantize_bf16_rows
from tools.convert.common.quantize import pick_device, quantize_matrix
from tools.convert.common.safetensors import ShardReader
from . import inventory, recipe


REFERENCE_FIXTURE_REVISION = "gemma4-reference-v1"
RESOURCE_FILES = {
    "frontend/tokenizer.json": "tokenizer.json",
    "frontend/tokenizer_config.json": "tokenizer_config.json",
    "frontend/chat_template.jinja": "chat_template.jinja",
    "frontend/generation_config.json": "generation_config.json",
}


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    max_error: float
    mean_error: float
    relative_l2: float
    cosine_similarity: float


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(model: Path, expected_sha256: str, label: str) -> Path:
    checkpoint = model / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError(f"{label} checkpoint must be one model.safetensors file")
    actual = sha256_file(checkpoint)
    if actual != expected_sha256:
        raise ValueError(f"{label} checkpoint SHA-256 {actual} != pinned {expected_sha256}")
    return checkpoint


def load_resources(model: Path) -> dict[str, bytes]:
    resources: dict[str, bytes] = {}
    for artifact_name, source_name in RESOURCE_FILES.items():
        path = model / source_name
        if not path.is_file():
            raise ValueError(f"missing required frontend resource: {path}")
        payload = path.read_bytes()
        if not payload:
            raise ValueError(f"frontend resource is empty: {path}")
        resources[artifact_name] = payload
    return resources


def _metrics(source: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor) -> ErrorMetrics:
    actual = source.detach().float()
    reconstructed = (codes.float() * scales.float().unsqueeze(-1)).reshape(actual.shape)
    error = reconstructed - actual
    source_norm = torch.linalg.vector_norm(actual.double())
    reconstructed_norm = torch.linalg.vector_norm(reconstructed.double())
    denominator = source_norm * reconstructed_norm
    cosine = torch.tensor(1.0, dtype=torch.float64, device=actual.device)
    if float(denominator.item()) != 0.0:
        cosine = torch.sum(actual.double() * reconstructed.double()) / denominator
    relative = torch.linalg.vector_norm(error.double()) / torch.clamp(source_norm, min=torch.finfo(torch.float64).tiny)
    return ErrorMetrics(
        max_error=float(error.abs().max().item()),
        mean_error=float(error.abs().mean().item()),
        relative_l2=float(relative.item()),
        cosine_similarity=float(cosine.item()),
    )


def _encode_grouped(weight: torch.Tensor, numeric_format: str, device: torch.device, collect_metrics: bool) -> tuple[bytes, ErrorMetrics | None]:
    quantized = quantize_matrix(weight, numeric_format, device=device)
    metrics = _metrics(weight.to(device=device), quantized.codes, quantized.scales) if collect_metrics else None
    payload = encode_row_split(quantized.codes, quantized.scales, numeric_format, tuple(weight.shape))
    return payload, metrics


def _metric_role(name: str) -> str | None:
    representatives = {
        "text/layers/0/attention/input_projection": "sliding_attention_input",
        "text/layers/0/attention/output": "sliding_attention_output",
        "text/layers/0/mlp/gate_up": "mlp_gate_up",
        "text/layers/0/mlp/down": "mlp_down",
        "text/layers/5/attention/input_projection": "full_attention_input",
        "text/layers/5/attention/output": "full_attention_output",
    }
    return representatives.get(name)


def _hashing_chunks(chunks: Iterable[bytes], digest: hashlib._Hash) -> Iterator[bytes]:
    for chunk in chunks:
        digest.update(chunk)
        yield chunk


def _object_record(obj: ArtifactObject, source_names: Sequence[str], checksum: str, transform: str) -> dict[str, object]:
    value: dict[str, object] = {
        "runtime_name": obj.name,
        "kind": obj.kind,
        "source_names": list(source_names),
        "bytes": obj.bytes,
        "sha256": checksum,
        "transform": transform,
    }
    if isinstance(obj, TensorObject):
        value.update({"shape": list(obj.shape), "format": obj.format, "layout": obj.layout})
    return value


def _summary(metrics: Mapping[str, ErrorMetrics]) -> dict[str, object]:
    records = {name: asdict(value) for name, value in metrics.items()}
    if not records:
        return {"representatives": {}, "maximum_relative_l2": None, "minimum_cosine_similarity": None}
    return {
        "representatives": records,
        "maximum_relative_l2": max(value.relative_l2 for value in metrics.values()),
        "minimum_cosine_similarity": min(value.cosine_similarity for value in metrics.values()),
    }


def convert(model_dir: str | Path, out_path: str | Path, *, assistant_dir: str | Path | None = None, device: str | torch.device = "cuda") -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    assistant = None if assistant_dir is None else Path(assistant_dir)
    output = Path(out_path)
    if output.suffix != ".ninfer":
        raise ValueError("output basename must end in .ninfer")
    target_checkpoint = _validate_checkpoint(model, recipe.TARGET_CHECKPOINT_SHA256, "target")
    assistant_checkpoint = None
    if assistant is not None:
        assistant_checkpoint = _validate_checkpoint(assistant, recipe.ASSISTANT_CHECKPOINT_SHA256, "assistant")
    config_summary = recipe.validate_target_config(recipe.load_json(model / "config.json"))
    assistant_config = None if assistant is None else recipe.validate_assistant_config(recipe.load_json(assistant / "config.json"))
    resources = load_resources(model)
    resolved_device = pick_device(device)

    with ShardReader.from_file(target_checkpoint) as reader:
        target_preflight = recipe.preflight_target(reader)
    assistant_preflight = None
    if assistant_checkpoint is not None:
        with ShardReader.from_file(assistant_checkpoint) as reader:
            assistant_preflight = recipe.preflight_assistant(reader)

    specs = inventory.object_specs(resources, assistant is not None)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    errors: dict[str, ErrorMetrics] = {}

    with ArtifactWriter(output, ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID), specs) as writer:
        by_name = {obj.name: obj for obj in writer.objects}
        for name in inventory.RESOURCE_NAMES:
            digest = hashlib.sha256(resources[name]).hexdigest()
            writer.write(name, resources[name])
            records.append(_object_record(by_name[name], (RESOURCE_FILES[name],), digest, "verbatim text resource"))

        with ShardReader.from_file(target_checkpoint) as reader:
            for index, spec in enumerate(inventory.TARGET_TENSOR_SPECS, 1):
                source_names = recipe.target_source_names(spec.name)
                digest = hashlib.sha256()
                if spec.name == "text/token_embedding":
                    chunks = iter_reader_payload(reader, source_names[0], spec.shape)
                    writer.write(spec.name, _hashing_chunks(chunks, digest))
                    transform = f"row-wise FP8 ({ENCODER_PROFILE}); physically tied output head"
                    sample = reader.get(source_names[0])[:256]
                    words = quantize_bf16_rows(sample)
                    source = sample.float()
                    reconstructed = words.codes.view(torch.float8_e4m3fn).float() * words.scales.float().unsqueeze(1)
                    diff = reconstructed - source
                    denom = torch.linalg.vector_norm(source.double())
                    errors["token_embedding_first_256_rows"] = ErrorMetrics(
                        float(diff.abs().max()), float(diff.abs().mean()),
                        float((torch.linalg.vector_norm(diff.double()) / denom).item()),
                        float(torch.nn.functional.cosine_similarity(source.flatten().double(), reconstructed.flatten().double(), dim=0).item()),
                    )
                else:
                    tensor = recipe.materialize_target(spec, reader, resolved_device)
                    if spec.format == inventory.BF16:
                        payload = encode_direct(tensor.cpu(), inventory.BF16)
                        metric = None
                        transform = "verbatim BF16"
                    else:
                        role = _metric_role(spec.name)
                        payload, metric = _encode_grouped(tensor, spec.format, resolved_device, role is not None)
                        transform = "decode source INT4 group-32; fuse rows where applicable; requantize Q4G64_F16S"
                        if role is not None and metric is not None:
                            errors[role] = metric
                    digest.update(payload)
                    writer.write(spec.name, payload)
                    del tensor, payload
                checksum = digest.hexdigest()
                records.append(_object_record(by_name[spec.name], source_names, checksum, transform))
                print(f"[target {index}/{len(inventory.TARGET_TENSOR_SPECS)}] {spec.name}", flush=True)

        if assistant is not None and assistant_checkpoint is not None:
            with ShardReader.from_file(assistant_checkpoint) as reader:
                for index, spec in enumerate(inventory.ASSISTANT_TENSOR_SPECS, 1):
                    source_names = recipe.assistant_source_names(spec.name)
                    digest = hashlib.sha256()
                    if spec.name == "assistant/token_embedding":
                        writer.write(spec.name, _hashing_chunks(iter_reader_payload(reader, source_names[0], spec.shape), digest))
                        transform = f"row-wise FP8 ({ENCODER_PROFILE}); physically tied assistant output head"
                    else:
                        tensor = recipe.materialize_assistant(spec, reader, resolved_device)
                        if spec.format == inventory.BF16:
                            payload = encode_direct(tensor.cpu(), inventory.BF16)
                            transform = "verbatim BF16"
                        else:
                            payload, _ = _encode_grouped(tensor, spec.format, resolved_device, False)
                            transform = "fuse rows where applicable; quantize W8G32_F16S"
                        digest.update(payload)
                        writer.write(spec.name, payload)
                        del tensor, payload
                    checksum = digest.hexdigest()
                    records.append(_object_record(by_name[spec.name], source_names, checksum, transform))
                    print(f"[assistant {index}/{len(inventory.ASSISTANT_TENSOR_SPECS)}] {spec.name}", flush=True)

    artifact_sha = sha256_file(output)
    report = {
        "schema": "ninfer-conversion-report-v2",
        "target_identity": f"{inventory.MODEL_ID}/{inventory.WEIGHTS_ID}",
        "source_model": recipe.TARGET_REPOSITORY,
        "source_revision": recipe.TARGET_REVISION,
        "source_checkpoint_sha256": recipe.TARGET_CHECKPOINT_SHA256,
        "assistant_model": recipe.ASSISTANT_REPOSITORY if assistant is not None else None,
        "assistant_revision": recipe.ASSISTANT_REVISION if assistant is not None else None,
        "assistant_checkpoint_sha256": recipe.ASSISTANT_CHECKPOINT_SHA256 if assistant is not None else None,
        "artifact_container_version": 2,
        "artifact_sha256": artifact_sha,
        "artifact_bytes": output.stat().st_size,
        "tensor_count": len(inventory.TARGET_TENSOR_SPECS) + (len(inventory.ASSISTANT_TENSOR_SPECS) if assistant is not None else 0),
        "resident_weight_bytes_estimate": inventory.resident_weight_bytes(False),
        "resident_weight_gib_estimate": inventory.resident_weight_bytes(False) / (1024 ** 3),
        "assistant_packaged_bytes": 0 if assistant is None else inventory.resident_weight_bytes(True) - inventory.resident_weight_bytes(False),
        "runtime_transforms": [
            "source signed symmetric INT4 group-32 -> FP32 -> Q4G64_F16S",
            f"tied token embedding/head -> {ENCODER_PROFILE}",
            "Q/K/V and gate/up row fusion",
            "learned per-layer BF16 scalars retained explicitly",
        ],
        "quantization_error_summary": _summary(errors),
        "omitted_multimodal_tensors": list(target_preflight.omitted_multimodal_tensors),
        "omitted_tied_alias_tensors": list(target_preflight.tied_alias_tensors),
        "reference_fixture_revision": REFERENCE_FIXTURE_REVISION,
        "config": config_summary,
        "assistant_config": assistant_config,
        "source_preflight": asdict(target_preflight),
        "assistant_preflight": None if assistant_preflight is None else asdict(assistant_preflight),
        "objects": records,
        "arguments": {"model": str(model.resolve()), "assistant": None if assistant is None else str(assistant.resolve()), "out": str(output.resolve()), "device": str(device)},
        "resolved_device": str(resolved_device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {output.stat().st_size} bytes sha256={artifact_sha}; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--assistant", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, assistant_dir=args.assistant, device=args.device)


if __name__ == "__main__":
    main()
