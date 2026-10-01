#!/usr/bin/env python3
"""Generate deterministic Gemma 4 31B text semantic fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import torch

from semantic import generate_semantic_fixtures


TARGET_REPOSITORY = "google/gemma-4-31B-it-qat-w4a16-ct"
TARGET_REVISION = "52f3f65bc7a02d555763bc923bd1d9094898219d"
ASSISTANT_REPOSITORY = "google/gemma-4-31B-it-assistant"
ASSISTANT_REVISION = "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"
TRANSFORMERS_REVISION = "d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a"

RESOURCE_HASHES = {
    "target/config.json": "b100d85e571c25b688e82919d819253063a1defb0e6cee3b9b1c1fbad99bd0c2",
    "target/generation_config.json": "70110aa886fa7ee25ce47346d102dd29ca5f967500490b4362fc72068e1db70a",
    "target/tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "target/tokenizer_config.json": "b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b",
    "target/chat_template.jinja": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    "target/processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    "assistant/config.json": "d09487c1083a334a97545f204dd41ab1a40b0438bdead19558f0e1357e26d66a",
    "assistant/generation_config.json": "8e58004dc0e2407b63410b190bb8470efbdcfeb71533f1770e09c20abe193a6f",
    "assistant/tokenizer.json": "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c",
    "assistant/tokenizer_config.json": "089594a3924fcfd4cb1c596a7906fbf476193519e5198f780912eed02b177e42",
}

TOKENIZER_PROMPTS = (
    "Hello, Gemma 4!",
    "The quick brown fox",
    "Repeat: token token token.",
)

ASSISTANT_SHARED_KV_MAPPING = {
    "schema": "ninfer.gemma4.assistant-shared-kv.v1",
    "position_semantics": {
        "value": "input_length - 1",
        "fixed_within_draft_loop": True,
        "assistant_appends_kv": False,
    },
    "layers": [
        {
            "assistant_layer": layer,
            "attention_type": "sliding_attention" if layer < 3 else "full_attention",
            "target_layer": 58 if layer < 3 else 59,
            "kv_heads": 16 if layer < 3 else 4,
            "head_dim": 256 if layer < 3 else 512,
            "mask": (
                "bidirectional q_len=1 over the retained target sliding view; "
                "window direction reversed to target-history order"
                if layer < 3
                else "bidirectional q_len=1 over the complete target full-attention view"
            ),
        }
        for layer in range(4)
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def verify_resources(source_dir: Path) -> None:
    failures = []
    for relative, expected in RESOURCE_HASHES.items():
        path = source_dir / relative
        if not path.is_file():
            failures.append(f"missing {path}")
        elif (actual := sha256(path)) != expected:
            failures.append(f"hash mismatch for {path}: {actual} != {expected}")
    if failures:
        raise RuntimeError("pinned resource verification failed:\n" + "\n".join(failures))


def generate_tokenizer_fixture(source_dir: Path) -> dict[str, object]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("tokenizers is required when --source-dir is supplied") from error

    tokenizer = Tokenizer.from_file(str(source_dir / "target" / "tokenizer.json"))
    return {
        "source_sha256": RESOURCE_HASHES["target/tokenizer.json"],
        "cases": [
            {
                "text": text,
                "ids": tokenizer.encode(text).ids,
                "tokens": tokenizer.encode(text).tokens,
            }
            for text in TOKENIZER_PROMPTS
        ],
    }


def generate(output_dir: Path, source_dir: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    tensors = generate_semantic_fixtures()
    arrays = {name: value.detach().cpu().numpy() for name, value in sorted(tensors.items())}
    np.savez(output_dir / "semantic.npz", **arrays)
    (output_dir / "assistant_mapping.json").write_text(
        json.dumps(ASSISTANT_SHARED_KV_MAPPING, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tokenizer_fixture = None
    if source_dir is not None:
        verify_resources(source_dir)
        tokenizer_fixture = generate_tokenizer_fixture(source_dir)
        (output_dir / "tokenizer.json").write_text(
            json.dumps(tokenizer_fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    metadata = {
        "schema": "ninfer.gemma4.reference-fixtures.v1",
        "scope": "text-only",
        "determinism": "CPU BF16 boundaries, torch deterministic algorithms enabled",
        "target": {"repository": TARGET_REPOSITORY, "revision": TARGET_REVISION},
        "assistant": {"repository": ASSISTANT_REPOSITORY, "revision": ASSISTANT_REVISION},
        "assistant_shared_kv_mapping": ASSISTANT_SHARED_KV_MAPPING,
        "transformers_revision": TRANSFORMERS_REVISION,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "resource_sha256": RESOURCE_HASHES,
        "tokenizer_fixture_present": tokenizer_fixture is not None,
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": array_hash(value),
            }
            for name, value in arrays.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="directory containing verified target/ and assistant/ pinned resources",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args(sys.argv[1:])
    generate(arguments.output_dir, arguments.source_dir)
