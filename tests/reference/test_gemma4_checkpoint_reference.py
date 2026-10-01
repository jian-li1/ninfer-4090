from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "gemma4_31b_it"
TOOLS = ROOT / "tools" / "reference" / "gemma4_31b_it"


def array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def test_checkpoint_fixture_metadata_and_contract():
    metadata_path = FIXTURES / "checkpoint_metadata.json"
    fixture_path = FIXTURES / "checkpoint.npz"
    if not metadata_path.is_file() or not fixture_path.is_file():
        pytest.skip("full-checkpoint fixture has not been generated")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema"] == "ninfer.gemma4.checkpoint-fixtures.v1"
    assert metadata["token_ids"] == [9259, 236764, 147224, 236743, 236812, 236888]
    with np.load(fixture_path) as fixture:
        required = {
            "target_embeddings",
            "target_layer0_output",
            "target_layer5_output",
            "target_final_hidden",
            "target_last_logits",
            "target_greedy_next_token",
            "target_shared_sliding_k",
            "target_shared_sliding_v",
            "target_shared_full_k",
            "target_shared_full_v",
            "assistant_position_id",
            "assistant_preprojection",
            "assistant_body_output",
            "assistant_feedback",
            "assistant_logits",
            "assistant_draft_token",
            "assistant_runtime_draft_tokens",
            "assistant_runtime_feedback",
            "assistant_runtime_top32_ids",
            "assistant_runtime_top32_logits",
        }
        assert required <= set(fixture.files)
        for name in fixture.files:
            assert metadata["arrays"][name]["sha256"] == array_hash(fixture[name])


def test_assistant_reference_matches_pinned_transformers():
    source = os.environ.get("NINFER_GEMMA4_SOURCE_DIR")
    if not source:
        pytest.skip("set NINFER_GEMMA4_SOURCE_DIR for source-gated parity")
    source_dir = Path(source)
    if not (source_dir / "assistant" / "model.safetensors").is_file():
        pytest.skip("pinned assistant checkpoint is absent")

    sys.path.insert(0, str(TOOLS))
    try:
        from generate_checkpoint import run_assistant, run_assistant_draft_loop
        from transformers import Gemma4AssistantForCausalLM
    finally:
        sys.path.pop(0)

    torch.manual_seed(0)
    target = {
        "target_token_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "target_embeddings": torch.randn(1, 2, 5376, dtype=torch.bfloat16),
        "target_final_hidden": torch.randn(1, 2, 5376, dtype=torch.bfloat16),
    }
    shared = {
        "sliding_attention": (
            torch.randn(1, 16, 2, 256, dtype=torch.bfloat16),
            torch.randn(1, 16, 2, 256, dtype=torch.bfloat16),
        ),
        "full_attention": (
            torch.randn(1, 4, 2, 512, dtype=torch.bfloat16),
            torch.randn(1, 4, 2, 512, dtype=torch.bfloat16),
        ),
    }
    manual = run_assistant(source_dir / "assistant" / "model.safetensors", target, shared)
    model = Gemma4AssistantForCausalLM.from_pretrained(
        source_dir / "assistant", dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    concatenated = torch.cat(
        (target["target_embeddings"][:, -1, :], target["target_final_hidden"][:, -1, :]), dim=-1
    )[:, None, :]
    with torch.inference_mode():
        expected = model(
            inputs_embeds=concatenated,
            position_ids=torch.tensor([[5]], dtype=torch.int64),
            shared_kv_states=shared,
        )
    torch.testing.assert_close(
        expected.last_hidden_state, manual["assistant_feedback"], rtol=0, atol=0
    )
    torch.testing.assert_close(expected.logits, manual["assistant_logits"], rtol=0, atol=0)

    with np.load(FIXTURES / "checkpoint.npz") as fixture:
        target_result = {
            "target_embedding_scale": torch.from_numpy(
                fixture["target_embedding_scale"].copy()
            ).to(torch.bfloat16),
            "target_greedy_next_token": torch.from_numpy(
                fixture["target_greedy_next_token"].copy()
            ),
            "target_final_hidden": torch.from_numpy(
                fixture["target_final_hidden"].copy()
            ).to(torch.bfloat16),
        }
        runtime_shared = {
            "sliding_attention": (
                torch.from_numpy(fixture["target_shared_sliding_k"].copy()).to(torch.bfloat16),
                torch.from_numpy(fixture["target_shared_sliding_v"].copy()).to(torch.bfloat16),
            ),
            "full_attention": (
                torch.from_numpy(fixture["target_shared_full_k"].copy()).to(torch.bfloat16),
                torch.from_numpy(fixture["target_shared_full_v"].copy()).to(torch.bfloat16),
            ),
        }
    manual_loop = run_assistant_draft_loop(
        source_dir / "assistant" / "model.safetensors",
        source_dir / "target" / "model.safetensors",
        target_result,
        runtime_shared,
    )
    with safe_open(
        source_dir / "target" / "model.safetensors", framework="pt", device="cpu"
    ) as target:
        embedding_weight = target.get_tensor("model.language_model.embed_tokens.weight")
    last_token = target_result["target_greedy_next_token"]
    last_hidden = target_result["target_final_hidden"][:, -1:, :]
    expected_tokens = []
    expected_feedback = []
    with torch.inference_mode():
        for _ in range(6):
            target_embedding = (
                embedding_weight[last_token] * target_result["target_embedding_scale"]
            )
            output = model(
                inputs_embeds=torch.cat((target_embedding, last_hidden), dim=-1),
                position_ids=torch.tensor([[6]], dtype=torch.int64),
                shared_kv_states=runtime_shared,
            )
            last_token = output.logits.argmax(dim=-1)
            last_hidden = output.last_hidden_state
            expected_tokens.append(last_token)
            expected_feedback.append(last_hidden)
    torch.testing.assert_close(
        manual_loop["assistant_runtime_draft_tokens"],
        torch.cat(expected_tokens, dim=1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        manual_loop["assistant_runtime_feedback"],
        torch.cat(expected_feedback, dim=1),
        rtol=0,
        atol=0,
    )
