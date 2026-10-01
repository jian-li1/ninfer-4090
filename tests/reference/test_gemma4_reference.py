from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "gemma4_31b_it"
SEMANTIC_PATH = ROOT / "tools" / "reference" / "gemma4_31b_it" / "semantic.py"


def load_semantic_module():
    spec = importlib.util.spec_from_file_location("gemma4_reference_semantic", SEMANTIC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def test_semantic_fixtures_reproduce_exactly():
    module = load_semantic_module()
    actual = {
        name: value.detach().cpu().numpy()
        for name, value in module.generate_semantic_fixtures().items()
    }
    with np.load(FIXTURES / "semantic.npz") as expected:
        assert set(actual) == set(expected.files)
        for name, value in actual.items():
            np.testing.assert_array_equal(value, expected[name], err_msg=name)


def test_fixture_metadata_hashes_every_array():
    metadata = json.loads((FIXTURES / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "ninfer.gemma4.reference-fixtures.v1"
    with np.load(FIXTURES / "semantic.npz") as fixture:
        for name in fixture.files:
            record = metadata["arrays"][name]
            assert record["shape"] == list(fixture[name].shape)
            assert record["dtype"] == str(fixture[name].dtype)
            assert record["sha256"] == array_hash(fixture[name])


def test_global_rope_rotates_only_first_quarter():
    with np.load(FIXTURES / "semantic.npz") as fixture:
        inverse_frequency = fixture["global_rope_inverse_frequency"]
        assert np.count_nonzero(inverse_frequency) == 64
        np.testing.assert_array_equal(inverse_frequency[64:], np.zeros(192, dtype=np.float32))
        np.testing.assert_array_equal(
            fixture["global_rope_output"][:, 64:256], fixture["global_rope_input"][:, 64:256]
        )
        np.testing.assert_array_equal(
            fixture["global_rope_output"][:, 320:], fixture["global_rope_input"][:, 320:]
        )


def test_full_attention_uses_projected_key_as_value_source():
    with np.load(FIXTURES / "semantic.npz") as fixture:
        np.testing.assert_array_equal(fixture["full_k_projected"], fixture["full_v_projected"])
        assert not np.array_equal(fixture["full_key"], fixture["full_value"])


def test_output_contracts_are_present():
    with np.load(FIXTURES / "semantic.npz") as fixture:
        required = {
            "embeddings",
            "rmsnorm_scaled",
            "rmsnorm_unscaled",
            "local_rope_output",
            "global_rope_output",
            "sliding_output",
            "full_output",
            "final_hidden",
            "logits",
            "greedy_next_token",
            "mtp_feedback",
            "mtp_logits",
            "mtp_draft_token",
        }
        assert required <= set(fixture.files)


def test_official_tokenizer_fixture():
    fixture = json.loads((FIXTURES / "tokenizer.json").read_text(encoding="utf-8"))
    assert fixture["source_sha256"] == (
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    )
    assert [case["ids"] for case in fixture["cases"]] == [
        [9259, 236764, 147224, 236743, 236812, 236888],
        [818, 3823, 8864, 37423],
        [55107, 236787, 8369, 8369, 8369, 236761],
    ]


def test_assistant_shared_kv_mapping_fixture():
    mapping = json.loads((FIXTURES / "assistant_mapping.json").read_text(encoding="utf-8"))
    assert mapping["position_semantics"] == {
        "assistant_appends_kv": False,
        "fixed_within_draft_loop": True,
        "value": "input_length - 1",
    }
    assert [
        (layer["assistant_layer"], layer["attention_type"], layer["target_layer"])
        for layer in mapping["layers"]
    ] == [
        (0, "sliding_attention", 58),
        (1, "sliding_attention", 58),
        (2, "sliding_attention", 58),
        (3, "full_attention", 59),
    ]
