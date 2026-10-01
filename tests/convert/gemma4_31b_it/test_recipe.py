from __future__ import annotations

import json

from safetensors.torch import save_file
import torch

from tools.convert.common.safetensors import ShardReader
from tools.convert.gemma4_31b_it import recipe


def _pack_signed_nibbles(codes: torch.Tensor) -> torch.Tensor:
    unsigned = (codes.to(torch.int32) + 8) & 0xF
    words = torch.zeros((codes.shape[0], codes.shape[1] // 8), dtype=torch.int32)
    for lane in range(8):
        words |= unsigned[:, lane::8] << (4 * lane)
    return words


def test_compressed_tensors_unpack_order_and_scale_semantics(tmp_path) -> None:
    logical = "linear.weight"
    codes = torch.arange(-8, 8, dtype=torch.int8).repeat(2, 4)
    scales = torch.tensor([[0.5, 0.25], [2.0, 1.0]], dtype=torch.bfloat16)
    save_file(
        {
            logical + "_packed": _pack_signed_nibbles(codes),
            logical + "_scale": scales,
            logical + "_shape": torch.tensor([2, 64], dtype=torch.int64),
        },
        tmp_path / "model.safetensors",
    )
    with ShardReader.from_file(tmp_path / "model.safetensors") as reader:
        decoded = recipe.decode_compressed_linear(reader, logical, torch.device("cpu"))
    expected = codes.reshape(2, 2, 32).float() * scales.float().unsqueeze(-1)
    assert torch.equal(decoded, expected.reshape(2, 64))


def test_runtime_source_mapping_omits_full_attention_value_projection() -> None:
    sliding = recipe.target_source_names("text/layers/0/attention/input_projection")
    full = recipe.target_source_names("text/layers/5/attention/input_projection")
    assert [name.rsplit(".", 2)[-2] for name in sliding] == ["q_proj", "k_proj", "v_proj"]
    assert [name.rsplit(".", 2)[-2] for name in full] == ["q_proj", "k_proj"]
