from tools.convert.gemma4_31b_it import inventory


def test_text_only_inventory_and_resident_checkpoint() -> None:
    assert len(inventory.TARGET_TENSOR_SPECS) == 662
    assert len(inventory.ASSISTANT_TENSOR_SPECS) == 44
    assert inventory.FULL_LAYERS == tuple(range(5, 60, 6))
    assert inventory.resident_weight_bytes(False) == 16_971_062_784
    assert inventory.resident_weight_bytes(True) == 17_453_691_904
    assert inventory.resident_weight_bytes(False) < int(17.5 * 1024**3)
    names = {spec.name for spec in inventory.TARGET_TENSOR_SPECS}
    assert "text/output_head" not in names
    assert not any("vision" in name or "audio" in name or "video" in name for name in names)


def test_attention_and_fusion_signatures_are_exact() -> None:
    by_name = {spec.name: spec for spec in inventory.TARGET_TENSOR_SPECS}
    assert by_name["text/layers/0/attention/input_projection"].shape == (16384, 5376)
    assert by_name["text/layers/5/attention/input_projection"].shape == (18432, 5376)
    assert by_name["text/layers/0/attention/output"].shape == (5376, 8192)
    assert by_name["text/layers/5/attention/output"].shape == (5376, 16384)
    assert by_name["text/layers/0/mlp/gate_up"].shape == (43008, 5376)
