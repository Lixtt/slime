from argparse import Namespace
import importlib.util
from pathlib import Path


def _load_payload_helper():
    path = (
        Path(__file__).resolve().parents[1]
        / "slime"
        / "backends"
        / "megatron_utils"
        / "update_weight"
        / "colocated_payload.py"
    )
    spec = importlib.util.spec_from_file_location("colocated_payload_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_colocated_tensor_payload_ranks_keeps_tp_for_pp_engine():
    select_colocated_tensor_payload_ranks = _load_payload_helper().select_colocated_tensor_payload_ranks
    args = Namespace(sglang_pp_size=5)

    assert select_colocated_tensor_payload_ranks(list(range(40)), args) == list(range(8))


def test_select_colocated_tensor_payload_ranks_keeps_all_ranks_without_pp():
    select_colocated_tensor_payload_ranks = _load_payload_helper().select_colocated_tensor_payload_ranks
    args = Namespace(sglang_pp_size=1)

    assert select_colocated_tensor_payload_ranks(list(range(8)), args) == list(range(8))


def test_select_colocated_tensor_payload_ranks_falls_back_on_irregular_shape():
    select_colocated_tensor_payload_ranks = _load_payload_helper().select_colocated_tensor_payload_ranks
    args = Namespace(sglang_pp_size=6)
    ranks = list(range(40))

    assert select_colocated_tensor_payload_ranks(ranks, args) == ranks


def test_get_sglang_pp_layer_ranges_matches_explicit_partition():
    helper = _load_payload_helper()

    assert helper.get_sglang_pp_layer_ranges(
        num_layers=78,
        pp_size=5,
        partition="14,16,16,16,16",
    ) == [(0, 14), (14, 30), (30, 46), (46, 62), (62, 78)]


def test_get_sglang_pp_layer_ranges_uses_env_partition(monkeypatch):
    helper = _load_payload_helper()
    monkeypatch.setenv("SGLANG_PP_LAYER_PARTITION", "14,16,16,16,16")

    assert helper.resolve_sglang_pp_layer_partition(None) == "14,16,16,16,16"
    assert helper.get_sglang_pp_layer_ranges(num_layers=78, pp_size=5) == [
        (0, 14),
        (14, 30),
        (30, 46),
        (46, 62),
        (62, 78),
    ]


def test_get_sglang_pp_layer_ranges_default_uses_last_remainder():
    helper = _load_payload_helper()

    assert helper.get_sglang_pp_layer_ranges(num_layers=78, pp_size=5, partition="") == [
        (0, 15),
        (15, 30),
        (30, 46),
        (46, 62),
        (62, 78),
    ]


def test_split_hf_named_tensors_for_sglang_pp_uses_layer_ranges():
    helper = _load_payload_helper()
    named_tensors = [
        ("model.layers.0.self_attn.q_a_proj.weight", "l0q"),
        ("model.layers.13.self_attn.kv_a_proj_with_mqa.weight", "l13kv"),
        ("model.layers.14.self_attn.q_a_proj.weight", "l14q"),
        ("model.layers.29.self_attn.kv_a_proj_with_mqa.weight", "l29kv"),
        ("model.layers.30.self_attn.q_a_proj.weight", "l30q"),
        ("model.layers.45.self_attn.kv_a_proj_with_mqa.weight", "l45kv"),
        ("model.layers.46.self_attn.q_a_proj.weight", "l46q"),
        ("model.layers.61.self_attn.kv_a_proj_with_mqa.weight", "l61kv"),
        ("model.layers.62.self_attn.q_a_proj.weight", "l62q"),
        ("model.layers.77.self_attn.kv_a_proj_with_mqa.weight", "l77kv"),
    ]

    split = helper.split_hf_named_tensors_for_sglang_pp(
        named_tensors,
        pp_size=5,
        num_layers=78,
        partition="14,16,16,16,16",
    )

    assert [[value for _name, value in stage] for stage in split] == [
        ["l0q", "l13kv"],
        ["l14q", "l29kv"],
        ["l30q", "l45kv"],
        ["l46q", "l61kv"],
        ["l62q", "l77kv"],
    ]


def test_split_hf_named_tensors_replicates_non_layer_weights():
    helper = _load_payload_helper()

    split = helper.split_hf_named_tensors_for_sglang_pp(
        [("model.norm.weight", "norm")],
        pp_size=2,
        num_layers=4,
        partition="2,2",
    )

    assert split == [[("model.norm.weight", "norm")], [("model.norm.weight", "norm")]]
