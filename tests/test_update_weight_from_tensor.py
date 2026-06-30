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
    return module.select_colocated_tensor_payload_ranks


def test_select_colocated_tensor_payload_ranks_keeps_tp_for_pp_engine():
    select_colocated_tensor_payload_ranks = _load_payload_helper()
    args = Namespace(sglang_pp_size=5)

    assert select_colocated_tensor_payload_ranks(list(range(40)), args) == list(range(8))


def test_select_colocated_tensor_payload_ranks_keeps_all_ranks_without_pp():
    select_colocated_tensor_payload_ranks = _load_payload_helper()
    args = Namespace(sglang_pp_size=1)

    assert select_colocated_tensor_payload_ranks(list(range(8)), args) == list(range(8))


def test_select_colocated_tensor_payload_ranks_falls_back_on_irregular_shape():
    select_colocated_tensor_payload_ranks = _load_payload_helper()
    args = Namespace(sglang_pp_size=6)
    ranks = list(range(40))

    assert select_colocated_tensor_payload_ranks(ranks, args) == ranks
