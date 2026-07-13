import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_common_with_stubbed_deps(monkeypatch):
    torch_mod = types.ModuleType("torch")
    dist_mod = types.ModuleType("torch.distributed")
    torch_mod.Tensor = object
    torch_mod.cat = lambda *args, **kwargs: None
    torch_mod.nn = types.SimpleNamespace(Module=object, Parameter=object)
    torch_mod.distributed = dist_mod
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, current_device=lambda: 0)

    mpu_mod = types.ModuleType("megatron.core.mpu")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0

    misc_utils_mod = types.ModuleType("slime.backends.megatron_utils.misc_utils")
    misc_utils_mod.strip_param_name_prefix = lambda name: name
    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: None
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.ParamInfo = object

    for name, module in {
        "torch": torch_mod,
        "torch.distributed": dist_mod,
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.mpu": mpu_mod,
        "megatron.core.transformer": types.ModuleType("megatron.core.transformer"),
        "megatron.core.transformer.transformer_layer": transformer_layer_mod,
        "slime": types.ModuleType("slime"),
        "slime.backends": types.ModuleType("slime.backends"),
        "slime.backends.megatron_utils": types.ModuleType("slime.backends.megatron_utils"),
        "slime.backends.megatron_utils.misc_utils": misc_utils_mod,
        "slime.utils": types.ModuleType("slime.utils"),
        "slime.utils.distributed_utils": distributed_utils_mod,
        "slime.utils.types": types_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    common_path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/update_weight/common.py"
    spec = importlib.util.spec_from_file_location("_update_weight_common_under_test", common_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_group_fused_qkv_a_sync_items_keeps_loader_pair_together(monkeypatch):
    common = _load_common_with_stubbed_deps(monkeypatch)
    items = [
        ("module.module.decoder.layers.0.self_attention.linear_kv_down_proj.weight", "kv0"),
        ("module.module.decoder.layers.0.self_attention.linear_q_down_proj.weight", "q0"),
        ("module.module.decoder.layers.0.self_attention.linear_proj.weight", "out0"),
        ("module.module.decoder.layers.1.self_attention.linear_q_down_proj.weight", "q1"),
    ]

    groups = common.group_fused_qkv_a_sync_items(items, lambda item: item[0])

    assert groups == [
        [items[0], items[1]],
        [items[2]],
        [items[3]],
    ]


@pytest.mark.unit
def test_fused_qkv_a_grouping_is_used_by_both_weight_update_paths():
    root = Path(__file__).resolve().parents[1]
    direct = (
        root
        / "slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py"
    ).read_text()
    distributed = (
        root
        / "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py"
    ).read_text()

    assert "for info_group in group_fused_qkv_a_sync_items" in direct
    assert "for param_group in group_fused_qkv_a_sync_items" in distributed


@pytest.mark.unit
def test_all_gather_param_returns_cpu_non_tp_param_without_materialization(monkeypatch):
    common = _load_common_with_stubbed_deps(monkeypatch)

    class DummyDevice:
        def __init__(self, device_type):
            self.type = device_type

    class DummyTensor:
        tensor_model_parallel = False
        parallel_mode = None

        def __init__(self, label, device_type="cpu"):
            self.label = label
            self.data = self
            self.device = DummyDevice(device_type)

        def to(self, *, device, non_blocking):
            raise AssertionError("all_gather_param should not materialize non-TP tensors to CUDA")

    common.torch.cuda = types.SimpleNamespace(is_available=lambda: True, current_device=lambda: 5)

    tensor = DummyTensor("w")
    gathered = common.all_gather_param("module.module.decoder.layers.0.self_attention.weight", tensor)

    assert gathered is tensor.data


@pytest.mark.unit
def test_all_gather_params_async_returns_direct_params_without_materialization(monkeypatch):
    common = _load_common_with_stubbed_deps(monkeypatch)

    class DummyTensor:
        tensor_model_parallel = False
        parallel_mode = None

        def __init__(self, label):
            self.label = label
            self.data = self

        def to(self, *, device, non_blocking):
            raise AssertionError("all_gather_params_async should not materialize non-TP tensors to CUDA")

    info = types.SimpleNamespace(name="module.module.decoder.layers.0.self_attention.weight")
    tensor = DummyTensor("w")

    gathered = common.all_gather_params_async([(info, tensor)])

    assert gathered == [tensor.data]
