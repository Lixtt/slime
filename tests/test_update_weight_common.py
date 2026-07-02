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
def test_all_gather_object_for_group_via_gloo_filters_target_group(monkeypatch):
    common = _load_common_with_stubbed_deps(monkeypatch)
    target_group = object()
    gloo_group = object()
    calls = {}

    common.get_gloo_group = lambda: gloo_group
    common.dist.get_world_size = lambda group=None: 4

    def fake_get_process_group_ranks(group):
        assert group is target_group
        return [1, 3]

    def fake_all_gather_object(*, obj, object_list, group):
        calls["obj"] = obj
        calls["group"] = group
        object_list[:] = [
            (0, "rank0"),
            (1, "rank1"),
            (2, "rank2"),
            (3, "rank3"),
        ]

    common.dist.get_process_group_ranks = fake_get_process_group_ranks
    common.dist.all_gather_object = fake_all_gather_object

    gathered = common.all_gather_object_for_group_via_gloo(("local", "payload"), target_group)

    assert calls == {"obj": ("local", "payload"), "group": gloo_group}
    assert gathered == [(1, "rank1"), (3, "rank3")]


@pytest.mark.unit
def test_all_gather_object_for_group_via_gloo_rejects_non_rank_payload(monkeypatch):
    common = _load_common_with_stubbed_deps(monkeypatch)
    target_group = object()
    gloo_group = object()

    common.get_gloo_group = lambda: gloo_group
    common.dist.get_world_size = lambda group=None: 2
    common.dist.get_process_group_ranks = lambda group: [0, 1]

    def fake_all_gather_object(*, obj, object_list, group):
        object_list[:] = [
            ["name.from.bad.caller"],
            ["another.bad.payload"],
        ]

    common.dist.all_gather_object = fake_all_gather_object

    with pytest.raises(ValueError, match="expects every gathered object"):
        common.all_gather_object_for_group_via_gloo(["name.from.bad.caller"], target_group)


@pytest.mark.unit
def test_get_gloo_group_for_process_group_creates_reported_subgroups_in_order(monkeypatch):
    common = _load_common_with_stubbed_deps(monkeypatch)
    target_group = object()
    world_gloo_group = object()
    created = []

    common.get_gloo_group = lambda: world_gloo_group
    common.dist.get_world_size = lambda group=None: 4
    common.dist.get_process_group_ranks = lambda group: [2, 7]

    def fake_all_gather_object(object_list, obj, group):
        assert isinstance(object_list, list)
        assert obj == (2, 7)
        assert group is world_gloo_group
        object_list[:] = [
            (5, 10),
            (0, 3),
            (2, 7),
            (0, 3),
        ]

    def fake_new_group(*, ranks, backend):
        created.append((tuple(ranks), backend))
        return f"group:{','.join(map(str, ranks))}"

    common.dist.all_gather_object = fake_all_gather_object
    common.dist.new_group = fake_new_group

    group = common.get_gloo_group_for_process_group(target_group)
    group_again = common.get_gloo_group_for_process_group(target_group)

    assert group == "group:2,7"
    assert group_again == group
    assert created == [
        ((0, 3), "gloo"),
        ((2, 7), "gloo"),
        ((5, 10), "gloo"),
    ]


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
def test_named_params_and_buffers_trainable_only_filters_frozen_params_and_buffers(monkeypatch):
    common = _load_common_with_stubbed_deps(monkeypatch)

    class DummyTensor:
        def __init__(self, requires_grad):
            self.requires_grad = requires_grad

    class DummyModule:
        def named_parameters(self):
            return [
                ("train.weight", DummyTensor(True)),
                ("frozen.weight", DummyTensor(False)),
            ]

        def named_buffers(self):
            return [
                ("expert_bias", DummyTensor(False)),
            ]

    items = list(
        common.named_params_and_buffers(
            args=object(),
            model=[DummyModule()],
            convert_to_global_name=False,
            trainable_only=True,
        )
    )

    assert [name for name, _ in items] == ["vp_stages.0.train.weight"]


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
