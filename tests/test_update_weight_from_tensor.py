import base64
import pickle
import sys
import types
from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest
import torch


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


def _load_update_weight_module_with_stubs(monkeypatch):
    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ray", module("ray", ObjectRef=object))
    monkeypatch.setitem(sys.modules, "ray.actor", module("ray.actor", ActorHandle=object))
    monkeypatch.setitem(sys.modules, "megatron", module("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", module("megatron.core", mpu=types.SimpleNamespace()))
    monkeypatch.setitem(
        sys.modules,
        "slime.utils.distributed_utils",
        module("slime.utils.distributed_utils", get_gloo_group=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.sglang",
        module(
            "slime.backends.megatron_utils.sglang",
            FlattenedTensorBucket=None,
            MultiprocessingSerializer=types.SimpleNamespace(serialize=lambda obj, output_str=False: b""),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.colocated_payload",
        module(
            "slime.backends.megatron_utils.update_weight.colocated_payload",
            get_sglang_pp_layer_ranges=lambda *args, **kwargs: [],
            resolve_sglang_pp_layer_partition=lambda partition=None: partition,
            select_colocated_tensor_payload_ranks=lambda ranks, args: ranks,
            split_hf_named_tensors_for_sglang_pp=lambda tensors, **kwargs: [tensors],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_base",
        module("slime.backends.megatron_utils.update_weight.hf_weight_iterator_base", HfWeightIteratorBase=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        module(
            "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
            connect_rollout_engines_from_distributed=lambda *args, **kwargs: None,
            disconnect_rollout_engines_from_distributed=lambda *args, **kwargs: None,
            post_process_weights=lambda *args, **kwargs: None,
            update_weights_from_distributed=lambda *args, **kwargs: [],
        ),
    )
    path = (
        Path(__file__).resolve().parents[1]
        / "slime"
        / "backends"
        / "megatron_utils"
        / "update_weight"
        / "update_weight_from_tensor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "slime.backends.megatron_utils.update_weight.update_weight_from_tensor_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_colocated_tensor_payload_ranks_keeps_tp_for_pp_engine():
    select_colocated_tensor_payload_ranks = _load_payload_helper().select_colocated_tensor_payload_ranks
    args = Namespace(sglang_pp_size=5)

    assert select_colocated_tensor_payload_ranks(list(range(40)), args) == list(range(8))


def test_serialize_flattened_bucket_falls_back_to_cpu_payload(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)

    class FakeFlattenedTensorBucket:
        def __init__(self, named_tensors):
            self.named_tensors = named_tensors

        def get_metadata(self):
            return [("weight", (2,), "float32")]

        def get_flattened_tensor(self):
            return torch.ones(2, dtype=torch.float32)

    def raise_from_cuda_ipc(_obj, output_str=False):
        raise RuntimeError("CUDA IPC unavailable")

    monkeypatch.setenv("COLOCATED_TENSOR_UPDATE_CUDA_IPC_FALLBACK_TO_CPU", "1")
    monkeypatch.setattr(module, "FlattenedTensorBucket", FakeFlattenedTensorBucket)
    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", raise_from_cuda_ipc)

    long_lived = []
    payload = module._serialize_flattened_bucket(
        [("weight", torch.ones(2, dtype=torch.float32))],
        long_live_tensors=long_lived,
        force_cpu_payload=False,
    )

    decoded = pickle.loads(base64.b64decode(payload))
    assert decoded["flattened_tensor"].device.type == "cpu"
    assert decoded["metadata"] == [("weight", (2,), "float32")]
    assert long_lived[-1]["flattened_tensor"].device.type == "cpu"


def test_serialize_flattened_bucket_logs_context_without_cpu_fallback(monkeypatch, caplog):
    module = _load_update_weight_module_with_stubs(monkeypatch)

    class FakeFlattenedTensorBucket:
        def __init__(self, named_tensors):
            self.named_tensors = named_tensors

        def get_metadata(self):
            return [("weight", (2,), "float32")]

        def get_flattened_tensor(self):
            return torch.ones(2, dtype=torch.float32)

    def raise_from_cuda_ipc(_obj, output_str=False):
        raise RuntimeError("CUDA IPC unavailable")

    monkeypatch.setenv("COLOCATED_TENSOR_UPDATE_CUDA_IPC_FALLBACK_TO_CPU", "0")
    monkeypatch.setattr(module, "FlattenedTensorBucket", FakeFlattenedTensorBucket)
    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", raise_from_cuda_ipc)
    caplog.set_level("ERROR", logger=module.__name__)

    with pytest.raises(RuntimeError, match="CUDA IPC unavailable"):
        module._serialize_flattened_bucket(
            [("weight", torch.ones(2, dtype=torch.float32))],
            long_live_tensors=[],
            force_cpu_payload=False,
        )

    assert "CPU fallback is disabled" in caplog.text
    assert "num_tensors=1" in caplog.text
    assert "sample_names=['weight']" in caplog.text


def test_serialize_flattened_bucket_disables_tms_for_cuda_ipc(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    calls = []

    class FakeDisableContext:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    fake_tms = types.SimpleNamespace(disable=lambda: FakeDisableContext())
    monkeypatch.setenv("LD_PRELOAD", "/tmp/torch_memory_saver_hook.so")
    monkeypatch.setitem(
        sys.modules,
        "torch_memory_saver",
        types.SimpleNamespace(torch_memory_saver=fake_tms),
    )

    class FakeFlattenedTensorBucket:
        def __init__(self, named_tensors):
            self.named_tensors = named_tensors

        def get_metadata(self):
            return [("weight", (2,), "float32")]

        def get_flattened_tensor(self):
            return torch.ones(2, dtype=torch.float32)

    serialize_calls = []

    def serialize(_obj, output_str=False):
        serialize_calls.append(list(calls))
        return "ipc-payload"

    monkeypatch.setattr(module, "FlattenedTensorBucket", FakeFlattenedTensorBucket)
    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", serialize)

    payload = module._serialize_flattened_bucket(
        [("weight", torch.ones(2, dtype=torch.float32))],
        long_live_tensors=[],
        force_cpu_payload=False,
    )

    assert payload == "ipc-payload"
    assert serialize_calls == [["enter"]]
    assert calls == ["enter", "exit"]


def test_serialize_flattened_bucket_skips_tms_disable_when_inactive(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    calls = []

    class FakeCdll:
        @staticmethod
        def tms_get_interesting_region():
            return 0

    def disable():
        calls.append("disable")
        raise AssertionError("disable should not be called")

    fake_tms = types.SimpleNamespace(
        disable=disable,
        _impl=types.SimpleNamespace(
            _binary_wrapper=types.SimpleNamespace(cdll=FakeCdll())
        ),
    )
    monkeypatch.setenv("LD_PRELOAD", "/tmp/torch_memory_saver_hook.so")
    monkeypatch.setitem(
        sys.modules,
        "torch_memory_saver",
        types.SimpleNamespace(torch_memory_saver=fake_tms),
    )

    class FakeFlattenedTensorBucket:
        def __init__(self, named_tensors):
            self.named_tensors = named_tensors

        def get_metadata(self):
            return [("weight", (2,), "float32")]

        def get_flattened_tensor(self):
            return torch.ones(2, dtype=torch.float32)

    monkeypatch.setattr(module, "FlattenedTensorBucket", FakeFlattenedTensorBucket)
    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", lambda _obj, output_str=False: "ipc-payload")

    payload = module._serialize_flattened_bucket(
        [("weight", torch.ones(2, dtype=torch.float32))],
        long_live_tensors=[],
        force_cpu_payload=False,
    )

    assert payload == "ipc-payload"
    assert calls == []


def test_serialize_flattened_bucket_tolerates_tms_inactive_assertion(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    calls = []

    class FakeDisableContext:
        def __enter__(self):
            calls.append("enter")
            raise AssertionError("disable() should be called only when tms is active")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    fake_tms = types.SimpleNamespace(disable=lambda: FakeDisableContext())
    monkeypatch.setenv("LD_PRELOAD", "/tmp/torch_memory_saver_hook.so")
    monkeypatch.setitem(
        sys.modules,
        "torch_memory_saver",
        types.SimpleNamespace(torch_memory_saver=fake_tms),
    )

    class FakeFlattenedTensorBucket:
        def __init__(self, named_tensors):
            self.named_tensors = named_tensors

        def get_metadata(self):
            return [("weight", (2,), "float32")]

        def get_flattened_tensor(self):
            return torch.ones(2, dtype=torch.float32)

    monkeypatch.setattr(module, "FlattenedTensorBucket", FakeFlattenedTensorBucket)
    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", lambda _obj, output_str=False: "ipc-payload")

    payload = module._serialize_flattened_bucket(
        [("weight", torch.ones(2, dtype=torch.float32))],
        long_live_tensors=[],
        force_cpu_payload=False,
    )

    assert payload == "ipc-payload"
    assert calls == ["enter"]


def test_send_to_colocated_engine_uses_guarded_serializer_for_pp1(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    serialize_calls = []

    def serialize_bucket(named_tensors, *, long_live_tensors, force_cpu_payload):
        serialize_calls.append(
            {
                "named_tensors": named_tensors,
                "force_cpu_payload": force_cpu_payload,
            }
        )
        long_live_tensors.append({"flattened_tensor": torch.ones(2), "metadata": []})
        return "guarded-ipc-payload"

    def gather_object(obj, object_gather_list=None, dst=0, group=None):
        object_gather_list[0] = obj

    class FakeRemote:
        def __init__(self):
            self.kwargs = None

        def remote(self, **kwargs):
            self.kwargs = kwargs
            return "ref"

    fake_remote = FakeRemote()
    fake_engine = types.SimpleNamespace(update_weights_from_tensor=fake_remote)
    monkeypatch.setattr(module, "_serialize_flattened_bucket", serialize_bucket)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "gather_object", gather_object)

    refs, long_lived = module._send_to_colocated_engine(
        [("weight", torch.ones(2, dtype=torch.float32))],
        args=Namespace(sglang_pp_size=1, num_layers=78),
        ipc_engine=fake_engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
        force_cpu_payload=False,
    )

    assert refs == ["ref"]
    assert len(long_lived) == 1
    assert len(serialize_calls) == 1
    assert serialize_calls[0]["force_cpu_payload"] is False
    assert len(serialize_calls[0]["named_tensors"]) == 1
    assert serialize_calls[0]["named_tensors"][0][0] == "weight"
    assert torch.equal(serialize_calls[0]["named_tensors"][0][1], torch.ones(2, dtype=torch.float32))
    assert fake_remote.kwargs == {
        "serialized_named_tensors": ["guarded-ipc-payload"],
        "load_format": "flattened_bucket",
        "weight_version": "7",
    }


def test_multinode_colocated_tensor_update_rejects_cuda_ipc_by_default(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.args = Namespace(
        actor_num_nodes=4,
        actor_num_gpus_per_node=8,
        rollout_num_gpus_per_engine=16,
        num_gpus_per_node=8,
        colocated_tensor_update_cpu_payload=False,
        sglang_pp_size=1,
    )
    updater._model_update_groups = None
    updater._ipc_gather_group = None

    monkeypatch.setenv("GLM52_ALLOW_UNSAFE_CROSS_NODE_CUDA_IPC", "0")
    monkeypatch.setattr(module.dist, "new_group", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)

    try:
        updater.connect_rollout_engines(
            [object()],
            object(),
            engine_gpu_counts=[16],
            engine_gpu_offsets=[0],
        )
    except RuntimeError as exc:
        assert "CUDA IPC handles are host-local" in str(exc)
    else:
        raise AssertionError("expected multi-node colocated CUDA IPC update to be rejected")


def test_multinode_colocated_tensor_update_allows_explicit_cpu_payload(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.args = Namespace(
        actor_num_nodes=4,
        actor_num_gpus_per_node=8,
        rollout_num_gpus_per_engine=16,
        num_gpus_per_node=8,
        colocated_tensor_update_cpu_payload=True,
        sglang_pp_size=1,
    )
    updater._model_update_groups = None
    updater._ipc_gather_group = None
    updater._ipc_engine = None

    monkeypatch.setattr(module.dist, "new_group", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)

    updater.connect_rollout_engines(
        [object()],
        object(),
        engine_gpu_counts=[16],
        engine_gpu_offsets=[0],
    )

    assert updater._ipc_force_cpu_payload is True


def test_select_colocated_tensor_payload_ranks_keeps_all_ranks_without_pp():
    select_colocated_tensor_payload_ranks = _load_payload_helper().select_colocated_tensor_payload_ranks
    args = Namespace(sglang_pp_size=1)

    assert select_colocated_tensor_payload_ranks(list(range(8)), args) == list(range(8))


def test_select_colocated_tensor_payload_ranks_keeps_all_tp16_ranks_without_pp():
    select_colocated_tensor_payload_ranks = _load_payload_helper().select_colocated_tensor_payload_ranks
    args = Namespace(sglang_pp_size=1)

    assert select_colocated_tensor_payload_ranks(list(range(16)), args) == list(range(16))


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
