import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch


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


class FakeFlattenedTensorBucket:
    def __init__(self, named_tensors):
        self.named_tensors = named_tensors

    def get_metadata(self):
        return [("weight", (2,), "float32")]

    def get_flattened_tensor(self):
        return torch.ones(2, dtype=torch.float32)


def test_serialize_flattened_bucket_returns_serializer_output(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)

    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", lambda _obj, output_str=False: "ipc-payload")

    payload = module._serialize_flattened_bucket(
        {"flattened_tensor": torch.ones(2, dtype=torch.float32), "metadata": []},
        [("weight", torch.ones(2, dtype=torch.float32))],
    )

    assert payload == "ipc-payload"


def test_serialize_flattened_bucket_logs_once_and_reraises(monkeypatch, caplog):
    module = _load_update_weight_module_with_stubs(monkeypatch)

    def raise_from_cuda_ipc(_obj, output_str=False):
        raise RuntimeError("CUDA IPC unavailable")

    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", raise_from_cuda_ipc)
    caplog.set_level("ERROR", logger=module.__name__)

    with pytest.raises(RuntimeError, match="CUDA IPC unavailable"):
        module._serialize_flattened_bucket(
            {"flattened_tensor": torch.ones(2, dtype=torch.float32), "metadata": []},
            [("weight", torch.ones(2, dtype=torch.float32))],
        )

    assert "num_tensors=1" in caplog.text
    assert "sample_names=['weight']" in caplog.text


def test_send_to_colocated_engine_serializes_and_dispatches(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)

    monkeypatch.setattr(module, "FlattenedTensorBucket", FakeFlattenedTensorBucket)
    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", lambda _obj, output_str=False: "ipc-payload")

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
    monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "gather_object", gather_object)

    refs, long_lived = module._send_to_colocated_engine(
        [("weight", torch.ones(2, dtype=torch.float32))],
        ipc_engine=fake_engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
    )

    assert refs == ["ref"]
    assert len(long_lived) == 1
    assert fake_remote.kwargs == {
        "serialized_named_tensors": ["ipc-payload"],
        "load_format": "flattened_bucket",
        "weight_version": "7",
    }


def test_send_to_colocated_engine_skips_when_no_gather_group(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)

    refs, long_lived = module._send_to_colocated_engine(
        [("weight", torch.ones(2, dtype=torch.float32))],
        ipc_engine=None,
        ipc_gather_src=0,
        ipc_gather_group=None,
        weight_version=1,
    )

    assert refs == []
    assert long_lived is None


def test_flattened_cuda_ipc_bucket_reuses_caller_allocator_pool(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    monkeypatch.setattr(module, "FlattenedTensorBucket", FakeFlattenedTensorBucket)
    monkeypatch.setattr(module.MultiprocessingSerializer, "serialize", lambda _obj, output_str=False: "ipc-payload")
    monkeypatch.setattr(module, "_torch_memory_saver_interesting_region", lambda: False)
    monkeypatch.setattr(
        module.torch.cuda,
        "MemPool",
        lambda: pytest.fail("CUDA IPC export must not create a nested MemPool"),
    )
    fake_cuda_tensor = types.SimpleNamespace(is_cuda=True)

    payload, owner = module._flatten_and_serialize_for_cuda_ipc([("weight", fake_cuda_tensor)])

    assert payload == "ipc-payload"
    assert owner["flattened_tensor"].shape == (2,)


def test_flattened_cuda_ipc_bucket_rejects_active_tms(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    monkeypatch.setattr(module, "FlattenedTensorBucket", FakeFlattenedTensorBucket)
    monkeypatch.setattr(module, "_torch_memory_saver_interesting_region", lambda: True)
    fake_cuda_tensor = types.SimpleNamespace(is_cuda=True)

    with pytest.raises(RuntimeError, match="torch_memory_saver is active"):
        module._flatten_and_serialize_for_cuda_ipc([("weight", fake_cuda_tensor)])


def test_collect_released_cuda_ipc_handles_uses_lightweight_cuda_collect(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    events = []
    monkeypatch.setattr(module.torch.cuda, "ipc_collect", lambda: events.append("ipc_collect"))

    module._collect_released_cuda_ipc_handles()

    assert events == ["ipc_collect"]


def test_fp8_tensor_update_refreshes_quantized_weights(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    post_process_calls = []
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.weight_version = 0
    updater.rollout_engines = []
    updater.quantization_config = {"quant_method": "fp8"}
    updater.weights_getter = lambda: {}
    updater._hf_weight_iterator = types.SimpleNamespace(get_hf_weight_chunks=lambda _weights: [])

    monkeypatch.setattr(module.ray, "get", lambda _refs: None, raising=False)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(module, "_collect_released_cuda_ipc_handles", lambda: None)
    monkeypatch.setattr(
        module,
        "post_process_weights",
        lambda **kwargs: post_process_calls.append(kwargs),
    )

    updater.update_weights()

    assert updater.weight_version == 1
    assert len(post_process_calls) == 1
    assert post_process_calls[0]["restore_weights_before_load"] is False
    assert post_process_calls[0]["post_process_quantization"] is True


def test_connect_rollout_engines_maps_single_node_colocated_engine(monkeypatch):
    module = _load_update_weight_module_with_stubs(monkeypatch)
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.args = Namespace(actor_num_nodes=4, actor_num_gpus_per_node=8)
    updater._model_update_groups = None
    updater._ipc_gather_group = None

    monkeypatch.setattr(module.dist, "new_group", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)

    fake_engine = object()
    updater.connect_rollout_engines(
        [fake_engine],
        object(),
        engine_gpu_counts=[8],
        engine_gpu_offsets=[0],
    )

    assert updater.use_distribute is False
    assert updater._ipc_engine is fake_engine
    assert updater._ipc_gather_src == 0
