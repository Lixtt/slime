import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path

import torch

SLIME_ROOT = Path(__file__).resolve().parents[1]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))


def _install_megatron_stubs(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")

    class _Mpu:
        @staticmethod
        def get_tensor_model_parallel_world_size():
            return 1

        @staticmethod
        def get_tensor_model_parallel_rank():
            return 0

        @staticmethod
        def get_pipeline_model_parallel_rank():
            return 0

        @staticmethod
        def get_context_parallel_rank():
            return 0

        @staticmethod
        def get_expert_model_parallel_rank():
            return 0

        @staticmethod
        def get_data_parallel_rank(with_context_parallel=False):
            return 0

        @staticmethod
        def get_pipeline_model_parallel_world_size():
            return 1

        @staticmethod
        def get_context_parallel_world_size():
            return 1

        @staticmethod
        def get_expert_model_parallel_world_size():
            return 1

    core.mpu = _Mpu

    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    mappings = types.ModuleType("megatron.core.tensor_parallel.mappings")
    mappings.gather_from_tensor_model_parallel_region = lambda tensor, group=None: tensor
    mappings.reduce_from_tensor_model_parallel_region = lambda tensor, group=None: tensor
    mappings.reduce_scatter_to_sequence_parallel_region = lambda tensor, group=None: tensor
    tensor_parallel.mappings = mappings

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel.mappings", mappings)


def _import_lora(monkeypatch):
    _install_megatron_stubs(monkeypatch)
    sys.modules.pop("slime.backends.megatron_utils.lora", None)
    return importlib.import_module("slime.backends.megatron_utils.lora")


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q_down_proj = torch.nn.Linear(4, 3, bias=False)
        self.experts = torch.nn.Module()
        self.experts.linear_q_down_proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.linear_q_down_proj(x)


def _lora_args(**overrides):
    values = {
        "lora_target_modules": "linear_q_down_proj",
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "megatron_lora_include_experts": False,
        "save": None,
        "load": None,
        "hf_checkpoint": "/base/hf",
    }
    values.update(overrides)
    return Namespace(**values)


def test_apply_megatron_lora_freezes_base_and_skips_experts_by_default(monkeypatch):
    lora = _import_lora(monkeypatch)
    model = _ToyModel()

    patched = lora.apply_megatron_lora(model, _lora_args())

    assert patched == 1
    assert model.linear_q_down_proj.weight.requires_grad is False
    assert model.linear_q_down_proj.slime_lora_A.requires_grad is True
    assert model.linear_q_down_proj.slime_lora_B.requires_grad is True
    assert not hasattr(model.experts.linear_q_down_proj, "slime_lora_A")


def test_merged_megatron_lora_matches_unmerged_forward(monkeypatch):
    lora = _import_lora(monkeypatch)
    model = _ToyModel()
    lora.apply_megatron_lora(model, _lora_args())

    module = model.linear_q_down_proj
    original_weight = module.weight.detach().clone()
    with torch.no_grad():
        module.slime_lora_A.fill_(0.5)
        module.slime_lora_B.fill_(0.25)

    x = torch.randn(5, 4)
    unmerged_output = model(x)
    expected_delta = (module.slime_lora_B @ module.slime_lora_A) * module.slime_lora_scaling

    with lora.merged_megatron_lora(model):
        assert torch.allclose(module.weight, original_weight + expected_delta)
        assert torch.allclose(model(x), unmerged_output)

    assert torch.allclose(module.weight, original_weight)
    assert torch.allclose(model(x), unmerged_output)


def test_save_and_load_megatron_lora_adapter_checkpoint(monkeypatch, tmp_path):
    lora = _import_lora(monkeypatch)
    source = _ToyModel()
    lora.apply_megatron_lora(source, _lora_args(save=str(tmp_path)))
    with torch.no_grad():
        source.linear_q_down_proj.slime_lora_A.fill_(0.75)
        source.linear_q_down_proj.slime_lora_B.fill_(0.125)

    lora.save_megatron_lora_checkpoint([source], _lora_args(save=str(tmp_path)), iteration=3)

    assert (tmp_path / "iter_0000003" / "model" / "adapter_tp00_pp00_cp00_ep00.pt").is_file()
    assert (tmp_path / "iter_0000003" / "meta.json").is_file()
    assert (tmp_path / "latest_megatron_lora_iteration.txt").read_text(encoding="utf-8").strip() == "3"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip() == "3"

    target = _ToyModel()
    lora.apply_megatron_lora(target, _lora_args())
    loaded_iteration = lora.load_megatron_lora_checkpoint([target], str(tmp_path))

    assert loaded_iteration == 3
    assert torch.allclose(target.linear_q_down_proj.slime_lora_A, source.linear_q_down_proj.slime_lora_A)
    assert torch.allclose(target.linear_q_down_proj.slime_lora_B, source.linear_q_down_proj.slime_lora_B)
