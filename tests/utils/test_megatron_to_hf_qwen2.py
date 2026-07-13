import importlib.util
import types
from pathlib import Path

import torch


def _load_convert_qwen2_to_hf():
    path = (
        Path(__file__).resolve().parents[2]
        / "slime"
        / "backends"
        / "megatron_utils"
        / "megatron_to_hf"
        / "qwen2.py"
    )
    spec = importlib.util.spec_from_file_location("qwen2_to_hf_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_qwen2_to_hf


def _args():
    return types.SimpleNamespace(
        hidden_size=16,
        kv_channels=4,
        num_attention_heads=4,
        num_query_groups=2,
    )


def test_qwen2_converter_accepts_bridge_input_layernorm_name():
    convert_qwen2_to_hf = _load_convert_qwen2_to_hf()
    weight = torch.ones(16)

    result = convert_qwen2_to_hf(
        _args(),
        "module.module.decoder.layers.0.input_layernorm.weight",
        weight,
    )

    assert result[0][0] == "model.layers.0.input_layernorm.weight"
    assert result[0][1] is weight


def test_qwen2_converter_accepts_bridge_pre_mlp_layernorm_name():
    convert_qwen2_to_hf = _load_convert_qwen2_to_hf()
    weight = torch.ones(16)

    result = convert_qwen2_to_hf(
        _args(),
        "module.module.decoder.layers.0.pre_mlp_layernorm.weight",
        weight,
    )

    assert result[0][0] == "model.layers.0.post_attention_layernorm.weight"
    assert result[0][1] is weight
