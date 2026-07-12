from types import SimpleNamespace

import torch

from slime.backends.megatron_utils import model_provider


def test_critic_value_head_accepts_base_lm_checkpoint_shape(monkeypatch):
    calls = []

    def fake_make_sharded_tensor(tensor, key, **kwargs):
        calls.append((tensor, key, kwargs))
        return key

    monkeypatch.setattr(model_provider, "make_sharded_tensor_for_checkpoint", fake_make_sharded_tensor)
    config = SimpleNamespace(hidden_size=16, sequence_parallel=False, init_method_std=0.01)

    head = model_provider._build_critic_output_layer(config)
    sharded = head.sharded_state_dict(prefix="output_layer.")

    assert head.weight.shape == (1, 16)
    assert head.bias is None
    assert sharded == {"output_layer.weight": "output_layer.weight"}
    assert len(calls) == 1
    tensor, key, kwargs = calls[0]
    assert tensor is head.weight
    assert key == "output_layer.weight"
    assert kwargs["allow_shape_mismatch"] is True


def test_critic_value_head_uses_model_init_std():
    torch.manual_seed(1234)
    config = SimpleNamespace(hidden_size=4096, sequence_parallel=False, init_method_std=0.005)

    head = model_provider._build_critic_output_layer(config)

    assert 0.0045 < head.weight.float().std().item() < 0.0055
