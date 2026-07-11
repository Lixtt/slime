from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch.utils.checkpoint import checkpoint

from slime.backends.megatron_utils.model import _collect_local_trainable_gradient_stats
from slime.backends.megatron_utils.model_provider import (
    enable_partial_train_activation_grad,
    freeze_model_params,
)


class _PartialTrainModel(torch.nn.Module):
    def __init__(self, *, recompute_granularity="full"):
        super().__init__()
        self.pre_process = True
        self.config = SimpleNamespace(recompute_granularity=recompute_granularity)
        self.embedding = torch.nn.Embedding(16, 4)
        self.decoder = torch.nn.Linear(4, 3, bias=False)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        return checkpoint(
            self.decoder,
            hidden,
            use_reentrant=True,
        )


def test_partial_train_recompute_enables_pp0_decoder_gradients():
    model = _PartialTrainModel()
    freeze_model_params(
        model,
        SimpleNamespace(
            only_train_params_name_list=[r"^decoder\."],
            freeze_params_name_list=None,
        ),
    )

    assert all(not param.requires_grad for param in model.embedding.parameters())
    assert all(param.requires_grad for param in model.decoder.parameters())
    assert enable_partial_train_activation_grad(model) is True
    assert enable_partial_train_activation_grad(model) is False

    model(torch.tensor([[1, 2, 3]])).sum().backward()

    assert model.embedding.weight.grad is None
    assert model.decoder.weight.grad is not None
    assert torch.count_nonzero(model.decoder.weight.grad) > 0


def test_partial_train_activation_grad_is_not_enabled_without_recompute():
    model = _PartialTrainModel(recompute_granularity=None)
    freeze_model_params(
        model,
        SimpleNamespace(
            only_train_params_name_list=[r"^decoder\."],
            freeze_params_name_list=None,
        ),
    )

    assert enable_partial_train_activation_grad(model) is False
    assert model.embedding(torch.tensor([[1]])).requires_grad is False


def test_trainable_gradient_stats_detect_an_empty_pipeline_stage():
    module = torch.nn.Linear(3, 2, bias=False)
    module.weight.main_grad = torch.zeros_like(module.weight)

    empty = _collect_local_trainable_gradient_stats([module])
    assert empty == {
        "trainable_param_count": 1,
        "trainable_numel": 6,
        "grad_param_count": 1,
        "grad_numel": 6,
        "has_nonzero_grad": False,
    }

    module.weight.main_grad[0, 0] = 1
    nonempty = _collect_local_trainable_gradient_stats([module, module])
    assert nonempty["trainable_param_count"] == 1
    assert nonempty["has_nonzero_grad"] is True
