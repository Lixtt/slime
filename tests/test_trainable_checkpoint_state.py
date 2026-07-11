import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "slime"
    / "backends"
    / "megatron_utils"
    / "trainable_checkpoint_state.py"
)
SPEC = importlib.util.spec_from_file_location("trainable_checkpoint_state_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
checkpoint_state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_state)

TRAINING_STATE_FORMAT = checkpoint_state.TRAINING_STATE_FORMAT
build_training_state_payload = checkpoint_state.build_training_state_payload
restore_training_state_payload = checkpoint_state.restore_training_state_payload


class _Tracker:
    def __init__(self):
        self.states = {"model-parallel-rng": torch.tensor([7], dtype=torch.uint8)}

    def get_states(self):
        return {key: value.clone() for key, value in self.states.items()}

    def set_states(self, states):
        self.states = states


class _TensorParallel:
    def __init__(self):
        self.tracker = _Tracker()

    def get_cuda_rng_tracker(self):
        return self.tracker

    @staticmethod
    def is_graph_safe_cuda_rng_tracker(_tracker):
        return False

    @staticmethod
    def convert_cuda_rng_state(value, *, to_graphable):
        assert to_graphable is False
        return value


class _StateOwner:
    is_stub_optimizer = False

    def __init__(self, state):
        self.state = state
        self.loaded = None

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.loaded = state


def _args(**overrides):
    values = {
        "no_save_optim": False,
        "no_save_rng": False,
        "consumed_train_samples": 24,
        "skipped_train_samples": 3,
        "consumed_valid_samples": 8,
    }
    values.update(overrides)
    return Namespace(**values)


def test_training_state_payload_round_trip_restores_all_state():
    random_module = __import__("random")
    random_module.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    tensor_parallel = _TensorParallel()
    optimizer = _StateOwner({"step": 5, "tensor": torch.tensor([1.5])})
    scheduler = _StateOwner({"num_steps": 24})
    args = _args()

    payload, metadata = build_training_state_payload(
        iteration=4,
        rank=2,
        world_size=8,
        topology={"tensor_model_parallel_rank": 2},
        args=args,
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        tensor_parallel=tensor_parallel,
    )

    assert payload["format"] == TRAINING_STATE_FORMAT
    assert metadata == {
        "optimizer_state_saved": True,
        "scheduler_state_saved": True,
        "rng_state_saved": True,
    }
    expected_python = random_module.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    tensor_parallel.tracker.states = {
        "model-parallel-rng": torch.tensor([99], dtype=torch.uint8)
    }
    target_args = _args(
        consumed_train_samples=0,
        skipped_train_samples=0,
        consumed_valid_samples=0,
    )
    target_optimizer = _StateOwner({})
    target_scheduler = _StateOwner({})

    loaded, warnings = restore_training_state_payload(
        payload,
        expected_iteration=4,
        expected_rank=2,
        expected_world_size=8,
        args=target_args,
        optimizer=target_optimizer,
        opt_param_scheduler=target_scheduler,
        tensor_parallel=tensor_parallel,
        strict=True,
        source="test.pt",
    )

    assert loaded is True
    assert warnings == []
    assert target_optimizer.loaded["step"] == 5
    assert target_scheduler.loaded == {"num_steps": 24}
    assert target_args.consumed_train_samples == 24
    assert target_args.skipped_train_samples == 3
    assert target_args.consumed_valid_samples == 8
    assert random_module.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    assert tensor_parallel.tracker.states["model-parallel-rng"].tolist() == [7]


def test_training_state_strict_resume_rejects_missing_optimizer():
    tensor_parallel = _TensorParallel()
    payload, _ = build_training_state_payload(
        iteration=0,
        rank=0,
        world_size=1,
        topology={},
        args=_args(no_save_optim=True),
        optimizer=_StateOwner({"step": 1}),
        opt_param_scheduler=_StateOwner({"num_steps": 1}),
        tensor_parallel=tensor_parallel,
    )

    with pytest.raises(KeyError, match="no optimizer state"):
        restore_training_state_payload(
            payload,
            expected_iteration=0,
            expected_rank=0,
            expected_world_size=1,
            args=_args(),
            optimizer=_StateOwner({}),
            opt_param_scheduler=_StateOwner({}),
            tensor_parallel=tensor_parallel,
            strict=True,
            source="test.pt",
        )


def test_training_state_rejects_topology_identity_mismatch():
    tensor_parallel = _TensorParallel()
    payload, _ = build_training_state_payload(
        iteration=2,
        rank=0,
        world_size=4,
        topology={},
        args=_args(),
        optimizer=_StateOwner({"step": 1}),
        opt_param_scheduler=_StateOwner({"num_steps": 1}),
        tensor_parallel=tensor_parallel,
    )

    with pytest.raises(ValueError, match="world_size mismatch"):
        restore_training_state_payload(
            payload,
            expected_iteration=2,
            expected_rank=0,
            expected_world_size=8,
            args=_args(),
            optimizer=_StateOwner({}),
            opt_param_scheduler=_StateOwner({}),
            tensor_parallel=tensor_parallel,
            strict=True,
            source="test.pt",
        )
