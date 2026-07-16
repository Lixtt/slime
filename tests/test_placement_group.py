import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.ray.placement_group import _create_placement_group, _get_placement_group_layout, create_training_models

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "actor_num_nodes": 2,
        "actor_num_gpus_per_node": 8,
        "rollout_num_gpus": 32,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": False,
        "rollout_external": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, (48, 16), id="normal_non_colocate"),
        pytest.param({"debug_train_only": True}, (16, 0), id="debug_train_only"),
        pytest.param({"debug_rollout_only": True}, (32, 0), id="debug_rollout_only"),
        pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (0, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected


def test_create_zero_gpu_placement_group_is_empty():
    assert _create_placement_group(0) == (None, [], [])


def test_rollout_only_does_not_allocate_training_actors(monkeypatch):
    load_calls = []

    class LoadMethod:
        @staticmethod
        def remote(rollout_id):
            load_calls.append(rollout_id)
            return rollout_id

    class RolloutManager:
        load = LoadMethod()

    def fail_allocate(*args, **kwargs):
        raise AssertionError("rollout-only mode must not allocate a training actor")

    args = _args(debug_rollout_only=True)
    args.start_rollout_id = None
    args.rollout_global_dataset = True
    monkeypatch.setattr("slime.ray.placement_group.allocate_train_group", fail_allocate)
    monkeypatch.setattr("slime.ray.placement_group.ray.get", lambda value: value)

    actor_model, critic_model = create_training_models(args, {"actor": None}, RolloutManager())

    assert actor_model is None
    assert critic_model is None
    assert args.start_rollout_id == 0
    assert load_calls == [-1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
