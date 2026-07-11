from pathlib import Path

import pytest

from slime.utils.misc import validate_rollout_window


def test_validate_rollout_window_accepts_nonempty_and_deferred_bounds():
    validate_rollout_window(1, 2)
    validate_rollout_window(None, 2)
    validate_rollout_window(1, None)


@pytest.mark.parametrize(("start_rollout_id", "num_rollout"), [(1, 1), (2, 1)])
def test_validate_rollout_window_rejects_empty_or_reversed_bounds(
    start_rollout_id: int,
    num_rollout: int,
):
    with pytest.raises(ValueError, match="exclusive terminal rollout id"):
        validate_rollout_window(start_rollout_id, num_rollout)


def test_validate_rollout_window_allows_explicit_eval_only_window():
    validate_rollout_window(0, 0, allow_empty=True)


@pytest.mark.parametrize("entrypoint", ["train.py", "train_async.py"])
def test_training_entrypoints_validate_inferred_rollout_window(entrypoint: str):
    source = (Path(__file__).resolve().parents[1] / entrypoint).read_text(encoding="utf-8")
    assert "validate_rollout_window(" in source
