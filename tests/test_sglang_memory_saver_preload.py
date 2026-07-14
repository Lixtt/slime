from pathlib import Path


def test_rollout_actor_does_not_preload_torch_memory_saver() -> None:
    rollout_source = (Path(__file__).parents[1] / "slime" / "ray" / "rollout.py").read_text()

    assert "_torch_memory_saver_preload_env" not in rollout_source
    assert 'env_vars["LD_PRELOAD"]' not in rollout_source
    assert 'return {"LD_PRELOAD"' not in rollout_source
