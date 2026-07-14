from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.ray.actor_group import RayTrainGroup


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeEngine:
    def __init__(self, *, reported_version=None):
        self.version = "default"
        self.reported_version = reported_version
        self.events = []
        self.pause_generation = _RemoteMethod(lambda: self.events.append("pause"))
        self.flush_cache = _RemoteMethod(lambda: self.events.append("flush"))
        self.update_weights_from_disk = _RemoteMethod(self._update_weights_from_disk)
        self.get_weight_version = _RemoteMethod(self._get_weight_version)
        self.continue_generation = _RemoteMethod(lambda: self.events.append("continue"))

    def _update_weights_from_disk(self, *, model_path, weight_version):
        self.events.append(("reload", model_path, weight_version))
        self.version = weight_version

    def _get_weight_version(self):
        self.events.append("verify")
        return self.reported_version if self.reported_version is not None else self.version


class _FakeRolloutManager:
    def __init__(self, engines):
        self.get_updatable_engines_and_lock = _RemoteMethod(lambda: (engines, None, 0, [], []))


def _group(**overrides):
    values = {
        "release_train": True,
        "save": "/checkpoints/save",
        "load": "/checkpoints/base",
        "megatron_trainable_only_load": "/checkpoints/previous-overlay",
        "ckpt_step": 7,
        "finetune": True,
        "no_load_optim": True,
        "no_load_rng": True,
        "no_save_optim": False,
        "train_env_vars": {},
        "update_weight_start_version": 0,
        "offload_rollout": False,
        "update_weight_local_checkpoint_dir": None,
        "update_weight_disk_keep_files": True,
        "ci_test": False,
        "verify_rollout_weight_version_after_update": True,
    }
    values.update(overrides)
    return RayTrainGroup(
        SimpleNamespace(**values),
        num_nodes=1,
        num_gpus_per_node=1,
        pg=None,
    )


@pytest.mark.unit
def test_release_train_full_checkpoint_reloads_saved_root(monkeypatch):
    monkeypatch.delenv("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE", raising=False)
    group = _group()

    group._prepare_release_train_reload()

    assert group.args.load == "/checkpoints/save"
    assert group.args.ckpt_step is None
    assert group.args.finetune is False
    assert group.args.no_load_optim is False
    assert group.args.no_load_rng is False


@pytest.mark.unit
def test_release_train_trainable_overlay_preserves_full_base(monkeypatch):
    monkeypatch.setenv("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE", "1")
    group = _group()

    group._prepare_release_train_reload()

    assert group.args.load == "/checkpoints/base"
    assert group.args.megatron_trainable_only_load == "/checkpoints/save"
    assert group.args.ckpt_step == 7
    assert group.args.finetune is True
    assert group.args.no_load_optim is True
    assert group.args.no_load_rng is True


@pytest.mark.unit
def test_release_train_trainable_overlay_reads_actor_env(monkeypatch):
    monkeypatch.delenv("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE", raising=False)
    group = _group(train_env_vars={"SLIME_MEGATRON_TRAINABLE_ONLY_SAVE": "true"})

    group._prepare_release_train_reload()

    assert group.args.load == "/checkpoints/base"
    assert group.args.megatron_trainable_only_load == "/checkpoints/save"


@pytest.mark.unit
def test_full_disk_reload_verifies_every_engine_after_reload(monkeypatch, tmp_path):
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda value: value)
    engines = [_FakeEngine(), _FakeEngine()]
    group = _group()
    group._rollout_manager = _FakeRolloutManager(engines)

    group._reload_rollout_weights_from_disk(Path(tmp_path), "3")

    expected = ["pause", "flush", ("reload", str(tmp_path), "3"), "verify", "continue"]
    assert [engine.events for engine in engines] == [expected, expected]


@pytest.mark.unit
def test_full_disk_reload_fails_closed_on_engine_version_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda value: value)
    engines = [_FakeEngine(), _FakeEngine(reported_version="default")]
    group = _group()
    group._rollout_manager = _FakeRolloutManager(engines)

    with pytest.raises(RuntimeError, match=r"Expected: 4; engine 1: default"):
        group._reload_rollout_weights_from_disk(Path(tmp_path), "4")

    assert "continue" not in engines[0].events
    assert "continue" not in engines[1].events
