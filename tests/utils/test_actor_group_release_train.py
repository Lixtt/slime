from types import SimpleNamespace

import pytest

from slime.ray.actor_group import RayTrainGroup


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
