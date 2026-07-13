from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
from slime.backends.fsdp_utils import checkpoint


def _actor(no_load_rng: bool = False):
    return types.SimpleNamespace(
        args=types.SimpleNamespace(no_load_rng=no_load_rng, start_rollout_id=None),
        global_step=0,
        micro_step=0,
    )


def test_finalize_load_skips_cuda_rng_when_device_count_changes(monkeypatch):
    called = False

    def fail_set_rng_state_all(_states):
        nonlocal called
        called = True

    monkeypatch.setattr(checkpoint.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(checkpoint.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(checkpoint.torch.cuda, "set_rng_state_all", fail_set_rng_state_all)
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)

    checkpoint.finalize_load(_actor(), {"rng": {"cuda": [torch.ByteTensor([0])] * 8}})

    assert called is False


def test_finalize_load_restores_cuda_rng_when_device_count_matches(monkeypatch):
    restored = []

    monkeypatch.setattr(checkpoint.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(checkpoint.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(checkpoint.torch.cuda, "set_rng_state_all", lambda states: restored.append(states))
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)

    states = [torch.ByteTensor([0]), torch.ByteTensor([1])]
    checkpoint.finalize_load(_actor(), {"rng": {"cuda": states}})

    assert restored == [states]
