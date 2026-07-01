import json

import pytest

from slime.utils import rollout_quality_gate


class _RemoteMethod:
    def remote(self, **kwargs):
        return kwargs


class _RolloutManager:
    generation_quality_check = _RemoteMethod()


def test_quality_gate_error_is_nonfatal_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ROLLOUT_GENERATION_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv("RUN_ROOT", str(tmp_path))

    def raise_timeout(_ref):
        raise TimeoutError("probe timed out")

    monkeypatch.setattr(rollout_quality_gate.ray, "get", raise_timeout)

    assert rollout_quality_gate.run_rollout_generation_quality_gate(_RolloutManager(), "pre_update") == []

    artifact = tmp_path / "preflight" / "rollout_generation_quality_gate_pre_update.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["strict"] is False
    assert payload["errors"] == ["TimeoutError: probe timed out"]


def test_quality_gate_strict_error_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("ROLLOUT_GENERATION_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv("ROLLOUT_GENERATION_QUALITY_GATE_STRICT", "1")
    monkeypatch.setenv("RUN_ROOT", str(tmp_path))

    def raise_timeout(_ref):
        raise TimeoutError("probe timed out")

    monkeypatch.setattr(rollout_quality_gate.ray, "get", raise_timeout)

    with pytest.raises(TimeoutError, match="probe timed out"):
        rollout_quality_gate.run_rollout_generation_quality_gate(_RolloutManager(), "pre_update")

    artifact = tmp_path / "preflight" / "rollout_generation_quality_gate_pre_update.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["strict"] is True
