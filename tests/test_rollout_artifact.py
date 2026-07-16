from __future__ import annotations

import json
from pathlib import Path

import pytest

from slime.utils.rollout_artifact import (
    behavior_policy_identity_digest,
    load_behavior_policy_manifest,
)


def _write_manifest(path: Path, *, rollout_id: int = 3) -> dict:
    identity = {
        "format": "openclaw_rollout_behavior_policy_v1",
        "expected_rollout_id": rollout_id,
        "model": {"path": "/models/iter2", "model_index_sha256": "a" * 64},
        "sampling": {"temperature": "0.6", "top_p": "1.0"},
        "target_train_iteration": rollout_id,
        "expected_weight_versions": ["default"],
    }
    manifest = {
        "schema_version": 1,
        "expected_rollout_id": rollout_id,
        "identity": identity,
        "identity_digest": behavior_policy_identity_digest(identity),
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_load_behavior_policy_manifest_accepts_matching_identity(tmp_path: Path):
    path = tmp_path / "policy.json"
    expected = _write_manifest(path)

    assert load_behavior_policy_manifest(path, rollout_id=3) == expected


def test_load_behavior_policy_manifest_rejects_rollout_or_digest_mismatch(tmp_path: Path):
    path = tmp_path / "policy.json"
    manifest = _write_manifest(path)

    with pytest.raises(ValueError, match="different rollout"):
        load_behavior_policy_manifest(path, rollout_id=4)

    manifest["identity"]["model"]["path"] = "/models/other"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity digest mismatch"):
        load_behavior_policy_manifest(path, rollout_id=3)


def test_load_behavior_policy_manifest_rejects_incomplete_identity(tmp_path: Path):
    path = tmp_path / "policy.json"
    manifest = _write_manifest(path)
    manifest["identity"]["expected_weight_versions"] = []
    manifest["identity_digest"] = behavior_policy_identity_digest(manifest["identity"])
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="expected weight versions"):
        load_behavior_policy_manifest(path, rollout_id=3)
