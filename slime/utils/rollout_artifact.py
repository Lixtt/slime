from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def behavior_policy_identity_digest(identity: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def load_behavior_policy_manifest(path: str | Path, *, rollout_id: int) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to load rollout behavior-policy manifest {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"Rollout behavior-policy manifest must be a JSON object: {manifest_path}")
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "Unsupported rollout behavior-policy manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("expected_rollout_id") != rollout_id:
        raise ValueError(
            "Rollout behavior-policy manifest is bound to a different rollout: "
            f"expected_rollout_id={manifest.get('expected_rollout_id')!r}, actual={rollout_id}"
        )

    identity = manifest.get("identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Rollout behavior-policy manifest has no identity object.")
    if identity.get("expected_rollout_id") != rollout_id:
        raise ValueError(
            "Rollout behavior-policy identity is bound to a different rollout: "
            f"expected_rollout_id={identity.get('expected_rollout_id')!r}, actual={rollout_id}"
        )
    model = identity.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("path"), str) or not model["path"].strip():
        raise ValueError("Rollout behavior-policy identity has no model path.")
    expected_weight_versions = identity.get("expected_weight_versions")
    if (
        not isinstance(expected_weight_versions, list)
        or not expected_weight_versions
        or any(not isinstance(version, str) or not version.strip() for version in expected_weight_versions)
    ):
        raise ValueError("Rollout behavior-policy identity has no valid expected weight versions.")
    actual_digest = behavior_policy_identity_digest(identity)
    if manifest.get("identity_digest") != actual_digest:
        raise ValueError(
            "Rollout behavior-policy manifest identity digest mismatch: "
            f"recorded={manifest.get('identity_digest')!r}, actual={actual_digest}"
        )
    return manifest
