import json
import logging
import os
import re
from pathlib import Path

import ray

logger = logging.getLogger(__name__)


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _artifact_name(label: str) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "probe"
    return f"rollout_generation_quality_gate_{safe_label}.json"


def run_rollout_generation_quality_gate(rollout_manager, label: str) -> list[dict]:
    """Run a tiny generation probe against rollout engines and fail fast on corruption."""

    if not _truthy(os.environ.get("ROLLOUT_GENERATION_QUALITY_GATE_ENABLED")):
        return []

    results = ray.get(rollout_manager.generation_quality_check.remote(label=label))
    results = [item for item in results if item is not None]

    run_root = os.environ.get("RUN_ROOT") or os.environ.get("A3S_CODE_RUN_ROOT")
    payload = {
        "label": label,
        "ok": bool(results) and all(item.get("ok") for item in results),
        "results": results,
    }
    if run_root:
        output_dir = Path(run_root) / "preflight"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / _artifact_name(label)).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if not results:
        raise RuntimeError(f"Rollout generation quality gate {label!r} returned no node-0 engine results.")

    failures = [item for item in results if not item.get("ok")]
    if failures:
        first = failures[0]
        errors = "; ".join(first.get("errors") or [])
        raise RuntimeError(
            f"Rollout generation quality gate {label!r} failed on engine {first.get('engine_rank')}: {errors}"
        )

    logger.info("Rollout generation quality gate %s passed on %d engine(s).", label, len(results))
    return results
