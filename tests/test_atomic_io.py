import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE_PATH = Path(__file__).resolve().parents[1] / "slime" / "utils" / "atomic_io.py"
SPEC = importlib.util.spec_from_file_location("atomic_io_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
atomic_io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atomic_io)


def test_atomic_torch_save_publishes_complete_payload(tmp_path: Path):
    path = tmp_path / "nested" / "rollout.pt"

    atomic_io.atomic_torch_save({"rollout_id": 3}, path, durable=True)

    assert torch.load(path, map_location="cpu", weights_only=False) == {"rollout_id": 3}
    assert list(path.parent.glob(f".{path.name}.tmp.*")) == []


def test_atomic_torch_save_keeps_previous_file_on_failure(monkeypatch, tmp_path: Path):
    path = tmp_path / "rollout.pt"
    atomic_io.atomic_torch_save({"version": "old"}, path)

    def fail_save(_obj, temporary):
        Path(temporary).write_bytes(b"partial")
        raise RuntimeError("simulated interrupted save")

    monkeypatch.setattr(atomic_io.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_io.atomic_torch_save({"version": "new"}, path)

    assert torch.load(path, map_location="cpu", weights_only=False) == {"version": "old"}
    assert list(tmp_path.glob(f".{path.name}.tmp.*")) == []
