import os
import uuid
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(obj: Any, path: Path, *, durable: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(obj, temporary)
        if durable:
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        if durable:
            try:
                parent_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            except OSError:
                pass
    finally:
        temporary.unlink(missing_ok=True)
