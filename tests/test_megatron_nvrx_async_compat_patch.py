import importlib.util
import sys
import types
from pathlib import Path


def test_nvrx_async_patch_adds_public_queue_alias(monkeypatch):
    filesystem_async = types.ModuleType(
        "nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async"
    )

    def private_queue():
        return "queue"

    filesystem_async._get_write_results_queue = private_queue

    async_ckpt = types.ModuleType("nvidia_resiliency_ext.checkpointing.async_ckpt")
    async_ckpt.filesystem_async = filesystem_async
    checkpointing = types.ModuleType("nvidia_resiliency_ext.checkpointing")
    checkpointing.async_ckpt = async_ckpt
    nvidia_ext = types.ModuleType("nvidia_resiliency_ext")
    nvidia_ext.checkpointing = checkpointing

    monkeypatch.setitem(sys.modules, "nvidia_resiliency_ext", nvidia_ext)
    monkeypatch.setitem(sys.modules, "nvidia_resiliency_ext.checkpointing", checkpointing)
    monkeypatch.setitem(sys.modules, "nvidia_resiliency_ext.checkpointing.async_ckpt", async_ckpt)
    monkeypatch.setitem(
        sys.modules,
        "nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async",
        filesystem_async,
    )

    module_path = (
        Path(__file__).resolve().parents[1]
        / "slime"
        / "backends"
        / "megatron_utils"
        / "megatron_patch"
        / "megatron_nvrx_async_compat_patch.py"
    )
    module_name = "test_megatron_nvrx_async_compat_patch_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert filesystem_async.get_write_results_queue is private_queue
