import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_megatron_utils_init(monkeypatch, *, initial_interesting_region: bool):
    """Load slime/backends/megatron_utils/__init__.py in isolation with stub
    deep_ep / torch_memory_saver modules, so the deep_ep.Buffer.__init__
    monkeypatch it installs can be exercised without real CUDA/Megatron.
    """
    # The real patch calls torch.cuda.synchronize() after constructing the
    # buffer; stub it so this test doesn't require an actual GPU.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None, raising=False)

    calls = []

    class FakeCdll:
        def __init__(self, state: bool):
            self._state = state

        def tms_get_interesting_region(self):
            return self._state

        def tms_set_interesting_region(self, value):
            calls.append(value)
            self._state = value

    fake_cdll = FakeCdll(initial_interesting_region)

    class FakeBuffer:
        init_calls = 0

        def __init__(self, *args, **kwargs):
            FakeBuffer.init_calls += 1
            # Record the interesting_region state as observed *during* the
            # real deep_ep.Buffer construction, matching production where
            # this would be False (memory-saver's pausable region disabled)
            # while the buffer itself is being built.
            self.region_during_init = fake_cdll.tms_get_interesting_region()

    fake_deep_ep = types.ModuleType("deep_ep")
    fake_deep_ep.Buffer = FakeBuffer

    fake_binary_wrapper = types.SimpleNamespace(cdll=fake_cdll)
    fake_impl = types.SimpleNamespace(_binary_wrapper=fake_binary_wrapper)
    fake_torch_memory_saver_singleton = types.SimpleNamespace(_impl=fake_impl)
    fake_tms_module = types.ModuleType("torch_memory_saver")
    fake_tms_module.torch_memory_saver = fake_torch_memory_saver_singleton

    monkeypatch.setitem(sys.modules, "deep_ep", fake_deep_ep)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", fake_tms_module)
    # Keep this unit test isolated from the real Megatron Bridge import graph.
    # The production module treats a missing bridge as optional.
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))

    # from . import megatron_patch is unconditional in __init__.py; stub it
    # out so loading the file doesn't require a real Megatron installation.
    package_name = "slime.backends.megatron_utils"
    monkeypatch.setitem(sys.modules, "slime", types.ModuleType("slime"))
    monkeypatch.setitem(sys.modules, "slime.backends", types.ModuleType("slime.backends"))
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.megatron_patch",
        types.ModuleType(f"{package_name}.megatron_patch"),
    )

    path = (
        Path(__file__).resolve().parents[1]
        / "slime"
        / "backends"
        / "megatron_utils"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location(package_name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    monkeypatch.setitem(sys.modules, package_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return fake_deep_ep.Buffer, fake_cdll, calls


def test_deep_ep_patch_restores_true_when_started_true(monkeypatch):
    """Baseline/no-regression case: deep_ep.Buffer() constructed outside any
    torch_memory_saver.disable() scope (interesting_region already True)
    must still end with interesting_region=True afterward.
    """
    patched_buffer_cls, fake_cdll, calls = _load_megatron_utils_init(
        monkeypatch, initial_interesting_region=True
    )

    patched_buffer_cls()

    assert fake_cdll.tms_get_interesting_region() is True
    assert calls == [False, True]


def test_deep_ep_patch_restores_false_when_nested_inside_disable(monkeypatch):
    """Regression test for the GLM5.2 colocated CUDA-IPC weight sync bug:
    deep_ep.Buffer() constructed while ALREADY nested inside an outer
    torch_memory_saver.disable() (interesting_region already False, e.g.
    during MoE/EP weight gather in update_weights()) must leave
    interesting_region False afterward, not force it back to True.
    Forcing it True silently re-enables the memory-saver's VMM-backed
    allocator for the remainder of the outer disable() scope, which makes
    any tensor allocated afterward (e.g. the CUDA-IPC flattened weight
    bucket) fail _share_cuda_ with "invalid argument".
    """
    patched_buffer_cls, fake_cdll, calls = _load_megatron_utils_init(
        monkeypatch, initial_interesting_region=False
    )

    instance = patched_buffer_cls()

    # The buffer construction itself still happens with interesting_region
    # forced False (memory-saver disabled), matching the original intent.
    assert instance.region_during_init is False
    # But the outer disable() context's guarantee must be preserved
    # afterward, not clobbered back to True.
    assert fake_cdll.tms_get_interesting_region() is False
    assert calls == [False, False]
