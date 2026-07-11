from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_sglang_stubs(monkeypatch) -> None:
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    server_args = types.ModuleType("sglang.srt.server_args")
    utils = types.ModuleType("sglang.srt.utils")
    router = types.ModuleType("sglang_router")
    ray_actor = types.ModuleType("slime.ray.ray_actor")
    qwen35 = types.ModuleType("slime.backends.sglang_utils.qwen3_5")
    http_utils = types.ModuleType("slime.utils.http_utils")

    class FakeServerArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    server_args.ServerArgs = FakeServerArgs
    utils.kill_process_tree = lambda *args, **kwargs: None
    router.__version__ = "0.2.2"
    ray_actor.RayActor = object
    qwen35.is_qwen35_model_path = lambda *args, **kwargs: False
    qwen35.maybe_prepare_qwen35_text_model = lambda server_args: server_args
    qwen35.patch_sglang_qwen35 = lambda: None
    http_utils.get_host_info = lambda: ("127.0.0.1", "127.0.0.1")

    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)
    monkeypatch.setitem(sys.modules, "sglang_router", router)
    monkeypatch.setitem(sys.modules, "slime.ray.ray_actor", ray_actor)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.qwen3_5", qwen35)
    monkeypatch.setitem(sys.modules, "slime.utils.http_utils", http_utils)


def _load_sglang_engine(monkeypatch):
    _install_sglang_stubs(monkeypatch)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("slime.backends.sglang_utils.sglang_engine", None)
    return importlib.import_module("slime.backends.sglang_utils.sglang_engine")


class _Response:
    def __init__(self, status_code: int, data: dict | None = None):
        self.status_code = status_code
        self.text = "OK"
        self._data = data or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._data


def test_wait_server_healthy_skips_generation_probe_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "false")
    module = _load_sglang_engine(monkeypatch)
    calls: list[str] = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            calls.append(url)
            if url.endswith("/health_generate"):
                raise AssertionError("/health_generate should not be probed when generation health is disabled")
            if url.endswith("/health"):
                health_calls = sum(call.endswith("/health") for call in calls)
                return _Response(200 if health_calls >= 2 else 503)
            if url.endswith("/flush_cache"):
                return _Response(200)
            raise AssertionError(url)

    monkeypatch.setattr(module.requests, "Session", FakeSession)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    module._wait_server_healthy("http://engine", "token", lambda: True)

    assert "http://engine/health_generate" not in calls
    assert calls == ["http://engine/health", "http://engine/health", "http://engine/flush_cache"]


def test_engine_health_method_uses_plain_health_when_generation_probe_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "false")
    module = _load_sglang_engine(monkeypatch)
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _Response(200)

    monkeypatch.setattr(module.requests, "get", fake_get)
    engine = object.__new__(module.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000

    assert engine.health_generate(timeout=1.0) is True
    assert calls == ["http://127.0.0.1:30000/health"]


def test_slime_override_can_reenable_generation_health_probe(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "false")
    monkeypatch.setenv("SLIME_SGLANG_ENABLE_HEALTH_GENERATE", "true")
    module = _load_sglang_engine(monkeypatch)

    assert module._health_check_endpoints() == ("/health", "/health_generate")


def test_get_weight_version_uses_model_info(monkeypatch) -> None:
    module = _load_sglang_engine(monkeypatch)
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _Response(200, {"weight_version": "7"})

    monkeypatch.setattr(module.requests, "get", fake_get)
    engine = object.__new__(module.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000

    assert engine.get_weight_version() == "7"
    assert calls == ["http://127.0.0.1:30000/model_info"]


def test_get_weight_version_falls_back_for_legacy_sglang(monkeypatch) -> None:
    module = _load_sglang_engine(monkeypatch)
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url.endswith("/model_info"):
            return _Response(404)
        return _Response(200, {"weight_version": "3"})

    monkeypatch.setattr(module.requests, "get", fake_get)
    engine = object.__new__(module.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000

    assert engine.get_weight_version() == "3"
    assert calls == [
        "http://127.0.0.1:30000/model_info",
        "http://127.0.0.1:30000/get_weight_version",
    ]


def test_runtime_token_capacity_gate_uses_profiled_values(monkeypatch) -> None:
    module = _load_sglang_engine(monkeypatch)
    engine = object.__new__(module.SGLangEngine)
    engine.node_rank = 0
    engine.args = types.SimpleNamespace(
        rollout_min_kv_tokens=191873,
        rollout_min_input_tokens=159040,
    )

    engine._validate_runtime_token_capacity(
        {
            "max_total_num_tokens": 199488,
            "max_req_input_len": 199482,
            "context_length": 200000,
            "max_total_tokens": 200000,
            "mem_fraction_static": 0.83,
        }
    )

    with pytest.raises(RuntimeError, match=r"KV token pool 118848 < required 191873"):
        engine._validate_runtime_token_capacity(
            {
                "max_total_num_tokens": 118848,
                "max_req_input_len": 118842,
                "context_length": 200000,
                "max_total_tokens": 200000,
                "mem_fraction_static": 0.80,
            }
        )
