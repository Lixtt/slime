#!/usr/bin/env python3
"""Probe CUDA IPC export under torch-memory-saver allocator states."""

import argparse
import gc
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from torch_memory_saver import torch_memory_saver


def _interesting_region() -> bool:
    torch_memory_saver._ensure_initialized()
    return bool(torch_memory_saver._impl._binary_wrapper.cdll.tms_get_interesting_region())


@contextmanager
def _disable_tms_hook_only():
    """Disable TMS interception without switching to a temporary CUDA MemPool."""
    torch_memory_saver._ensure_initialized()
    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    prior = bool(cdll.tms_get_interesting_region())
    cdll.tms_set_interesting_region(False)
    try:
        yield
    finally:
        cdll.tms_set_interesting_region(prior)


@contextmanager
def _disable_tms_with_export_pool():
    with torch_memory_saver.disable():
        export_pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(export_pool):
            yield


def _serialize_bucket(size_bytes: int) -> int:
    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive")
    left = size_bytes // 2
    right = size_bytes - left
    bucket = FlattenedTensorBucket(
        named_tensors=[
            ("left", torch.zeros(left, dtype=torch.uint8, device="cuda")),
            ("right", torch.ones(right, dtype=torch.uint8, device="cuda")),
        ]
    )
    flattened = bucket.get_flattened_tensor()
    torch.cuda.synchronize()
    payload = MultiprocessingSerializer.serialize(
        {"flattened_tensor": flattened, "metadata": bucket.get_metadata()},
        output_str=True,
    )
    return len(payload)


def _run_case(name, size_bytes, context_factory):
    try:
        with context_factory():
            state_during = _interesting_region()
            payload_bytes = _serialize_bucket(size_bytes)
        return {
            "name": name,
            "ok": True,
            "tms_interesting_region": state_during,
            "payload_chars": payload_bytes,
        }
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "tms_interesting_region": _interesting_region(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        from torch.multiprocessing import reductions

        reductions.shared_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _run_after_tms_pause(size_bytes: int, paused_allocation_bytes: int, repeats: int):
    resident = []
    paused = False
    try:
        chunk_bytes = 256 * 1024 * 1024
        remaining = paused_allocation_bytes
        while remaining > 0:
            allocation_bytes = min(chunk_bytes, remaining)
            resident.append(torch.zeros(allocation_bytes, dtype=torch.uint8, device="cuda"))
            remaining -= allocation_bytes
        torch.cuda.synchronize()
        torch_memory_saver.pause()
        paused = True
        with torch_memory_saver.disable():
            state_during = _interesting_region()
            payload_bytes = []
            for _ in range(repeats):
                payload_bytes.append(_serialize_bucket(size_bytes))
                from torch.multiprocessing import reductions

                reductions.shared_cache.clear()
                gc.collect()
                torch.cuda.ipc_collect()
        return {
            "name": "after_tms_pause",
            "ok": True,
            "tms_interesting_region": state_during,
            "paused_allocation_bytes": paused_allocation_bytes,
            "repeats": repeats,
            "payload_chars": payload_bytes,
        }
    except Exception as exc:
        return {
            "name": "after_tms_pause",
            "ok": False,
            "tms_interesting_region": _interesting_region(),
            "paused_allocation_bytes": paused_allocation_bytes,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if paused:
            torch_memory_saver.resume()
        resident.clear()
        from torch.multiprocessing import reductions

        reductions.shared_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@contextmanager
def _no_context():
    yield


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=216)
    parser.add_argument("--size-bytes", type=int)
    parser.add_argument("--paused-allocation-mib", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--cases",
        default="after_tms_pause,tms_active,tms_disable_mem_pool,tms_disable_nested_export_pool,tms_hook_only_disabled",
    )
    parser.add_argument("--require-tms-disable-pass", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    monkey_patch_torch_reductions()
    size_bytes = args.size_bytes if args.size_bytes is not None else args.size_mib * 1024 * 1024

    initial_state = _interesting_region()
    case_runners = {
        "after_tms_pause": lambda: _run_after_tms_pause(
            size_bytes,
            args.paused_allocation_mib * 1024 * 1024,
            args.repeats,
        ),
        "tms_active": lambda: _run_case("tms_active", size_bytes, _no_context),
        "tms_disable_mem_pool": lambda: _run_case(
            "tms_disable_mem_pool", size_bytes, torch_memory_saver.disable
        ),
        "tms_disable_nested_export_pool": lambda: _run_case(
            "tms_disable_nested_export_pool", size_bytes, _disable_tms_with_export_pool
        ),
        "tms_hook_only_disabled": lambda: _run_case(
            "tms_hook_only_disabled", size_bytes, _disable_tms_hook_only
        ),
    }
    selected_cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    unknown_cases = sorted(set(selected_cases) - set(case_runners))
    if unknown_cases:
        raise ValueError(f"Unknown cases: {unknown_cases}")
    results = [case_runners[case]() for case in selected_cases]
    report = {
        "schema": "openclaw.cuda-ipc-tms-probe/v1",
        "pid": os.getpid(),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "size_bytes": size_bytes,
        "initial_tms_interesting_region": initial_state,
        "results": results,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    exit_code = 0
    if args.require_tms_disable_pass:
        required_names = {"after_tms_pause", "tms_disable_mem_pool"}
        required = [item for item in results if item["name"] in required_names]
        exit_code = 0 if len(required) == len(required_names) and all(item["ok"] for item in required) else 1

    # This probe intentionally creates producer-only CUDA IPC handles. Avoid
    # PyTorch's process-exit warning/segfault for handles with no real consumer.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
