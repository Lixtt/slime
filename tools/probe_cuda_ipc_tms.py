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


def _serialize_bucket_payload(size_bytes: int) -> str:
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
    return MultiprocessingSerializer.serialize(
        {"flattened_tensor": flattened, "metadata": bucket.get_metadata()},
        output_str=True,
    )


def _serialize_bucket(size_bytes: int) -> int:
    return len(_serialize_bucket_payload(size_bytes))


def _ipc_consumer(connection, force_gc: bool) -> None:
    try:
        monkey_patch_torch_reductions()
        while True:
            payload = connection.recv()
            if payload is None:
                connection.send({"ok": True, "stopped": True})
                return

            tensor_data = MultiprocessingSerializer.deserialize(payload)
            flattened = tensor_data["flattened_tensor"]
            if flattened.is_cuda:
                torch.cuda.synchronize(flattened.device)
            del flattened, tensor_data, payload

            from torch.multiprocessing import reductions

            reductions.shared_cache.clear()
            if force_gc:
                gc.collect()
            torch.cuda.ipc_collect()
            connection.send({"ok": True})
    except BaseException as exc:
        connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        connection.close()


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


def _memory_snapshot(iteration: int) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "iteration": iteration,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def _run_after_tms_pause(
    size_bytes: int,
    paused_allocation_bytes: int,
    repeats: int,
    memory_sample_interval: int,
    consumer_force_gc: bool,
):
    resident = []
    paused = False
    consumer = None
    producer_connection = None
    try:
        context = torch.multiprocessing.get_context("spawn")
        producer_connection, consumer_connection = context.Pipe()
        consumer = context.Process(
            target=_ipc_consumer,
            args=(consumer_connection, consumer_force_gc),
        )
        consumer.start()
        consumer_connection.close()

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
            memory_samples = [_memory_snapshot(0)]
            for iteration in range(1, repeats + 1):
                payload = _serialize_bucket_payload(size_bytes)
                payload_bytes.append(len(payload))
                producer_connection.send(payload)
                if not producer_connection.poll(120):
                    raise TimeoutError(f"CUDA IPC consumer timed out at iteration {iteration}")
                consumer_result = producer_connection.recv()
                if not consumer_result.get("ok"):
                    raise RuntimeError(
                        "CUDA IPC consumer failed at iteration "
                        f"{iteration}: {consumer_result.get('error_type')}: {consumer_result.get('error')}"
                    )
                del payload, consumer_result
                torch.cuda.ipc_collect()
                if iteration % memory_sample_interval == 0 or iteration == repeats:
                    memory_samples.append(_memory_snapshot(iteration))
        min_free_bytes = min(sample["free_bytes"] for sample in memory_samples)
        free_memory_loss_bytes = max(0, memory_samples[0]["free_bytes"] - min_free_bytes)
        producer_connection.send(None)
        if not producer_connection.poll(30):
            raise TimeoutError("CUDA IPC consumer did not stop cleanly")
        stop_result = producer_connection.recv()
        if not stop_result.get("ok"):
            raise RuntimeError(f"CUDA IPC consumer stop failed: {stop_result}")
        consumer.join(timeout=30)
        if consumer.is_alive() or consumer.exitcode != 0:
            raise RuntimeError(f"CUDA IPC consumer exitcode={consumer.exitcode}")
        consumer_exitcode = consumer.exitcode
        producer_connection.close()
        producer_connection = None
        return {
            "name": "after_tms_pause",
            "ok": True,
            "tms_interesting_region": state_during,
            "paused_allocation_bytes": paused_allocation_bytes,
            "repeats": repeats,
            "payload_chars": payload_bytes,
            "memory_samples": memory_samples,
            "free_memory_loss_bytes": free_memory_loss_bytes,
            "consumer_exitcode": consumer_exitcode,
            "consumer_force_gc": consumer_force_gc,
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
        if producer_connection is not None:
            try:
                if consumer is not None and consumer.is_alive():
                    producer_connection.send(None)
                    if producer_connection.poll(30):
                        producer_connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            producer_connection.close()
        if consumer is not None:
            consumer.join(timeout=30)
            if consumer.is_alive():
                consumer.terminate()
                consumer.join(timeout=30)
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
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--memory-sample-interval", type=int, default=16)
    parser.add_argument("--max-free-memory-loss-mib", type=int, default=1024)
    parser.add_argument("--consumer-force-gc", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--cases",
        default="after_tms_pause,tms_active,tms_disable_mem_pool,tms_disable_nested_export_pool,tms_hook_only_disabled",
    )
    parser.add_argument("--require-tms-disable-pass", action="store_true")
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.memory_sample_interval <= 0:
        raise ValueError("--memory-sample-interval must be positive")
    if args.max_free_memory_loss_mib < 0:
        raise ValueError("--max-free-memory-loss-mib must be non-negative")

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
            args.memory_sample_interval,
            args.consumer_force_gc,
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
    if not initial_state:
        raise RuntimeError(
            "probe requires the production actor TMS lifecycle; set TMS_INIT_ENABLE=1"
        )
    active_state = initial_state
    results = [case_runners[case]() for case in selected_cases]
    report = {
        "schema": "openclaw.cuda-ipc-tms-probe/v1",
        "pid": os.getpid(),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "size_bytes": size_bytes,
        "initial_tms_interesting_region": initial_state,
        "active_tms_interesting_region": active_state,
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
        paused_result = next((item for item in required if item["name"] == "after_tms_pause"), None)
        memory_bounded = (
            paused_result is not None
            and paused_result.get("free_memory_loss_bytes", 0)
            <= args.max_free_memory_loss_mib * 1024 * 1024
        )
        exit_code = (
            0
            if len(required) == len(required_names)
            and all(item["ok"] for item in required)
            and memory_bounded
            else 1
        )

    # Repeated CUDA IPC setup/teardown can trip PyTorch's process-exit cleanup
    # after all measurements are already durable. Exit without re-running it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
