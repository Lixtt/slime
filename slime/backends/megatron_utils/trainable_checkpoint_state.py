import random
from argparse import Namespace
from typing import Any

import numpy as np
import torch


TRAINING_STATE_FORMAT = "slime_megatron_trainable_training_state_v1"


def capture_rank_local_rng_state(tensor_parallel: Any) -> dict:
    return {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }


def restore_rank_local_rng_state(state: dict, tensor_parallel: Any) -> None:
    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_state = state.get("cuda_rng_state")
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    tracker_states = state.get("rng_tracker_states")
    if not tracker_states:
        raise KeyError("rng_tracker_states")
    cuda_rng_tracker = tensor_parallel.get_cuda_rng_tracker()
    is_graph_safe = getattr(tensor_parallel, "is_graph_safe_cuda_rng_tracker", None)
    convert_state = getattr(tensor_parallel, "convert_cuda_rng_state", None)
    if is_graph_safe is not None and convert_state is not None:
        graph_safe_rng = is_graph_safe(cuda_rng_tracker)
        tracker_states = {
            key: convert_state(value, to_graphable=graph_safe_rng)
            for key, value in tracker_states.items()
        }
    cuda_rng_tracker.set_states(tracker_states)


def training_progress_state(args: Namespace) -> dict:
    fields = (
        "consumed_train_samples",
        "skipped_train_samples",
        "consumed_valid_samples",
        "num_floating_point_operations_so_far",
    )
    return {field: getattr(args, field) for field in fields if hasattr(args, field)}


def restore_training_progress_state(args: Namespace, progress: dict) -> None:
    if not isinstance(progress, dict):
        raise ValueError("trainable-only training progress must be a dict")
    for key, value in progress.items():
        setattr(args, key, value)


def build_training_state_payload(
    *,
    iteration: int,
    rank: int,
    world_size: int,
    topology: dict[str, int | None],
    args: Namespace,
    optimizer: Any,
    opt_param_scheduler: Any,
    tensor_parallel: Any,
) -> tuple[dict, dict[str, bool]]:
    save_optimizer = not getattr(args, "no_save_optim", False)
    optimizer_state = None
    if (
        save_optimizer
        and optimizer is not None
        and not getattr(optimizer, "is_stub_optimizer", False)
    ):
        optimizer_state = optimizer.state_dict()

    scheduler_state = None
    if save_optimizer and opt_param_scheduler is not None:
        scheduler_state = opt_param_scheduler.state_dict()

    rng_state = None
    if not getattr(args, "no_save_rng", False):
        rng_state = capture_rank_local_rng_state(tensor_parallel)

    payload = {
        "format": TRAINING_STATE_FORMAT,
        "iteration": iteration,
        "rank": rank,
        "world_size": world_size,
        **topology,
        "optimizer": optimizer_state,
        "opt_param_scheduler": scheduler_state,
        "rng_state": rng_state,
        "progress": training_progress_state(args),
    }
    metadata = {
        "optimizer_state_saved": optimizer_state is not None,
        "scheduler_state_saved": scheduler_state is not None,
        "rng_state_saved": rng_state is not None,
    }
    return payload, metadata


def restore_training_state_payload(
    payload: dict,
    *,
    expected_iteration: int,
    expected_rank: int,
    expected_world_size: int,
    args: Namespace,
    optimizer: Any,
    opt_param_scheduler: Any,
    tensor_parallel: Any,
    strict: bool,
    source: str,
) -> tuple[bool, list[str]]:
    if payload.get("format") != TRAINING_STATE_FORMAT:
        raise ValueError(
            f"Unsupported trainable-only training-state format in {source}: "
            f"{payload.get('format')}"
        )
    checks = {
        "iteration": expected_iteration,
        "rank": expected_rank,
        "world_size": expected_world_size,
    }
    for key, expected in checks.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(
                f"Training-state {key} mismatch in {source}: "
                f"checkpoint={payload.get(key)}, expected={expected}"
            )

    warnings: list[str] = []
    optimizer_state_loaded = False
    optimizer_state = payload.get("optimizer")
    if optimizer_state is None:
        if strict:
            raise KeyError(f"Training state {source} has no optimizer state")
        warnings.append(f"Training state {source} has no optimizer state")
    elif optimizer is None or getattr(optimizer, "is_stub_optimizer", False):
        raise RuntimeError(
            f"Training state {source} contains optimizer state, but the runtime optimizer is unavailable"
        )
    else:
        optimizer.load_state_dict(optimizer_state)
        optimizer_state_loaded = True

    scheduler_state = payload.get("opt_param_scheduler")
    if scheduler_state is None:
        if strict:
            raise KeyError(f"Training state {source} has no scheduler state")
        warnings.append(f"Training state {source} has no scheduler state")
    elif opt_param_scheduler is None:
        raise RuntimeError(
            f"Training state {source} contains scheduler state, but the runtime scheduler is unavailable"
        )
    else:
        opt_param_scheduler.load_state_dict(scheduler_state)

    rng_state = payload.get("rng_state")
    if rng_state is None:
        if strict:
            raise KeyError(f"Training state {source} has no RNG state")
        warnings.append(f"Training state {source} has no RNG state")
    else:
        restore_rank_local_rng_state(rng_state, tensor_parallel)

    restore_training_progress_state(args, payload.get("progress", {}))
    return optimizer_state_loaded, warnings
