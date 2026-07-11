import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRAINING_STATE_FORMAT = "slime_megatron_trainable_training_state_v2"


def save_optimizer_parameter_state_atomically(
    optimizer: Any,
    destination: str | Path,
    *,
    writer_rank: bool,
) -> None:
    """Save Megatron distributed-optimizer tensors without publishing a partial file.

    Every rank in the optimizer's data-parallel group must call this function.
    Megatron writes only on data-parallel rank zero, represented by
    ``writer_rank`` here.
    """

    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if writer_rank:
        temporary.unlink(missing_ok=True)
    try:
        optimizer.save_parameter_state(str(temporary))
        if not writer_rank:
            return
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(
                f"Megatron did not write distributed optimizer parameter state to {temporary}"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            parent_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            pass
    finally:
        if writer_rank:
            temporary.unlink(missing_ok=True)


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


def prepare_scheduler_for_exact_restore(
    args: Namespace,
    opt_param_scheduler: Any,
    *,
    source: str,
) -> None:
    """Make the persisted scheduler authoritative for an exact resume."""
    override_requested = bool(
        getattr(args, "override_opt_param_scheduler", False)
        or getattr(opt_param_scheduler, "override_opt_param_scheduler", False)
    )
    if override_requested:
        raise ValueError(
            f"Strict training-state restore from {source} cannot override the checkpoint scheduler; "
            "disable strict restore for an intentional new-schedule branch"
        )

    if hasattr(args, "use_checkpoint_opt_param_scheduler"):
        args.use_checkpoint_opt_param_scheduler = True
    if hasattr(opt_param_scheduler, "use_checkpoint_opt_param_scheduler"):
        opt_param_scheduler.use_checkpoint_opt_param_scheduler = True


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
    optimizer_parameter_state_required: bool = False,
    optimizer_parameter_state_file: str | None = None,
    optimizer_parameter_state_saved: bool = False,
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

    optimizer_common_state_saved = optimizer_state is not None
    optimizer_state_saved = optimizer_common_state_saved and (
        not optimizer_parameter_state_required or optimizer_parameter_state_saved
    )

    payload = {
        "format": TRAINING_STATE_FORMAT,
        "iteration": iteration,
        "rank": rank,
        "world_size": world_size,
        **topology,
        "optimizer": optimizer_state,
        "optimizer_parameter_state": {
            "required": optimizer_parameter_state_required,
            "file": optimizer_parameter_state_file,
            "saved": optimizer_parameter_state_saved,
        },
        "opt_param_scheduler": scheduler_state,
        "rng_state": rng_state,
        "progress": training_progress_state(args),
    }
    metadata = {
        "optimizer_state_saved": optimizer_state_saved,
        "optimizer_common_state_saved": optimizer_common_state_saved,
        "optimizer_parameter_state_required": optimizer_parameter_state_required,
        "optimizer_parameter_state_saved": optimizer_parameter_state_saved,
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
    optimizer_parameter_state_path: str | Path | None = None,
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
    parameter_state = payload.get("optimizer_parameter_state")
    if not isinstance(parameter_state, dict):
        raise ValueError(
            f"Training state {source} has no distributed optimizer parameter-state metadata"
        )
    parameter_state_required = bool(parameter_state.get("required"))
    parameter_state_saved = bool(parameter_state.get("saved"))
    parameter_state_path = (
        Path(optimizer_parameter_state_path)
        if optimizer_parameter_state_path is not None
        else None
    )

    optimizer_error: Exception | None = None
    if optimizer_state is None:
        optimizer_error = KeyError(f"Training state {source} has no optimizer common state")
    elif optimizer is None or getattr(optimizer, "is_stub_optimizer", False):
        raise RuntimeError(
            f"Training state {source} contains optimizer state, but the runtime optimizer is unavailable"
        )
    elif parameter_state_required and not parameter_state_saved:
        optimizer_error = KeyError(
            f"Training state {source} has no distributed optimizer parameter state"
        )
    elif parameter_state_required and (
        parameter_state_path is None or not parameter_state_path.is_file()
    ):
        optimizer_error = FileNotFoundError(
            "Missing distributed optimizer parameter state for "
            f"{source}: {parameter_state_path}"
        )
    elif parameter_state_required and not hasattr(optimizer, "load_parameter_state"):
        optimizer_error = RuntimeError(
            "Training state requires distributed optimizer parameter state, but "
            "the runtime optimizer has no load_parameter_state()"
        )

    if optimizer_error is not None:
        if strict:
            raise optimizer_error
        warnings.append(str(optimizer_error))
    else:
        optimizer.load_state_dict(optimizer_state)
        if parameter_state_required:
            optimizer.load_parameter_state(str(parameter_state_path))
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
        if strict:
            prepare_scheduler_for_exact_restore(
                args,
                opt_param_scheduler,
                source=source,
            )
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
