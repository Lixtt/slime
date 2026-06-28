from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

logger = logging.getLogger(__name__)

_ADAPTER_FORMAT = "slime_megatron_lora_sharded_v1"


def _split_target_modules(target_modules: str | None) -> list[str]:
    if not target_modules:
        return []
    return [item.strip() for item in target_modules.split(",") if item.strip()]


def _target_matches(module_name: str, targets: Iterable[str]) -> bool:
    leaf_name = module_name.rsplit(".", 1)[-1]
    return any(target == leaf_name or target in module_name for target in targets)


def _is_supported_linear(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.nn.Parameter) and weight.ndim == 2


def _parallel_mode(module: torch.nn.Module) -> str | None:
    mode = getattr(module, "parallel_mode", None)
    if mode in {"column", "row", "duplicated"}:
        return mode

    class_name = type(module).__name__
    if class_name.endswith("ColumnParallelLinear"):
        return "column"
    if class_name.endswith("RowParallelLinear"):
        return "row"
    return mode


def _tp_group(module: torch.nn.Module):
    return getattr(module, "_tp_group", None) or getattr(module, "tp_group", None)


def _safe_mpu_value(getter_name: str, *args, default=0, **kwargs):
    getter = getattr(mpu, getter_name, None)
    if getter is None:
        return default
    try:
        return getter(*args, **kwargs)
    except Exception:
        return default


def _dist_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _dist_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _as_model_chunks(model: torch.nn.Module | Sequence[torch.nn.Module]) -> list[torch.nn.Module]:
    if isinstance(model, torch.nn.Module):
        return [model]
    return list(model)


def _set_lora_param_attrs(param: torch.nn.Parameter, *, average_across_tp: bool) -> None:
    # Megatron DDP should reduce LoRA grads across data parallel ranks. For
    # duplicated adapters, also average gradients across tensor parallel ranks.
    setattr(param, "allreduce", True)
    setattr(param, "tensor_model_parallel", False)
    if average_across_tp:
        setattr(param, "average_gradients_across_tp_domain", True)


def _lora_delta(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = module.slime_lora_dropout(x)  # type: ignore[attr-defined]
    delta = F.linear(x, module.slime_lora_A)  # type: ignore[attr-defined]
    delta = F.linear(delta, module.slime_lora_B)  # type: ignore[attr-defined]
    return delta * module.slime_lora_scaling  # type: ignore[attr-defined]


def _maybe_reduce_row_delta(module: torch.nn.Module, delta: torch.Tensor) -> torch.Tensor:
    if getattr(module, "explicit_expert_comm", False):
        return delta
    if getattr(module, "sequence_parallel", False):
        return reduce_scatter_to_sequence_parallel_region(delta, group=_tp_group(module))
    return reduce_from_tensor_model_parallel_region(delta, group=_tp_group(module))


def _maybe_gather_column_delta(module: torch.nn.Module, delta: torch.Tensor) -> torch.Tensor:
    if getattr(module, "gather_output", False):
        return gather_from_tensor_model_parallel_region(delta, group=_tp_group(module))
    return delta


def _lora_forward_hook(module: torch.nn.Module, inputs, output):
    if getattr(module, "slime_lora_merged", False):
        return output
    if not inputs:
        return output

    delta = _lora_delta(module, inputs[0])
    mode = _parallel_mode(module)
    if mode == "row":
        delta = _maybe_reduce_row_delta(module, delta)
    elif mode == "column":
        delta = _maybe_gather_column_delta(module, delta)

    if isinstance(output, tuple):
        if len(output) != 2:
            raise RuntimeError(f"Unsupported LoRA-wrapped output tuple length: {len(output)}")
        return output[0] + delta, output[1]
    return output + delta


def apply_megatron_lora(model: torch.nn.Module | Sequence[torch.nn.Module], args) -> int:
    """Attach Megatron-compatible LoRA adapters to matching linear modules."""

    targets = _split_target_modules(getattr(args, "lora_target_modules", None))
    if not targets:
        raise ValueError("--use-megatron-lora requires --lora-target-modules")

    rank = int(getattr(args, "lora_rank", 8))
    if rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {rank}")
    alpha = float(getattr(args, "lora_alpha", rank))
    dropout = float(getattr(args, "lora_dropout", 0.0))
    include_experts = bool(getattr(args, "megatron_lora_include_experts", False))

    for param in _iter_all_parameters(model):
        param.requires_grad_(False)

    patched = 0
    for name, module in _iter_named_modules(model):
        if not _target_matches(name, targets) or not _is_supported_linear(module):
            continue
        if ".experts." in name and not include_experts:
            continue
        if getattr(module, "slime_lora_enabled", False):
            continue

        weight = module.weight
        out_features, in_features = weight.shape
        lora_a = torch.nn.Parameter(torch.empty(rank, in_features, device=weight.device, dtype=weight.dtype))
        lora_b = torch.nn.Parameter(torch.zeros(out_features, rank, device=weight.device, dtype=weight.dtype))
        torch.nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))

        mode = _parallel_mode(module)
        average_across_tp = (
            mode in {None, "duplicated"} and _safe_mpu_value("get_tensor_model_parallel_world_size", default=1) > 1
        )
        _set_lora_param_attrs(lora_a, average_across_tp=average_across_tp)
        _set_lora_param_attrs(lora_b, average_across_tp=average_across_tp)

        module.register_parameter("slime_lora_A", lora_a)
        module.register_parameter("slime_lora_B", lora_b)
        module.slime_lora_scaling = alpha / rank
        module.slime_lora_dropout = torch.nn.Dropout(dropout)
        module.slime_lora_enabled = True
        module.slime_lora_merged = False
        module.register_forward_hook(_lora_forward_hook)
        patched += 1

    if patched == 0:
        raise ValueError(f"No Megatron modules matched --lora-target-modules={','.join(targets)}")
    logger.info("Applied Megatron LoRA to %s module(s): %s", patched, ",".join(targets))
    return patched


def _iter_all_parameters(model: torch.nn.Module | Sequence[torch.nn.Module]):
    for model_chunk in _as_model_chunks(model):
        yield from model_chunk.parameters()


def _iter_named_modules(model: torch.nn.Module | Sequence[torch.nn.Module]):
    for chunk_id, model_chunk in enumerate(_as_model_chunks(model)):
        prefix = f"chunks.{chunk_id}"
        for name, module in model_chunk.named_modules():
            yield f"{prefix}.{name}" if name else prefix, module


def has_megatron_lora(model: torch.nn.Module | Sequence[torch.nn.Module]) -> bool:
    return any(getattr(module, "slime_lora_enabled", False) for _name, module in _iter_named_modules(model))


def iter_lora_modules(model: torch.nn.Module | Sequence[torch.nn.Module]):
    for name, module in _iter_named_modules(model):
        if getattr(module, "slime_lora_enabled", False):
            yield name, module


def _local_delta_weight(module: torch.nn.Module) -> torch.Tensor:
    return (module.slime_lora_B @ module.slime_lora_A) * module.slime_lora_scaling


@contextmanager
def merged_megatron_lora(model: torch.nn.Module | Sequence[torch.nn.Module]):
    """Temporarily fold LoRA deltas into local base weight shards."""

    merged_modules = []
    with torch.no_grad():
        for _name, module in iter_lora_modules(model):
            if getattr(module, "slime_lora_merged", False):
                continue
            module.weight.add_(_local_delta_weight(module).to(dtype=module.weight.dtype))
            module.slime_lora_merged = True
            merged_modules.append(module)
    try:
        yield
    finally:
        with torch.no_grad():
            for module in reversed(merged_modules):
                module.weight.sub_(_local_delta_weight(module).to(dtype=module.weight.dtype))
                module.slime_lora_merged = False


def _adapter_shard_name() -> str:
    return "adapter_tp{tp:02d}_pp{pp:02d}_cp{cp:02d}_ep{ep:02d}.pt".format(
        tp=_safe_mpu_value("get_tensor_model_parallel_rank"),
        pp=_safe_mpu_value("get_pipeline_model_parallel_rank"),
        cp=_safe_mpu_value("get_context_parallel_rank"),
        ep=_safe_mpu_value("get_expert_model_parallel_rank"),
    )


def _adapter_shard_path(model_dir: Path) -> Path:
    return model_dir / _adapter_shard_name()


def _is_adapter_writer_rank() -> bool:
    try:
        return mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    except Exception:
        return _dist_rank_world()[0] == 0


def _resolve_adapter_path(path: str | None) -> Path | None:
    if not path:
        return None

    adapter_path = Path(path).expanduser()
    if adapter_path.is_file():
        return adapter_path

    if adapter_path.is_dir():
        for marker in (
            "latest_megatron_lora_iteration.txt",
            "latest_checkpointed_iteration.txt",
            "latest_trainable_iteration.txt",
        ):
            marker_path = adapter_path / marker
            if marker_path.is_file():
                step = int(marker_path.read_text(encoding="utf-8").strip())
                adapter_path = adapter_path / f"iter_{step:07d}"
                break

    if adapter_path.is_dir() and (adapter_path / "model").is_dir():
        adapter_path = adapter_path / "model"

    if adapter_path.is_dir():
        shard_path = _adapter_shard_path(adapter_path)
        if shard_path.is_file():
            return shard_path
        legacy_path = adapter_path / "adapter_weights.pt"
        if legacy_path.is_file():
            return legacy_path
        return shard_path

    return adapter_path


def save_megatron_lora_checkpoint(model: Sequence[torch.nn.Module], args, iteration: int) -> None:
    if not getattr(args, "save", None):
        return

    rank, world_size = _dist_rank_world()
    save_root = Path(args.save).expanduser()
    checkpoint_dir = save_root / f"iter_{iteration:07d}"
    model_dir = checkpoint_dir / "model"
    if _is_adapter_writer_rank():
        adapter_state = {}
        for module_name, module in iter_lora_modules(model):
            adapter_state[f"{module_name}.slime_lora_A"] = module.slime_lora_A.detach().cpu()
            adapter_state[f"{module_name}.slime_lora_B"] = module.slime_lora_B.detach().cpu()
        _atomic_torch_save(adapter_state, _adapter_shard_path(model_dir))
        del adapter_state

    if rank == 0:
        metadata = {
            "format": _ADAPTER_FORMAT,
            "iteration": iteration,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "world_size": world_size,
            "requires_full_base_checkpoint": True,
            "base_load": getattr(args, "load", None),
            "base_hf_checkpoint": getattr(args, "hf_checkpoint", None),
            "optimizer_state_saved": False,
            "lora_rank": getattr(args, "lora_rank", None),
            "lora_alpha": getattr(args, "lora_alpha", None),
            "lora_dropout": getattr(args, "lora_dropout", None),
            "lora_target_modules": getattr(args, "lora_target_modules", None),
            "megatron_lora_include_experts": getattr(args, "megatron_lora_include_experts", None),
            "tensor_model_parallel_size": _safe_mpu_value("get_tensor_model_parallel_world_size", default=1),
            "pipeline_model_parallel_size": _safe_mpu_value("get_pipeline_model_parallel_world_size", default=1),
            "context_parallel_size": _safe_mpu_value("get_context_parallel_world_size", default=1),
            "expert_model_parallel_size": _safe_mpu_value("get_expert_model_parallel_world_size", default=1),
            "created_by": "slime.megatron_lora",
        }
        _atomic_write_text(checkpoint_dir / "meta.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        _atomic_write_text(save_root / "latest_megatron_lora_iteration.txt", f"{iteration}\n")
        # Keep the conventional tracker so existing monitor scripts can detect
        # that a checkpoint was produced, even though this is adapter-only.
        _atomic_write_text(save_root / "latest_checkpointed_iteration.txt", f"{iteration}\n")
        logger.info("Saved Megatron LoRA adapter checkpoint iteration=%s to %s", iteration, model_dir)
    _dist_barrier()


@torch.no_grad()
def load_megatron_lora_checkpoint(model: Sequence[torch.nn.Module], path: str | None) -> int | None:
    adapter_path = _resolve_adapter_path(path)
    if adapter_path is None:
        return None
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Megatron LoRA adapter checkpoint not found: {adapter_path}")

    state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    missing = []
    for module_name, module in iter_lora_modules(model):
        for suffix, param in (("slime_lora_A", module.slime_lora_A), ("slime_lora_B", module.slime_lora_B)):
            key = f"{module_name}.{suffix}"
            if key not in state:
                missing.append(key)
                continue
            tensor = state[key]
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint={tuple(tensor.shape)}, runtime={tuple(param.shape)}"
                )
            param.copy_(tensor.to(device=param.device, dtype=param.dtype, non_blocking=True))
    if missing:
        raise RuntimeError(f"Megatron LoRA adapter checkpoint missing keys: {missing[:8]}")

    iteration = None
    checkpoint_dir = adapter_path.parent.parent if adapter_path.parent.name == "model" else adapter_path.parent
    metadata_path = checkpoint_dir / "meta.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") not in {_ADAPTER_FORMAT, "megatron_lora_sharded"}:
            raise ValueError(f"Unsupported Megatron LoRA adapter format in {metadata_path}: {metadata.get('format')}")
        if metadata.get("iteration") is not None:
            iteration = int(metadata["iteration"])

    logger.info("Loaded Megatron LoRA adapter checkpoint from %s", adapter_path)
    return iteration
