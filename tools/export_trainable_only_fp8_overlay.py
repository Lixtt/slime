import argparse
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from slime.backends.megatron_utils.initialize import init as init_megatron
from slime.backends.megatron_utils.model import initialize_model_and_optimizer
from slime.backends.megatron_utils.hf_checkpoint_saver import _copy_hf_assets
from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers
from slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect
from slime.utils.arguments import parse_args
from slime.utils.distributed_utils import init_gloo_group


_DEFAULT_TRAINABLE_NAME_SUBSTRINGS = (
    "self_attention.linear_q_down_proj.weight",
    "self_attention.linear_kv_down_proj.weight",
)


def add_export_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--trainable-only-checkpoint",
        required=True,
        help=(
            "Path to a trainable-only checkpoint iteration directory, or to a "
            "checkpoint root containing latest_trainable_iteration.txt."
        ),
    )
    parser.add_argument(
        "--base-fp8-hf-checkpoint",
        required=True,
        help="Existing FP8 HuggingFace checkpoint directory used as the overlay base.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the FP8 overlay checkpoint.",
    )
    parser.add_argument(
        "--include-megatron-name-substring",
        action="append",
        default=[],
        help=(
            "Megatron global parameter-name substring to export into the overlay. "
            "Defaults to q/down and kv/down projection weights."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output directory before writing the overlay.",
    )
    return parser


def _rank_zero() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _init_torch_distributed(args) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    args.local_rank = local_rank
    init_gloo_group()


def _read_index(base_dir: Path) -> dict[str, Any]:
    index_path = base_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing FP8 HF index: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _prepare_output_dir(base_dir: Path, output_dir: Path, *, force: bool) -> None:
    if output_dir.resolve() == base_dir.resolve():
        raise ValueError("--output-dir must not be the same as --base-fp8-hf-checkpoint")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"OUTPUT_DIR already exists: {output_dir}; pass --force after inspection")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _copy_hf_assets(str(base_dir), output_dir)


def _filter_iterator_to_trainable_params(iterator: HfWeightIteratorDirect, substrings: tuple[str, ...]) -> None:
    filtered_buckets = []
    for bucket in iterator.megatron_local_param_info_buckets:
        selected = [info for info in bucket if any(token in info.name for token in substrings)]
        if selected:
            filtered_buckets.append(selected)
    iterator.megatron_local_param_info_buckets = filtered_buckets


def _collect_overlay_updates(args, model) -> dict[str, Any]:
    hf_config = AutoConfig.from_pretrained(args.base_fp8_hf_checkpoint, trust_remote_code=True)
    model_name = type(hf_config).__name__.lower() if args.model_name is None else args.model_name
    quantization_config = getattr(hf_config, "quantization_config", None)
    if not isinstance(quantization_config, dict) or quantization_config.get("quant_method") != "fp8":
        raise ValueError(f"--base-fp8-hf-checkpoint must have FP8 quantization_config: {args.base_fp8_hf_checkpoint}")

    iterator = HfWeightIteratorDirect(
        args=args,
        model=model,
        model_name=model_name,
        quantization_config=quantization_config,
    )
    include_substrings = tuple(args.include_megatron_name_substring) or _DEFAULT_TRAINABLE_NAME_SUBSTRINGS
    _filter_iterator_to_trainable_params(iterator, include_substrings)
    if not iterator.megatron_local_param_info_buckets:
        raise ValueError(
            "No Megatron parameters matched overlay include substrings: "
            + ", ".join(include_substrings)
        )

    local_weights = dict(named_params_and_buffers(args, model, convert_to_global_name=True))
    updates = {}
    for hf_named_tensors in iterator.get_hf_weight_chunks(
        local_weights,
        progress_desc="Export trainable-only FP8 overlay tensors",
    ):
        if _rank_zero():
            for name, tensor in hf_named_tensors:
                if name in updates:
                    raise ValueError(f"Duplicate HF tensor while collecting overlay updates: {name}")
                updates[name] = tensor.detach().cpu().contiguous()
        del hf_named_tensors

    return updates


def _symlink_or_copy_base_file(base_file: Path, output_file: Path) -> None:
    try:
        os.symlink(base_file.resolve(), output_file)
    except OSError:
        shutil.copy2(base_file, output_file)


def _write_overlay_checkpoint(
    base_dir: Path,
    output_dir: Path,
    updates: dict[str, Any],
    *,
    force: bool,
    trainable_only_checkpoint: str,
) -> None:
    if not updates:
        raise ValueError("No HF tensors were produced for the FP8 overlay")

    index_data = _read_index(base_dir)
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid HF index weight_map in {base_dir / 'model.safetensors.index.json'}")

    missing = sorted(set(updates) - set(weight_map))
    if missing:
        preview = ", ".join(missing[:16])
        raise KeyError(f"{len(missing)} overlay tensors are missing from the FP8 base index: {preview}")

    _prepare_output_dir(base_dir, output_dir, force=force)
    affected_files = {weight_map[name] for name in updates}
    all_weight_files = sorted(set(weight_map.values()))

    for filename in all_weight_files:
        base_file = base_dir / filename
        output_file = output_dir / filename
        if filename not in affected_files:
            _symlink_or_copy_base_file(base_file, output_file)
            continue

        state = load_file(base_file, device="cpu")
        replaced = 0
        for name, tensor in updates.items():
            if weight_map[name] != filename:
                continue
            if name not in state:
                raise KeyError(f"Tensor {name} is mapped to {filename} but is absent from the base shard")
            if tuple(state[name].shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: base={tuple(state[name].shape)} overlay={tuple(tensor.shape)}"
                )
            state[name] = tensor.to(dtype=state[name].dtype)
            replaced += 1
        if replaced == 0:
            raise RuntimeError(f"Internal error: affected shard {filename} had no replacements")
        save_file(state, output_file, metadata={"format": "pt"})

    shutil.copy2(base_dir / "model.safetensors.index.json", output_dir / "model.safetensors.index.json")
    manifest = {
        "format": "slime_fp8_overlay_v1",
        "base_fp8_hf_checkpoint": str(base_dir),
        "output_dir": str(output_dir),
        "trainable_only_checkpoint": trainable_only_checkpoint,
        "updated_tensor_count": len(updates),
        "affected_shard_count": len(affected_files),
        "linked_or_copied_shard_count": len(all_weight_files) - len(affected_files),
        "affected_shards": sorted(affected_files),
        "updated_tensors": sorted(updates),
    }
    (output_dir / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args(add_custom_arguments=add_export_args)

    os.environ["SLIME_MEGATRON_TRAINABLE_ONLY_LOAD"] = args.trainable_only_checkpoint
    args.no_load_optim = True
    args.no_load_rng = True
    args.save = args.save or str(Path(args.output_dir).with_suffix(".dummy_megatron_save"))

    _init_torch_distributed(args)
    init_megatron(args)
    model, optimizer, _, _ = initialize_model_and_optimizer(args)
    if optimizer is not None:
        del optimizer

    updates = _collect_overlay_updates(args, model)
    if _rank_zero():
        _write_overlay_checkpoint(
            Path(args.base_fp8_hf_checkpoint),
            Path(args.output_dir),
            updates,
            force=args.force,
            trainable_only_checkpoint=args.trainable_only_checkpoint,
        )

    _barrier()
    if mpu.is_initialized():
        mpu.destroy_model_parallel()


if __name__ == "__main__":
    main()
