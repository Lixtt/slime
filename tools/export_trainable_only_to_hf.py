import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.backends.megatron_utils.initialize import init as init_megatron
from slime.backends.megatron_utils.model import initialize_model_and_optimizer
from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path
from slime.utils.arguments import parse_args
from slime.utils.distributed_utils import init_gloo_group


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
        "--output-dir",
        required=True,
        help="Directory to write the merged HuggingFace checkpoint.",
    )
    return parser


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


def main() -> None:
    args = parse_args(add_custom_arguments=add_export_args)

    # The trainable-only loader is deliberately env-gated inside the Megatron
    # utility layer so normal training and full checkpoint loading are unchanged.
    import os

    os.environ["SLIME_MEGATRON_TRAINABLE_ONLY_LOAD"] = args.trainable_only_checkpoint

    args.no_load_optim = True
    args.no_load_rng = True
    args.save = args.save or str(Path(args.output_dir).with_suffix(".dummy_megatron_save"))

    _init_torch_distributed(args)
    init_megatron(args)
    model, optimizer, _, _ = initialize_model_and_optimizer(args)
    if optimizer is not None:
        del optimizer

    save_hf_model_to_path(
        args,
        Path(args.output_dir),
        model,
        progress_desc="Export trainable-only checkpoint to HF",
    )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if mpu.is_initialized():
        mpu.destroy_model_parallel()


if __name__ == "__main__":
    main()
