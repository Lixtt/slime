import dataclasses
import gc
import json
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model
from tqdm import tqdm

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model
from slime.utils import logging_utils
from slime.utils.atomic_io import atomic_torch_save
from slime.utils.memory_utils import clear_memory

from .checkpoint import load_checkpoint, save_checkpoint
from .cp_utils import reduce_train_step_metrics
from .data import DataIterator, get_batch
from .lora import load_megatron_lora_checkpoint, save_megatron_lora_checkpoint
from .loss import ROLLOUT_TOP_P_TOKEN_KEYS, get_rollout_top_p_logprob_kwargs, loss_function
from .model_provider import get_model_provider_func
from .trainable_checkpoint_state import (
    TRAINING_STATE_FORMAT,
    build_training_state_payload,
    restore_training_state_payload,
)

logger = logging.getLogger(__name__)


def _disable_tqdm_for_non_main_rank() -> bool:
    return not (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )


def _should_update_microbatch_pbar(model) -> bool:
    if _disable_tqdm_for_non_main_rank():
        return False

    while hasattr(model, "module"):
        model = model.module
    vp_stage = getattr(model, "vp_stage", None)
    if mpu.get_virtual_pipeline_model_parallel_world_size() is not None and vp_stage is not None:
        return mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    return mpu.is_pipeline_last_stage(ignore_virtual=True)


def _wrap_forward_step_with_microbatch_pbar(forward_step_func, pbar):
    if pbar is None:
        return forward_step_func

    def wrapped_forward_step(*args, **kwargs):
        result = forward_step_func(*args, **kwargs)
        model = args[1] if len(args) > 1 else kwargs.get("model")
        if model is not None and _should_update_microbatch_pbar(model):
            pbar.update(1)
        return result

    return wrapped_forward_step


def _with_rollout_top_p_token_keys(args: Namespace, keys: Sequence[str]) -> list[str]:
    if args.rollout_top_p == 1.0:
        return list(keys)
    return [*keys, *ROLLOUT_TOP_P_TOKEN_KEYS]


def _iter_critic_output_layers(model: Sequence[DDP]):
    for chunk_id, module in enumerate(unwrap_model(model)):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is not None:
            yield chunk_id, output_layer


def _critic_output_layer_needs_reinit(args: Namespace, model: Sequence[DDP], role: str) -> bool:
    if role != "critic" or args.load is None:
        return False

    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
    from megatron.training.checkpointing import get_load_checkpoint_path_by_args

    checkpoint_path = Path(get_load_checkpoint_path_by_args(args))
    if not (checkpoint_path / ".metadata").is_file():
        return False

    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        for name in ("weight", "bias"):
            param = getattr(output_layer, name, None)
            if param is None:
                continue

            param_name = f"output_layer.{name}"
            ckpt_tensor_metadata = next(
                (
                    tensor_metadata
                    for key, tensor_metadata in checkpoint_metadata.items()
                    if key == param_name or key.endswith(f".{param_name}")
                ),
                None,
            )
            expected_shape = tuple(param.shape)
            checkpoint_shape = tuple(ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata is not None else None
            if checkpoint_shape == expected_shape:
                continue

            reason = (
                "missing from checkpoint metadata"
                if checkpoint_shape is None
                else f"shape mismatch checkpoint={checkpoint_shape} runtime={expected_shape}"
            )
            logger.warning(
                "Will reinitialize critic %s after checkpoint load because it is %s",
                param_name,
                reason,
            )
            return True

    return False


@torch.no_grad()
def _reinitialize_critic_output_layer(model: Sequence[DDP]) -> None:
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        output_layer.weight.data.normal_(mean=0.0, std=0.02)
        if output_layer.bias is not None:
            output_layer.bias.data.zero_()


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training. ``train_iters`` is an estimate of the total
    # number of training steps — it's only used to size Megatron's LR decay
    # schedule (and ``lr_decay_iters`` defaults to it). With variable per-rollout
    # sample counts (dynamic sampling / filtering / custom step splitter) the
    # *actual* total can drift; the schedule still tracks the true progress via
    # ``opt_param_scheduler.num_steps`` (samples consumed, also persisted across
    # resume), so the worst case is the cosine/linear schedule reaches its
    # plateau slightly early or late. Pass ``--lr-decay-iters`` explicitly if you
    # need exact decay control.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
    use_rollout_top_p_replay: bool = False,
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.
        use_rollout_top_p_replay (bool): Whether to pass rollout top-p token sets
            to the post-forward log-prob callback when top-p rollout is enabled.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])
    batch_keys = [
        "tokens",
        "loss_masks",
        "multimodal_train_inputs",
        "total_lengths",
        "response_lengths",
    ]
    if use_rollout_top_p_replay:
        batch_keys = _with_rollout_top_p_token_keys(args, batch_keys)

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        batch = get_batch(
            data_iterator,
            batch_keys,
            args.data_pad_size_multiplier,
            args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        forward_kwargs = {
            "input_ids": tokens,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": packed_seq_params,
            "loss_mask": batch["full_loss_masks"],
        }
        if batch["multimodal_train_inputs"] is not None:
            forward_kwargs.update(batch["multimodal_train_inputs"])
        output_tensor = model(**forward_kwargs)

        output_kwargs = {
            "args": args,
            "unconcat_tokens": unconcat_tokens,
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "with_entropy": args.use_rollout_entropy,
        }
        if use_rollout_top_p_replay:
            output_kwargs.update(get_rollout_top_p_logprob_kwargs(args, batch))

        return output_tensor, partial(f, **output_kwargs)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from slime.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{(store_prefix or getattr(model[0], 'role', 'actor')).rstrip('_')} forward",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )
    forward_step_with_progress = _wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar)
    for step_id in range(num_steps_per_rollout):
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step_with_progress,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
        )
    microbatch_pbar.close()

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                origin_values = [None] * len(values)
                origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                for value, origin_index in zip(values, origin_indices, strict=False):
                    origin_values[origin_index] = value
                values = origin_values
            rollout_data[f"{store_prefix}{key}"] = values

    # Release references to intermediate forward tensors before returning.
    del forward_data_store
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    step_global_batch_size: int,
    microbatch_pbar=None,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.
        step_global_batch_size (int): Rollout count for this training step
            (total across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts, which may emit >1 training
            sample under compact / subagent). Used both as the loss
            normalizer inside the closure and as the LR scheduler
            ``increment``. In the common case (1 rollout = 1 sample) this
            equals the per-step sample count, so behavior is unchanged.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    args = get_args()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            _with_rollout_top_p_token_keys(
                args,
                [
                    "tokens",
                    "multimodal_train_inputs",
                    "packed_seq_params",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "log_probs",
                    "ref_log_probs",
                    "values",
                    "advantages",
                    "returns",
                    "rollout_log_probs",
                    "teacher_log_probs",
                    "rollout_mask_sums",
                ],
            ),
            args.data_pad_size_multiplier,
            args.allgather_cp,
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            position_ids = None
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=position_ids,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=batch["full_loss_masks"],
            )
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }

            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            output_tensor = model(**forward_kwargs)

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(loss_function, args, batch, num_microbatches, step_global_batch_size)

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    grad_norm = float("nan")
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training:
        from slime.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate. Use the per-step global_batch_size when dynamic
        # batching is on so the scheduler's samples-seen counter tracks reality.
        assert update_successful
        opt_param_scheduler.step(increment=step_global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = reduce_train_step_metrics(
            losses_reduced,
            calculate_per_token_loss=args.calculate_per_token_loss,
            step_global_batch_size=step_global_batch_size,
            cp_size=mpu.get_context_parallel_world_size(),
            dp_with_cp_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    global_batch_sizes: Sequence[int],
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
        global_batch_sizes (Sequence[int]): Rollout count per step (total
            across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts of a prompt). Same length as
            ``num_microbatches``; consumed by ``train_one_step`` for loss
            scaling and LR scheduler increments. Equals per-step sample count
            in the common case (1 rollout = 1 sample).
    """
    args = get_args()

    assert len(num_microbatches) == len(global_batch_sizes), (
        f"num_microbatches and global_batch_sizes must have the same length, "
        f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
    )

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            print("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "step" in state:
                    if isinstance(state["step"], torch.Tensor):
                        state["step"].zero_()
                    else:
                        state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{getattr(model[0], 'role', 'actor')} train",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            global_batch_sizes[step_id],
            microbatch_pbar=microbatch_pbar,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from slime.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            log_dict = {
                f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }
            log_dict[f"train/{role_tag}grad_norm"] = grad_norm
            if args.enable_mtp_training:
                log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses

            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            # Per-step gbs — uneven step sizes are easy to miss without this.
            log_dict[f"train/{role_tag}global_batch_size"] = global_batch_sizes[step_id]
            log_dict["train/step"] = accumulated_step_id
            logging_utils.log(args, log_dict, step_key="train/step")

            if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
                assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1, f"{log_dict=}"

            if args.ci_test and not args.ci_disable_kl_checker:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    # TODO: figure out why KL is not exactly zero when using PPO loss with KL clipping, and whether this is expected behavior or a bug.
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                # R3 replays rollout routing for the actor path, while ref
                # log-probs are computed with normal routing. The initial
                # actor/ref KL is therefore not expected to be exactly zero.
                if (
                    accumulated_step_id == 0
                    and not getattr(args, "use_rollout_routing_replay", False)
                    and "train/kl_loss" in log_dict
                ):
                    assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_save_grad_norm is not None:
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
    microbatch_pbar.close()
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def _drop_rank_local_common_checkpoint_state(common_state_dict):
    filtered = dict(common_state_dict)
    filtered.pop("args", None)
    return filtered


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _dist_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _safe_mpu_value(getter_name: str, *args, **kwargs):
    getter = getattr(mpu, getter_name, None)
    if getter is None:
        return None
    try:
        return getter(*args, **kwargs)
    except Exception:
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _save_trainable_only_checkpoint(
    iteration: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: OptimizerParamScheduler | None,
) -> None:
    """Save only rank-local trainable parameter shards.

    This bypasses Megatron's full distributed checkpoint metadata path, which is
    too memory-heavy for GLM5.2 744B resource-limited smoke/fine-tuning runs.
    The checkpoint is a delta against ``args.load``/``args.ref_load`` and is not
    a standalone Megatron checkpoint.
    """

    args = get_args()
    rank, world_size = _dist_rank_world()
    save_root = Path(args.save)
    iteration_dir = save_root / f"iter_{iteration:07d}"
    if rank == 0:
        iteration_dir.mkdir(parents=True, exist_ok=True)
    _safe_barrier()

    params = {}
    param_numel = 0
    modules = unwrap_model(model)
    if not isinstance(modules, Sequence):
        modules = [modules]
    for chunk_idx, module in enumerate(modules):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            key = f"model_{chunk_idx}.{name}"
            tensor = param.detach().cpu()
            params[key] = tensor
            param_numel += tensor.numel()

    tensor_model_parallel_rank = _safe_mpu_value("get_tensor_model_parallel_rank")
    pipeline_model_parallel_rank = _safe_mpu_value("get_pipeline_model_parallel_rank")
    expert_model_parallel_rank = _safe_mpu_value("get_expert_model_parallel_rank")
    data_parallel_rank = _safe_mpu_value("get_data_parallel_rank", with_context_parallel=True)
    rank_payload = {
        "format": "slime_megatron_trainable_only_v1",
        "iteration": iteration,
        "rank": rank,
        "world_size": world_size,
        "tensor_model_parallel_rank": tensor_model_parallel_rank,
        "pipeline_model_parallel_rank": pipeline_model_parallel_rank,
        "expert_model_parallel_rank": expert_model_parallel_rank,
        "data_parallel_rank": data_parallel_rank,
        "only_train_params_name_list": getattr(args, "only_train_params_name_list", None),
        "param_count": len(params),
        "param_numel": param_numel,
        "params": params,
    }
    rank_file = iteration_dir / f"trainable_rank_{rank:05d}.pt"
    atomic_torch_save(rank_payload, rank_file)

    rank_meta = {k: v for k, v in rank_payload.items() if k != "params"}
    _atomic_write_text(
        iteration_dir / f"trainable_rank_{rank:05d}.json",
        json.dumps(rank_meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    del params, rank_payload
    clear_memory(clear_host_memory=True)

    save_training_state = _env_flag("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE_TRAINING_STATE")
    optimizer_state_saved = False
    scheduler_state_saved = False
    rng_state_saved = False
    if save_training_state:
        topology = {
            "tensor_model_parallel_rank": tensor_model_parallel_rank,
            "pipeline_model_parallel_rank": pipeline_model_parallel_rank,
            "expert_model_parallel_rank": expert_model_parallel_rank,
            "data_parallel_rank": data_parallel_rank,
        }
        training_state_payload, training_state_metadata = build_training_state_payload(
            iteration=iteration,
            rank=rank,
            world_size=world_size,
            topology=topology,
            args=args,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            tensor_parallel=tensor_parallel,
        )
        optimizer_state_saved = training_state_metadata["optimizer_state_saved"]
        scheduler_state_saved = training_state_metadata["scheduler_state_saved"]
        rng_state_saved = training_state_metadata["rng_state_saved"]
        if not optimizer_state_saved and rank == 0:
            logger.warning(
                "Trainable-only training-state save was requested without optimizer state; "
                "set SAVE_OPTIMIZER=1 for an exact resume"
            )
        training_state_file = f"training_state_rank_{rank:05d}.pt"
        atomic_torch_save(training_state_payload, iteration_dir / training_state_file)
        _atomic_write_text(
            iteration_dir / f"training_state_rank_{rank:05d}.json",
            json.dumps(
                {
                    "format": training_state_payload["format"],
                    "iteration": iteration,
                    "rank": rank,
                    "world_size": world_size,
                    "optimizer_state_saved": optimizer_state_saved,
                    "scheduler_state_saved": scheduler_state_saved,
                    "rng_state_saved": rng_state_saved,
                    "progress": training_state_payload["progress"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        del training_state_payload
        clear_memory(clear_host_memory=True)
    _safe_barrier()

    if rank == 0:
        if save_training_state:
            training_rank_metadata = [
                json.loads(
                    (iteration_dir / f"training_state_rank_{idx:05d}.json").read_text(
                        encoding="utf-8"
                    )
                )
                for idx in range(world_size)
            ]
            optimizer_state_saved = all(
                item.get("optimizer_state_saved") is True for item in training_rank_metadata
            )
            scheduler_state_saved = all(
                item.get("scheduler_state_saved") is True for item in training_rank_metadata
            )
            rng_state_saved = all(
                item.get("rng_state_saved") is True for item in training_rank_metadata
            )
        common = {
            "format": "slime_megatron_trainable_only_v1",
            "iteration": iteration,
            "world_size": world_size,
            "requires_full_base_checkpoint": True,
            "base_ref_load": getattr(args, "load", None),
            "save_root": str(save_root),
            "only_train_params_name_list": getattr(args, "only_train_params_name_list", None),
            "tensor_model_parallel_size": _safe_mpu_value("get_tensor_model_parallel_world_size"),
            "pipeline_model_parallel_size": _safe_mpu_value("get_pipeline_model_parallel_world_size"),
            "expert_model_parallel_size": _safe_mpu_value("get_expert_model_parallel_world_size"),
            "data_parallel_size": _safe_mpu_value("get_data_parallel_world_size", with_context_parallel=True),
            "rank_files": [f"trainable_rank_{idx:05d}.pt" for idx in range(world_size)],
            "training_state_format": TRAINING_STATE_FORMAT if save_training_state else None,
            "training_state_files": (
                [f"training_state_rank_{idx:05d}.pt" for idx in range(world_size)]
                if save_training_state
                else []
            ),
            "training_state_requested": save_training_state,
            "optimizer_state_saved": optimizer_state_saved,
            "scheduler_state_saved": scheduler_state_saved,
            "rng_state_saved": rng_state_saved,
            "created_by": "slime.megatron.trainable_only_save",
        }
        _atomic_write_text(
            iteration_dir / "trainable_common.json",
            json.dumps(common, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write_text(save_root / "latest_trainable_iteration.txt", f"{iteration}\n")
        logger.info(
            "Saved trainable-only Megatron checkpoint iteration=%s world_size=%s to %s",
            iteration,
            world_size,
            iteration_dir,
        )
    _safe_barrier()


def _resolve_trainable_only_checkpoint_dir(path: str) -> Path:
    checkpoint_path = Path(path)
    if (checkpoint_path / "trainable_common.json").is_file():
        return checkpoint_path

    latest_iteration_file = checkpoint_path / "latest_trainable_iteration.txt"
    if latest_iteration_file.is_file():
        latest_iteration = int(latest_iteration_file.read_text(encoding="utf-8").strip())
        iteration_dir = checkpoint_path / f"iter_{latest_iteration:07d}"
        if (iteration_dir / "trainable_common.json").is_file():
            return iteration_dir
        raise FileNotFoundError(
            f"latest_trainable_iteration.txt points to {latest_iteration}, "
            f"but {iteration_dir / 'trainable_common.json'} does not exist"
        )

    raise FileNotFoundError(
        f"{checkpoint_path} is not a trainable-only checkpoint directory; expected "
        "trainable_common.json or latest_trainable_iteration.txt"
    )


@torch.no_grad()
def _load_trainable_only_checkpoint_if_requested(
    model: Sequence[DDP],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: OptimizerParamScheduler | None,
) -> tuple[int | None, bool]:
    load_path = os.environ.get("SLIME_MEGATRON_TRAINABLE_ONLY_LOAD")
    if not load_path:
        return None, False

    checkpoint_dir = _resolve_trainable_only_checkpoint_dir(load_path)
    common_path = checkpoint_dir / "trainable_common.json"
    common = json.loads(common_path.read_text(encoding="utf-8"))
    if common.get("format") != "slime_megatron_trainable_only_v1":
        raise ValueError(f"Unsupported trainable-only checkpoint format in {common_path}: {common.get('format')}")
    checkpoint_iteration = int(common.get("iteration", -1))
    if checkpoint_iteration < 0:
        raise ValueError(f"Invalid trainable-only checkpoint iteration in {common_path}: {common.get('iteration')}")

    rank, world_size = _dist_rank_world()
    checkpoint_world_size = int(common.get("world_size", -1))
    if checkpoint_world_size != world_size:
        raise ValueError(
            f"Trainable-only checkpoint world_size mismatch: checkpoint={checkpoint_world_size}, runtime={world_size}"
        )
    parallel_size_checks = {
        "tensor_model_parallel_size": _safe_mpu_value("get_tensor_model_parallel_world_size"),
        "pipeline_model_parallel_size": _safe_mpu_value("get_pipeline_model_parallel_world_size"),
        "expert_model_parallel_size": _safe_mpu_value("get_expert_model_parallel_world_size"),
        "data_parallel_size": _safe_mpu_value("get_data_parallel_world_size", with_context_parallel=True),
    }
    for key, runtime_value in parallel_size_checks.items():
        checkpoint_value = common.get(key)
        if checkpoint_value is None or runtime_value is None:
            continue
        if int(checkpoint_value) != int(runtime_value):
            raise ValueError(
                f"Trainable-only checkpoint {key} mismatch: "
                f"checkpoint={checkpoint_value}, runtime={runtime_value}. "
                "Load the matching trainable-only checkpoint for this Megatron layout, "
                "or start from the full base checkpoint."
            )

    rank_files = common.get("rank_files") or []
    rank_file_name = rank_files[rank] if rank < len(rank_files) else f"trainable_rank_{rank:05d}.pt"
    rank_file = checkpoint_dir / rank_file_name
    if not rank_file.is_file():
        raise FileNotFoundError(f"Missing rank-local trainable-only checkpoint shard: {rank_file}")

    payload = torch.load(rank_file, map_location="cpu", weights_only=False)
    if payload.get("format") != "slime_megatron_trainable_only_v1":
        raise ValueError(f"Unsupported rank shard format in {rank_file}: {payload.get('format')}")
    if int(payload.get("rank", -1)) != rank:
        raise ValueError(f"Rank shard mismatch in {rank_file}: checkpoint={payload.get('rank')}, runtime={rank}")
    if int(payload.get("world_size", -1)) != world_size:
        raise ValueError(
            f"Rank shard world_size mismatch in {rank_file}: checkpoint={payload.get('world_size')}, runtime={world_size}"
        )

    modules = unwrap_model(model)
    if not isinstance(modules, Sequence):
        modules = [modules]
    runtime_params = {
        f"model_{chunk_idx}.{name}": param
        for chunk_idx, module in enumerate(modules)
        for name, param in module.named_parameters()
    }

    checkpoint_params = payload.get("params")
    if not isinstance(checkpoint_params, dict):
        raise ValueError(f"Rank shard {rank_file} does not contain a params dict")

    missing = sorted(set(checkpoint_params) - set(runtime_params))
    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"{len(missing)} trainable-only tensors are missing from the runtime model: {preview}")

    loaded_param_count = len(checkpoint_params)
    loaded_numel = 0
    for name, tensor in checkpoint_params.items():
        param = runtime_params[name]
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint={tuple(tensor.shape)}, runtime={tuple(param.shape)}"
            )
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype, non_blocking=True))
        loaded_numel += tensor.numel()

    del checkpoint_params, payload, runtime_params
    clear_memory(clear_host_memory=True)
    _safe_barrier()

    optimizer_state_loaded = False
    if _env_flag("SLIME_MEGATRON_TRAINABLE_ONLY_LOAD_TRAINING_STATE"):
        strict = os.getenv(
            "SLIME_MEGATRON_TRAINABLE_ONLY_LOAD_TRAINING_STATE_STRICT",
            "1",
        ).strip().lower() in {"1", "true", "yes", "on"}
        training_state_files = common.get("training_state_files") or []
        training_state_name = (
            training_state_files[rank]
            if rank < len(training_state_files)
            else f"training_state_rank_{rank:05d}.pt"
        )
        training_state_file = checkpoint_dir / training_state_name
        if not training_state_file.is_file():
            message = f"Missing rank-local trainable-only training state: {training_state_file}"
            if strict:
                raise FileNotFoundError(message)
            logger.warning("%s; continuing as a model-only warm start", message)
        else:
            training_state = torch.load(training_state_file, map_location="cpu", weights_only=False)
            optimizer_state_loaded, warnings = restore_training_state_payload(
                training_state,
                expected_iteration=checkpoint_iteration,
                expected_rank=rank,
                expected_world_size=world_size,
                args=get_args(),
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                tensor_parallel=tensor_parallel,
                strict=strict,
                source=str(training_state_file),
            )
            for warning in warnings:
                logger.warning("%s", warning)
            del training_state
            clear_memory(clear_host_memory=True)

    if rank == 0:
        logger.info(
            "Loaded trainable-only Megatron checkpoint iteration=%s world_size=%s from %s; "
            "rank-local tensors are restored on every rank, rank0_tensor_count=%s rank0_numel=%s "
            "training_state_requested=%s optimizer_state_loaded=%s",
            common.get("iteration"),
            world_size,
            checkpoint_dir,
            loaded_param_count,
            loaded_numel,
            _env_flag("SLIME_MEGATRON_TRAINABLE_ONLY_LOAD_TRAINING_STATE"),
            optimizer_state_loaded,
        )
    return checkpoint_iteration, optimizer_state_loaded


def save(
    iteration: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    checkpointing_context: dict | None = None,
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    disable_hook = should_disable_forward_pre_hook(args)
    if disable_hook:
        disable_forward_pre_hook(model)
    try:
        if getattr(args, "use_megatron_lora", False) and getattr(args, "megatron_lora_save_adapter_only", True):
            save_megatron_lora_checkpoint(model, args, iteration)
            return
        if _env_flag("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE"):
            _save_trainable_only_checkpoint(iteration, model, optimizer, opt_param_scheduler)
            return
        save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far=0,
            checkpointing_context=checkpointing_context,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=_drop_rank_local_common_checkpoint_state,
        )
    finally:
        if disable_hook:
            enable_forward_pre_hook(model)


def _maybe_redirect_lora_adapter_load(args: Namespace) -> None:
    if not getattr(args, "use_megatron_lora", False):
        return
    if getattr(args, "megatron_lora_adapter_load", None):
        return

    load_path = getattr(args, "load", None)
    if not load_path:
        return
    load_root = Path(load_path)
    if not (load_root / "latest_megatron_lora_iteration.txt").is_file():
        return

    args.megatron_lora_adapter_load = str(load_root)
    base_load = None
    try:
        latest_iteration = int((load_root / "latest_megatron_lora_iteration.txt").read_text(encoding="utf-8").strip())
        metadata_path = load_root / f"iter_{latest_iteration:07d}" / "meta.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            base_load = metadata.get("base_load") or metadata.get("base_hf_checkpoint")
    except Exception as exc:
        logger.warning("Could not inspect Megatron LoRA adapter metadata under %s: %s", load_root, exc)

    args.load = base_load or getattr(args, "ref_load", None) or getattr(args, "hf_checkpoint", None)
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = True
    logger.info(
        "Detected adapter-only Megatron LoRA load root %s; loading base weights from %s and adapter from %s",
        load_root,
        args.load,
        args.megatron_lora_adapter_load,
    )


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    _maybe_redirect_lora_adapter_load(args)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    if reinit_critic_output_layer:
        _reinitialize_critic_output_layer(model)
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
    loaded_lora_iteration = None
    if getattr(args, "use_megatron_lora", False):
        loaded_lora_iteration = load_megatron_lora_checkpoint(model, getattr(args, "megatron_lora_adapter_load", None))
        if loaded_lora_iteration is not None:
            iteration = loaded_lora_iteration
    loaded_trainable_only_iteration, trainable_optimizer_state_loaded = (
        _load_trainable_only_checkpoint_if_requested(model, optimizer, opt_param_scheduler)
    )
    if loaded_trainable_only_iteration is not None:
        iteration = loaded_trainable_only_iteration
    if (
        (loaded_lora_iteration is not None or loaded_trainable_only_iteration is not None)
        and optimizer is not None
        and not trainable_optimizer_state_loaded
        and hasattr(optimizer, "reload_model_params")
    ):
        optimizer.reload_model_params()
    clear_memory()

    return model, optimizer, opt_param_scheduler, iteration
