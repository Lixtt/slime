import dataclasses
import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group
from slime.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf
from ..sglang import monkey_patch_torch_reductions
from .common import all_gather_object_for_group_via_gloo, all_gather_params_async, named_params_and_buffers
from .hf_weight_iterator_base import HfWeightIteratorBase
from .tensor_bytes import restore_tensor_from_cpu_byte_tensor, tensor_to_cpu_byte_tensor

logger = logging.getLogger(__name__)


class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(
            self.args,
            self.model,
            trainable_only=self.trainable_only,
        )
        if self.trainable_only and dist.get_rank() == 0:
            param_count = sum(len(bucket) for bucket in self.megatron_local_param_info_buckets)
            logger.info(
                "Megatron raw HF weight iterator will sync %d trainable parameter group(s) in %d bucket(s).",
                param_count,
                len(self.megatron_local_param_info_buckets),
            )
        if getattr(self.args, "megatron_direct_pp_broadcast_backend", "nccl") == "gloo" and dist.get_rank() == 0:
            logger.info("Megatron raw HF weight iterator will use CPU/Gloo for PP tensor broadcasts.")

    def get_hf_weight_chunks(self, megatron_local_weights, progress_desc: str = "Update weights"):
        rank = dist.get_rank()

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc=progress_desc
        ):
            megatron_full_params = _get_megatron_full_params(
                self.args,
                megatron_local_param_infos,
                megatron_local_weights,
            )
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
            yield hf_named_tensors
            del megatron_full_params

    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )
        return hf_named_tensors


def _get_megatron_full_params(
    args: Namespace,
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    rank = dist.get_rank()
    # init params:
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    torch.cuda.synchronize()

    # broadcast params across pp ranks
    if pp_size > 1:
        if getattr(args, "megatron_direct_pp_broadcast_backend", "nccl") == "gloo":
            _broadcast_params_across_pp_ranks_via_gloo(megatron_local_param_infos, params)
        else:
            handles = []
            for info, param in zip(megatron_local_param_infos, params, strict=False):
                if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                    handles.append(
                        torch.distributed.broadcast(
                            param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                        )
                    )
            for handle in handles:
                handle.wait()

    # broadcast params across ep ranks
    if ep_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                handles.append(
                    torch.distributed.broadcast(
                        param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # Set tp attrs for all params
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))

    return gathered_params


def _broadcast_params_across_pp_ranks_via_gloo(
    param_infos: Sequence[ParamInfo],
    params: Sequence[torch.Tensor],
) -> None:
    """Broadcast PP params through CPU byte buffers over the world Gloo group.

    Gloo does not reliably support every model dtype we may see here, so the
    tensor payload is sent as raw uint8 bytes and restored to the original dtype
    and shape before copying back to the CUDA tensor.
    """

    gloo_group = get_gloo_group()
    rank = dist.get_rank()
    for info, param in zip(param_infos, params, strict=False):
        if rank == info.src_rank:
            cpu_tensor = param.detach().cpu().contiguous()
            byte_tensor = tensor_to_cpu_byte_tensor(cpu_tensor)
            if byte_tensor.numel() != info.size:
                raise RuntimeError(
                    f"Unexpected byte size for {info.name}: {byte_tensor.numel()} != {info.size}"
                )
        else:
            byte_tensor = torch.empty(info.size, dtype=torch.uint8, device="cpu")

        dist.broadcast(byte_tensor, src=info.src_rank, group=gloo_group)
        if rank != info.src_rank:
            restored = restore_tensor_from_cpu_byte_tensor(byte_tensor, shape=info.shape, dtype=info.dtype)
            param.copy_(restored.to(device=param.device, non_blocking=True))
        del byte_tensor
        if rank == info.src_rank:
            del cpu_tensor

    torch.cuda.synchronize()


def _get_megatron_local_param_info_buckets(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    trainable_only: bool = False,
) -> list[list[ParamInfo]]:
    """
    Partition params into buckets ≤ update_weight_buffer_size (with TP replication).
    """
    param_infos = _get_megatron_local_param_infos(args, model, trainable_only=trainable_only)
    param_info_buckets = [[]]  # Start with one empty bucket
    buffer_size = 0  # Track current bucket size in bytes

    for info in param_infos:
        # Expert params use expert-TP size, others use regular-TP size
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        # Full param size = shard size × TP replicas (all-gather will reconstruct full param)
        param_size = info.size * tp_size

        # If adding this param exceeds limit AND current bucket has params: start new bucket
        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0

        # Add param to current bucket and update size
        param_info_buckets[-1].append(info)
        buffer_size += param_size

    return param_info_buckets


def _get_megatron_local_param_infos(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    trainable_only: bool = False,
) -> list[ParamInfo]:
    """
    Build global param metadata: collect → exchange PP/EP → resolve duplicates (MTP virtual PP)
    by min src_rank → validate. Returns sorted ParamInfo identical across all ranks.
    """
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    param_infos = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model, trainable_only=trainable_only):
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if pp_size > 1:
        param_infos_list = all_gather_object_for_group_via_gloo(
            obj=(rank, param_infos),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        assert len(param_infos_list) == pp_size, f"Expected {pp_size} PP metadata payloads, got {len(param_infos_list)}"
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    old_info = param_infos[name]
                    if old_info.src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = all_gather_object_for_group_via_gloo(
            obj=(rank, param_infos),
            group=mpu.get_expert_model_parallel_group(),
        )
        assert len(param_infos_list) == ep_size, f"Expected {ep_size} EP metadata payloads, got {len(param_infos_list)}"
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name not in param_infos:
                    # here we need to set the src_rank to the rank within the expert model parallel group
                    info = dataclasses.replace(info, src_rank=src_rank)
                    param_infos[name] = info

    param_infos = list(param_infos.values())
    param_infos = sorted(param_infos, key=lambda info: info.name)

    # Check all ranks has the same parameter info
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )
    if trainable_only and len(param_infos) == 0:
        raise RuntimeError(
            "No trainable Megatron parameters matched for rollout weight sync. "
            "Check --only-train-params-name-list and model parameter names."
        )
    for infos in all_param_info_list:
        assert len(infos) == len(param_infos), (
            f"Parameter info length mismatch: {len(infos)} != {len(param_infos)}"
        )
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"

    return param_infos
