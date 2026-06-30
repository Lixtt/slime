import base64
import logging
import pickle
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .colocated_payload import select_colocated_tensor_payload_ranks, split_hf_named_tensors_for_sglang_pp
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)


logger = logging.getLogger(__name__)
_LOGGED_NESTED_PP_PAYLOAD = False


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            trainable_only=getattr(args, "update_weights_trainable_only", False),
        )
        self._full_hf_weight_iterator = None
        if getattr(args, "update_weights_trainable_only", False) and getattr(
            args, "update_weights_initial_full_sync", False
        ):
            self._full_hf_weight_iterator = HfWeightIteratorBase.create(
                args=args,
                model=model,
                model_name=model_name,
                quantization_config=quantization_config,
                trainable_only=False,
            )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
        self._ipc_force_cpu_payload = False

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                payload_group_ranks = select_colocated_tensor_payload_ranks(group_ranks, self.args)
                new_group = dist.new_group(ranks=payload_group_ranks, backend="gloo")
                if dist.get_rank() in payload_group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = payload_group_ranks[0]
                if dist.get_rank() == payload_group_ranks[0] and len(payload_group_ranks) != len(group_ranks):
                    logger.info(
                        "Colocated tensor update will gather %d TP payload rank(s) for PP engine=%s "
                        "instead of %d engine rank(s): payload_ranks=%s engine_ranks=%s sglang_pp_size=%s",
                        len(payload_group_ranks),
                        i,
                        len(group_ranks),
                        payload_group_ranks,
                        group_ranks,
                        getattr(self.args, "sglang_pp_size", 1),
                    )

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine
                spans_nodes = colocate_gpu_counts[i] > getattr(
                    self.args, "num_gpus_per_node", colocate_gpu_counts[i]
                )
                explicit_cpu_payload = getattr(self.args, "colocated_tensor_update_cpu_payload", False)
                self._ipc_force_cpu_payload = bool(explicit_cpu_payload or spans_nodes)
                if dist.get_rank() == start and self._ipc_force_cpu_payload:
                    logger.info(
                        "Using CPU payload for colocated tensor update: engine=%s ranks=%s-%s spans_nodes=%s explicit=%s",
                        i,
                        start,
                        end - 1,
                        spans_nodes,
                        explicit_cpu_payload,
                    )

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        weight_iterator = self._hf_weight_iterator
        progress_desc = "Update weights"
        if self.weight_version == 1 and self._full_hf_weight_iterator is not None:
            weight_iterator = self._full_hf_weight_iterator
            progress_desc = "Initial full update weights"

        for hf_named_tensors in weight_iterator.get_hf_weight_chunks(
            megatron_local_weights,
            progress_desc=progress_desc,
        ):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # The colocated PP path may gather payloads only from the effective
            # TP ranks because SGLang indexes payloads by tp_rank. Keep every
            # Megatron rank chunk-synchronous before the next PP/TP collective.
            dist.barrier(group=get_gloo_group())
            # Free GPU tensors so the caching allocator can reuse the blocks,
            # then release CUDA IPC cache entries whose consumers (sglang engines)
            # have already closed their IPC handles.
            del long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        torch.cuda.ipc_collect()

        if rank == 0:
            # Quantized rollout weights need the same post-load processing that
            # SGLang runs during initial model load. FP8 uses it to refresh
            # packed/fused derived tensors after online trainable-only updates.
            if self.quantization_config and self.quantization_config["quant_method"] in [
                "compressed-tensors",
                "fp8",
            ]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            args=self.args,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
            force_cpu_payload=self._ipc_force_cpu_payload,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    args: Namespace | None,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
    force_cpu_payload: bool = False,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    long_live_tensors = []
    pp_size = int(getattr(args, "sglang_pp_size", 1) or 1) if args is not None else 1
    num_layers = int(getattr(args, "num_layers", 0) or 0) if args is not None else 0
    nested_pp_payload = pp_size > 1
    if nested_pp_payload and num_layers <= 0:
        raise ValueError("SGLang PP tensor update requires args.num_layers to split payloads by PP rank")

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        if nested_pp_payload:
            pp_named_tensors = split_hf_named_tensors_for_sglang_pp(
                named_tensors,
                pp_size=pp_size,
                num_layers=num_layers,
                partition=getattr(args, "sglang_pp_layer_partition", None),
            )
            serialized_tensors.append(
                [
                    _serialize_flattened_bucket(
                        pp_tensors,
                        long_live_tensors=long_live_tensors,
                        force_cpu_payload=force_cpu_payload,
                    )
                    for pp_tensors in pp_named_tensors
                ]
            )
        else:
            serialized_tensors.append(
                _serialize_flattened_bucket(
                    named_tensors,
                    long_live_tensors=long_live_tensors,
                    force_cpu_payload=force_cpu_payload,
                )
            )

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.
        num_dtypes = len(serialized_named_tensors[0])
        if nested_pp_payload:
            global _LOGGED_NESTED_PP_PAYLOAD
            if not _LOGGED_NESTED_PP_PAYLOAD:
                logger.info(
                    "Colocated tensor update will send nested SGLang PP payloads: pp_size=%s "
                    "tp_payload_ranks=%s num_layers=%s partition=%s",
                    pp_size,
                    len(serialized_named_tensors),
                    num_layers,
                    getattr(args, "sglang_pp_layer_partition", None),
                )
                _LOGGED_NESTED_PP_PAYLOAD = True
        for i in range(num_dtypes):
            if nested_pp_payload:
                payload = [
                    [rank_tensors[i][pp_rank] for rank_tensors in serialized_named_tensors]
                    for pp_rank in range(pp_size)
                ]
            else:
                payload = [tensors[i] for tensors in serialized_named_tensors]
            kwargs = {
                "serialized_named_tensors": payload,
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors


def _serialize_flattened_bucket(
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    long_live_tensors: list[Any],
    force_cpu_payload: bool,
):
    if named_tensors:
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()
    else:
        metadata = []
        flattened_tensor = torch.empty(0, dtype=torch.uint8, device="cpu")

    if force_cpu_payload:
        # CUDA IPC handles are node-local. A colocated PP rollout engine can
        # span nodes, so serialize tensor data through CPU for that path.
        flattened_tensor = flattened_tensor.detach().cpu().contiguous()
    flattened_tensor_data = {
        "flattened_tensor": flattened_tensor,
        "metadata": metadata,
    }
    long_live_tensors.append(flattened_tensor_data)
    if force_cpu_payload:
        payload = pickle.dumps(flattened_tensor_data, protocol=pickle.HIGHEST_PROTOCOL)
        return base64.b64encode(payload).decode("utf-8")
    return MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
