import logging
import os
import re
from argparse import Namespace
from collections.abc import Sequence
from typing import TypeVar


logger = logging.getLogger(__name__)
NamedTensor = TypeVar("NamedTensor")

_HF_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def select_colocated_tensor_payload_ranks(group_ranks: Sequence[int], args: Namespace) -> list[int]:
    """Return ranks that should serialize payloads for a colocated SGLang engine.

    SGLang's tensor update endpoint deserializes payloads with
    ``serialized_named_tensors[self.tp_rank]``. For pipeline-parallel rollout
    engines, ``tp_rank`` repeats on every PP stage, so sending all engine ranks
    duplicates the same TP payload ``pp_size`` times and SGLang ignores the
    extras. Megatron's direct HF iterator has already broadcast PP parameters to
    all ranks, therefore the first effective-TP ranks have the complete payload
    needed by every PP stage.
    """

    ranks = list(group_ranks)
    pp_size = int(getattr(args, "sglang_pp_size", 1) or 1)
    if pp_size <= 1:
        return ranks
    if len(ranks) % pp_size != 0:
        logger.warning(
            "Cannot derive SGLang effective TP payload ranks from %d ranks and pp_size=%d; using all ranks.",
            len(ranks),
            pp_size,
        )
        return ranks
    tp_size = len(ranks) // pp_size
    if tp_size <= 0:
        return ranks
    return ranks[:tp_size]


def get_sglang_pp_layer_ranges(
    *,
    num_layers: int,
    pp_size: int,
    partition: str | None = None,
) -> list[tuple[int, int]]:
    """Return SGLang-compatible ``(start, end)`` layer ranges for each PP rank."""

    if pp_size <= 1:
        return [(0, num_layers)]

    partition = (partition if partition is not None else os.getenv("SGLANG_PP_LAYER_PARTITION", "")).strip()
    if partition:
        try:
            partitions = [int(part) for part in partition.split(",")]
        except ValueError as exc:
            raise ValueError(f"Invalid SGLANG_PP_LAYER_PARTITION={partition!r}") from exc
        if len(partitions) != pp_size:
            raise ValueError(f"SGLANG_PP_LAYER_PARTITION has {len(partitions)} parts, expected {pp_size}")
        if sum(partitions) != num_layers:
            raise ValueError(f"SGLANG_PP_LAYER_PARTITION sums to {sum(partitions)}, expected {num_layers}")
        if any(part <= 0 for part in partitions):
            raise ValueError(f"SGLANG_PP_LAYER_PARTITION contains non-positive sizes: {partitions}")

        ranges = []
        start = 0
        for size in partitions:
            ranges.append((start, start + size))
            start += size
        return ranges

    base_layers = num_layers // pp_size
    remainder = num_layers % pp_size
    ranges = []
    for pp_rank in range(pp_size):
        if pp_rank >= pp_size - remainder:
            partitions_without_extra_layer = pp_size - remainder
            start_layer = pp_rank * (base_layers + 1) - partitions_without_extra_layer
            end_layer = start_layer + (base_layers + 1)
        else:
            start_layer = pp_rank * base_layers
            end_layer = start_layer + base_layers
        ranges.append((start_layer, end_layer))
    return ranges


def get_hf_layer_index(name: str) -> int | None:
    match = _HF_LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def split_hf_named_tensors_for_sglang_pp(
    named_tensors: Sequence[tuple[str, NamedTensor]],
    *,
    pp_size: int,
    num_layers: int,
    partition: str | None = None,
) -> list[list[tuple[str, NamedTensor]]]:
    """Split HF-style layer tensors into SGLang PP-stage payloads.

    Non-layer tensors are replicated to preserve the previous flat payload
    behavior. The GLM5.2 trainable-only path uses layer-local q/kv tensors, so
    the normal production path avoids that fallback.
    """

    if pp_size <= 1:
        return [list(named_tensors)]

    layer_ranges = get_sglang_pp_layer_ranges(
        num_layers=num_layers,
        pp_size=pp_size,
        partition=partition,
    )
    by_pp: list[list[tuple[str, NamedTensor]]] = [[] for _ in range(pp_size)]

    for item in named_tensors:
        name, _tensor = item
        layer_idx = get_hf_layer_index(name)
        if layer_idx is None or layer_idx < 0 or layer_idx >= num_layers:
            for bucket in by_pp:
                bucket.append(item)
            continue

        for pp_rank, (start, end) in enumerate(layer_ranges):
            if start <= layer_idx < end:
                by_pp[pp_rank].append(item)
                break
        else:
            logger.warning(
                "Could not map HF tensor %s with layer_idx=%s to SGLang PP ranges %s; replicating it.",
                name,
                layer_idx,
                layer_ranges,
            )
            for bucket in by_pp:
                bucket.append(item)

    return by_pp
