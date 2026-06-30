import logging
from argparse import Namespace
from collections.abc import Sequence


logger = logging.getLogger(__name__)


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
