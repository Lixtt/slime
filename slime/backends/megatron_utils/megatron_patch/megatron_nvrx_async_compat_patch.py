"""Compatibility shim for Megatron's NVRx async checkpoint imports.

Some nvidia-resiliency-ext builds expose the write-result queue helper only as
``_get_write_results_queue`` while newer Megatron imports the public
``get_write_results_queue`` symbol.  Adding the alias before checkpoint loading
keeps Megatron's NVRx path usable without changing the selected async strategy.
"""

import logging
import warnings

logger = logging.getLogger(__name__)

try:
    from nvidia_resiliency_ext.checkpointing.async_ckpt import filesystem_async

    if not hasattr(filesystem_async, "get_write_results_queue") and hasattr(
        filesystem_async, "_get_write_results_queue"
    ):
        filesystem_async.get_write_results_queue = filesystem_async._get_write_results_queue
        logger.info("Added nvidia_resiliency_ext get_write_results_queue compatibility alias.")
except ImportError as exc:
    warnings.warn(
        f"slime NVRx async checkpoint compatibility patch not applied ({exc!r}).",
        stacklevel=2,
    )
