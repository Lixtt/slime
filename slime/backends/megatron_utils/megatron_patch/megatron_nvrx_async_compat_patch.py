"""Compatibility shim for Megatron's NVRx async checkpoint imports.

Some nvidia-resiliency-ext builds expose the write-result queue helper only as
``_get_write_results_queue`` while newer Megatron imports the public
``get_write_results_queue`` symbol.  Adding the alias before checkpoint loading
keeps Megatron's NVRx path usable without changing the selected async strategy.

The same package family also changed ``CachedMetadataFileSystemReader`` from a
``(path, cache_metadata=...)`` constructor to ``(path)`` while Megatron still
passes the keyword.  A tiny subclass keeps that call compatible and preserves
the package's own metadata caching behavior.
"""

import inspect
import logging
import warnings

logger = logging.getLogger(__name__)

try:
    from nvidia_resiliency_ext.checkpointing.async_ckpt import (
        cached_metadata_filesystem_reader,
        filesystem_async,
    )

    if not hasattr(filesystem_async, "get_write_results_queue") and hasattr(
        filesystem_async, "_get_write_results_queue"
    ):
        filesystem_async.get_write_results_queue = filesystem_async._get_write_results_queue
        logger.info("Added nvidia_resiliency_ext get_write_results_queue compatibility alias.")

    _reader_cls = cached_metadata_filesystem_reader.CachedMetadataFileSystemReader
    _reader_params = inspect.signature(_reader_cls.__init__).parameters
    if "cache_metadata" not in _reader_params:

        class CachedMetadataFileSystemReaderCompat(_reader_cls):
            def __init__(self, path, *args, cache_metadata=None, **kwargs):
                del cache_metadata
                super().__init__(path, *args, **kwargs)

        cached_metadata_filesystem_reader.CachedMetadataFileSystemReader = (
            CachedMetadataFileSystemReaderCompat
        )
        logger.info(
            "Wrapped nvidia_resiliency_ext CachedMetadataFileSystemReader for cache_metadata kwarg."
        )
except ImportError as exc:
    warnings.warn(
        f"slime NVRx async checkpoint compatibility patch not applied ({exc!r}).",
        stacklevel=2,
    )
