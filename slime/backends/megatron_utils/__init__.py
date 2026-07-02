import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        # Save/restore the prior interesting_region instead of unconditionally
        # forcing it back to True. deep_ep.Buffer() can be constructed while
        # already nested inside an outer torch_memory_saver.disable() (e.g.
        # during colocated weight sync for MoE/EP models like GLM5.2, where
        # weights_getter()'s EP gather step touches DeepEP buffers). Forcing
        # True unconditionally silently re-enables the memory-saver's
        # VMM-backed allocator for the rest of that disable() scope, which
        # makes any tensor allocated afterward (e.g. the CUDA-IPC flattened
        # weight bucket) fail _share_cuda_ with "invalid argument" even
        # though the caller believes it is still inside disable().
        cdll = None
        prior_interesting_region = None
        if torch_memory_saver._impl is not None:
            cdll = torch_memory_saver._impl._binary_wrapper.cdll
            prior_interesting_region = cdll.tms_get_interesting_region()
            cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if cdll is not None:
            cdll.tms_set_interesting_region(prior_interesting_region)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    def patch_rotary_embedding(cls):
        _original_forward = cls.forward

        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = _patched_forward

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)

from . import megatron_patch  # noqa: F401, E402
