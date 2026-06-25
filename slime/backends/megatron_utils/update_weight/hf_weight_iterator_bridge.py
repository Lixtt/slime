import dataclasses


from slime.utils import megatron_bridge_utils
from slime.utils.misc import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


def _patch_bridge_expert_cache_to_cpu():
    """Monkey-patch GPTOSSBridge class to cache expert weights on CPU.

    This avoids GPU OOM when torch.cat merges all experts, especially in
    colocated mode where SGLang and Megatron share the same GPU.
    """
    try:
        from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
    except ImportError:
        return

    if getattr(GPTOSSBridge, "_cpu_cache_patched", False):
        return

    _orig = GPTOSSBridge.maybe_modify_converted_hf_weight

    def _patched(self, task, converted_weights_dict):
        cpu_dict = {k: v.cpu() for k, v in converted_weights_dict.items()}
        result = _orig(self, task, cpu_dict)
        # Move merged result back to GPU for CUDA IPC serialization
        return {k: v.cuda() for k, v in result.items()} if result else result

    GPTOSSBridge.maybe_modify_converted_hf_weight = _patched
    GPTOSSBridge._cpu_cache_patched = True


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.trainable_only:
            raise NotImplementedError("trainable-only rollout weight sync is only implemented for raw HF conversion.")
        self._bridge, self._hf_pretrained, _ = megatron_bridge_utils.build_bridge_for_hf_checkpoint(
            self.args.hf_checkpoint,
            load_weights=False,
        )
        if self._hf_pretrained is not None and getattr(self._bridge, "hf_pretrained", None) is None:
            self._bridge.hf_pretrained = self._hf_pretrained
        _patch_bridge_expert_cache_to_cpu()

    def get_hf_weight_chunks(self, megatron_local_weights, progress_desc: str = "Update weights"):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            if hasattr(self._bridge, "get_conversion_tasks"):
                conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            else:
                conversion_tasks = self._bridge.build_conversion_tasks(self._hf_pretrained, self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            if hasattr(self._bridge, "export_hf_weights"):
                named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)
            else:
                named_weights = self._bridge.stream_weights_megatron_to_hf(
                    self.model,
                    self._hf_pretrained,
                    cpu=False,
                    conversion_tasks=conversion_tasks,
                )

            hf_to_megatron_name = _hf_to_megatron_name_map(conversion_tasks)

            def _streaming_quantized():
                for hf_param_name, weight, megatron_param_name in _iter_named_weights_with_megatron_names(
                    named_weights,
                    hf_to_megatron_name,
                ):
                    processed_weight = postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    )
                    converted_named_params = [(hf_param_name, processed_weight)]
                    quantized_batch = quantize_params(
                        args=self.args,
                        megatron_name=megatron_param_name,
                        converted_named_params=converted_named_params,
                        quantization_config=self.quantization_config,
                    )
                    yield from quantized_batch

            yield from chunk_named_params_by_size(
                _streaming_quantized(), chunk_size=self.args.update_weight_buffer_size
            )


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task is None:
            return None
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)


def _hf_to_megatron_name_map(conversion_tasks):
    """Best-effort map for bridge outputs that do not include Megatron names."""
    ret = {}
    for task in conversion_tasks:
        if task is None:
            continue
        mapping = getattr(task, "mapping", None)
        hf_param = getattr(mapping, "hf_param", None)
        megatron_name = getattr(task, "global_param_name", None) or getattr(task, "param_name", None)
        if not megatron_name:
            continue
        if isinstance(hf_param, str):
            ret[hf_param] = megatron_name
        elif isinstance(hf_param, dict):
            for name in hf_param.values():
                ret[name] = megatron_name

    ret.setdefault("lm_head.weight", "output_layer.weight")
    for hf_name in list(ret):
        if hf_name.endswith(".embed_tokens.weight"):
            ret.setdefault(hf_name, "embedding.word_embeddings.weight")
    return ret


def _iter_named_weights_with_megatron_names(named_weights, hf_to_megatron_name):
    for item in named_weights:
        if len(item) == 3:
            hf_param_name, weight, megatron_param_name = item
        elif len(item) == 2:
            hf_param_name, weight = item
            megatron_param_name = hf_to_megatron_name.get(hf_param_name, hf_param_name)
        else:
            raise ValueError(f"Unexpected bridge weight tuple length: {len(item)}")
        yield hf_param_name, weight, megatron_param_name
