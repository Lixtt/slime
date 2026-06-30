import re

import torch

from slime.backends.megatron_utils.kernels.fp8_kernel import blockwise_cast_to_fp8_triton

from ...sglang import quant_weight_ue8m0, should_deepgemm_weight_requant_ue8m0, transform_scale_ue8m0


def quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "fp8"
    fmt = quantization_config.get("fmt", "e4m3")
    assert fmt == "e4m3", f"Unsupported FP8 format: {fmt}"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
        match = re.match(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size))

            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size))

            return quantize_named_params

    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        # indexer
        "self_attention.wq_b.weight",
        "self_attention.wk.weight",
        # linear attention
        "self_attention.linear_attn.in_proj_qkv.weight",
        "self_attention.linear_attn.in_proj_z.weight",
        "self_attention.linear_attn.out_proj.weight",
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_param(name, weight, weight_block_size):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    weight = _to_current_cuda_if_available(weight)
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        if should_deepgemm_weight_requant_ue8m0 and should_deepgemm_weight_requant_ue8m0(
            weight_block_size=weight_block_size
        ):
            qweight, scale = quant_weight_ue8m0(weight, weight_block_size=weight_block_size)
            scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
        else:
            qweight, scale = blockwise_cast_to_fp8_triton(weight, weight_block_size)
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # per tensor quant
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]


def _to_current_cuda_if_available(tensor):
    device = getattr(tensor, "device", None)
    if getattr(device, "type", None) == "cuda":
        return tensor
    cuda_device = _current_cuda_device_for_fp8_quant(tensor)
    moved = tensor.to(device=cuda_device, non_blocking=True)
    moved_device = getattr(moved, "device", None)
    if getattr(moved_device, "type", None) == "cuda":
        return moved
    try:
        copied = torch.empty_strided(
            size=tuple(tensor.size()),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=cuda_device,
        )
        copied.copy_(tensor, non_blocking=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to materialize FP8-quantized update tensor on CUDA: "
            f"source_device={device}, target_device={cuda_device}, tensor_type={type(tensor)!r}"
        ) from exc
    copied_device = getattr(copied, "device", None)
    if getattr(copied_device, "type", None) != "cuda":
        raise RuntimeError(
            "FP8 quantization tensor stayed off CUDA after materialization: "
            f"source_device={device}, target_device={cuda_device}, result_device={copied_device}"
        )
    return copied


def _current_cuda_device_for_fp8_quant(tensor):
    try:
        current_device = torch.cuda.current_device()
    except Exception as exc:
        is_available = torch.cuda.is_available()
        raise RuntimeError(
            "Cannot materialize FP8 quantization tensor on CUDA because current CUDA device is unavailable: "
            f"source_device={getattr(tensor, 'device', None)}, cuda_available={is_available}"
        ) from exc
    return f"cuda:{current_device}"
