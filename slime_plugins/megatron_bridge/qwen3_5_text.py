"""
Text-only Qwen3.5 bridge for megatron.bridge.

This is used only for the isolated text-only retool path. Multimodal runs keep
using the local VLM bridge registered in `qwen3_5.py`.
"""

from __future__ import annotations

import logging

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.core.models.gpt import GPTModel

from slime_plugins.megatron_bridge.qwen35_vl import Qwen35VLModelProvider

logger = logging.getLogger(__name__)

# Use a string so we don't need transformers to have Qwen3.5 at import time
_Qwen3_5HF = "Qwen3_5ForConditionalGeneration"
_Qwen3_5MoeHF = "Qwen3_5MoeForConditionalGeneration"
_Qwen3_5MoeCausalHF = "Qwen3_5MoeForCausalLM"


def _get_text_config(hf_config):
    """Unwrap text_config from VLM config if present."""
    return getattr(hf_config, "text_config", hf_config)


class Qwen35TextModelProvider(Qwen35VLModelProvider):
    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        return self.provide_language_model(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


class _Qwen35GroupedExpertMapping(MegatronParamMapping[torch.Tensor]):
    @property
    def is_expert(self) -> bool:
        return True

    def _target_param(self, megatron_module, name):
        _, target_param = get_module_and_param_from_name(megatron_module, name)
        return target_param

    def _export_shape(self, megatron_module):
        cache_key = f"{self.hf_param}:qwen35_expert_shape"
        if megatron_module is None:
            return self.broadcast_obj_from_pp_rank(None, cache_key)

        model_config = self._get_config(megatron_module)
        shape = (
            model_config.num_moe_experts,
            model_config.hidden_size,
            model_config.moe_ffn_hidden_size,
        )
        return self.broadcast_obj_from_pp_rank(shape, cache_key)

    def _gather_tp_shards(self, megatron_weights, *, dim: int):
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return None
        megatron_weights = self.maybe_dequantize(megatron_weights)
        if self.tp_size == 1:
            return megatron_weights
        return torch.cat(self.gather_from_tp_ranks(megatron_weights), dim=dim)

    def _validate_patterns(self, *args, **kwargs):
        pass


class Qwen35GroupedExpertGateUpMapping(_Qwen35GroupedExpertMapping):
    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
        target_param = self._target_param(megatron_module, "weight1")
        if self.ep_size != 1:
            raise NotImplementedError("Qwen3.5 grouped expert load currently expects expert parallel size 1")

        splits = None
        if self.tp_rank == 0:
            full_weight = hf_weights.transpose(1, 2).contiguous().view(
                target_param.shape[0],
                target_param.shape[1] * self.tp_size,
            )
            full_weight = full_weight.to(dtype=target_param.dtype)
            splits = torch.chunk(full_weight, self.tp_size, dim=1)

        return self.scatter_to_tp_ranks(splits, target_param.shape, target_param.dtype, target_param.device)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        if self.ep_size != 1:
            raise NotImplementedError("Qwen3.5 grouped expert export currently expects expert parallel size 1")

        full_weight = self._gather_tp_shards(megatron_weights, dim=1)
        if full_weight is None:
            return {}

        num_experts, hidden_size, moe_ffn_hidden_size = self._export_shape(megatron_module)
        hf_weight = full_weight.view(num_experts, hidden_size, 2 * moe_ffn_hidden_size)
        hf_weight = hf_weight.transpose(1, 2).contiguous()
        return {str(self.hf_param): hf_weight}


class Qwen35GroupedExpertDownMapping(_Qwen35GroupedExpertMapping):
    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
        target_param = self._target_param(megatron_module, "weight2")
        if self.ep_size != 1:
            raise NotImplementedError("Qwen3.5 grouped expert load currently expects expert parallel size 1")

        splits = None
        if self.tp_rank == 0:
            full_weight = hf_weights.transpose(1, 2).contiguous().view(
                target_param.shape[0] * self.tp_size,
                target_param.shape[1],
            )
            full_weight = full_weight.to(dtype=target_param.dtype)
            splits = torch.chunk(full_weight, self.tp_size, dim=0)

        return self.scatter_to_tp_ranks(splits, target_param.shape, target_param.dtype, target_param.device)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        if self.ep_size != 1:
            raise NotImplementedError("Qwen3.5 grouped expert export currently expects expert parallel size 1")

        full_weight = self._gather_tp_shards(megatron_weights, dim=0)
        if full_weight is None:
            return {}

        num_experts, hidden_size, moe_ffn_hidden_size = self._export_shape(megatron_module)
        hf_weight = full_weight.view(num_experts, moe_ffn_hidden_size, hidden_size)
        hf_weight = hf_weight.transpose(1, 2).contiguous()
        return {str(self.hf_param): hf_weight}


@MegatronModelBridge.register_bridge(source=_Qwen3_5MoeCausalHF, target=GPTModel)
@MegatronModelBridge.register_bridge(source=_Qwen3_5MoeHF, target=GPTModel)
@MegatronModelBridge.register_bridge(source=_Qwen3_5HF, target=GPTModel)
class MegatronQwen35TextBridge(MegatronModelBridge):
    """Bridge between HuggingFace Qwen3.5 and Megatron GPTModel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_pretrained = None

    def load_weights_hf_to_megatron(self, hf_pretrained, model):
        """Store hf_pretrained before calling parent's load method."""
        self.hf_pretrained = hf_pretrained
        return super().load_weights_hf_to_megatron(hf_pretrained, model)

    def provider_bridge(self, hf_pretrained):
        """Create a GPT ModelProvider from Qwen3.5 HF config."""
        hf_config = hf_pretrained.config
        text_config = _get_text_config(hf_config)

        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)

        rope_params = getattr(text_config, "rope_parameters", {}) or {}
        rope_theta = rope_params.get("rope_theta", getattr(text_config, "rope_theta", 10000000))
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 0.25)
        )

        is_moe = bool(getattr(text_config, "num_experts", None))
        ffn_hidden_size = getattr(
            text_config,
            "intermediate_size",
            max(text_config.hidden_size * 4, getattr(text_config, "moe_intermediate_size", 0)),
        )

        provider = Qwen35TextModelProvider(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=getattr(text_config, "head_dim", 256),
            init_method_std=getattr(text_config, "initializer_range", 0.02),
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rope_theta,
            rotary_percent=partial_rotary_factor,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", True),
            vocab_size=text_config.vocab_size,
            seq_length=getattr(text_config, "max_position_embeddings", 262144),
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            qk_layernorm=True,
            attention_output_gate=True,
        )

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
        provider.add_bias_linear = False
        provider.hidden_dropout = 0.0
        provider.layernorm_zero_centered_gamma = False
        provider.experimental_attention_variant = "gated_delta_net"
        provider.linear_attention_freq = getattr(text_config, "full_attention_interval", 4)
        provider.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)
        provider.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        provider.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
        provider.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        provider.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 48)
        provider.position_embedding_type = "mrope"
        provider.hf_text_config = text_config
        provider.head_dim = getattr(text_config, "head_dim", 256)
        provider.language_max_sequence_length = getattr(
            text_config,
            "max_position_embeddings",
            getattr(text_config, "seq_length", 262144),
        )
        provider.mrope_section = list(getattr(text_config, "rope_parameters", {}).get("mrope_section", [11, 11, 10]))

        if is_moe:
            provider.num_moe_experts = text_config.num_experts
            provider.num_experts = text_config.num_experts
            provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
            provider.moe_router_topk = text_config.num_experts_per_tok
            provider.moe_shared_expert_intermediate_size = getattr(
                text_config,
                "shared_expert_intermediate_size",
                None,
            )
            provider.moe_shared_expert_gate = True
            provider.moe_grouped_gemm = True
            provider.moe_layer_freq = [1] * text_config.num_hidden_layers
            provider.moe_router_score_function = "softmax"
            provider.moe_router_dtype = "fp32"
            provider.moe_aux_loss_coeff = 0
            provider.moe_router_load_balancing_type = "none"
            provider.moe_token_dispatcher_type = "alltoall"

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Weight mappings from HF Qwen3.5 to Megatron format."""
        if self.hf_pretrained is None:
            raise RuntimeError(
                "hf_pretrained is not set. Ensure load_weights_hf_to_megatron() "
                "is called before mapping_registry()."
            )

        hf_config = self.hf_pretrained.config
        is_vlm = hasattr(hf_config, "text_config")
        pfx = "model.language_model" if is_vlm else "model"

        auto_param_mappings = {
            f"embedding.word_embeddings.weight": f"{pfx}.embed_tokens.weight",
            f"output_layer.weight": "lm_head.weight",
            f"decoder.final_layernorm.weight": f"{pfx}.norm.weight",
            f"decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": f"{pfx}.layers.*.input_layernorm.weight",
            f"decoder.layers.*.input_layernorm.weight": f"{pfx}.layers.*.input_layernorm.weight",
            f"decoder.layers.*.self_attention.linear_proj.weight": f"{pfx}.layers.*.self_attn.o_proj.weight",
            f"decoder.layers.*.self_attention.q_layernorm.weight": f"{pfx}.layers.*.self_attn.q_norm.weight",
            f"decoder.layers.*.self_attention.k_layernorm.weight": f"{pfx}.layers.*.self_attn.k_norm.weight",
            f"decoder.layers.*.mlp.linear_fc1.layer_norm_weight": f"{pfx}.layers.*.post_attention_layernorm.weight",
            f"decoder.layers.*.pre_mlp_layernorm.weight": f"{pfx}.layers.*.post_attention_layernorm.weight",
            f"decoder.layers.*.mlp.linear_fc2.weight": f"{pfx}.layers.*.mlp.down_proj.weight",
        }

        text_config = _get_text_config(hf_config)
        is_moe = bool(getattr(text_config, "num_experts", None))
        if is_moe:
            auto_param_mappings.pop("decoder.layers.*.mlp.linear_fc1.weight", None)
            auto_param_mappings.pop("decoder.layers.*.mlp.linear_fc2.weight", None)
            auto_param_mappings.update(
                {
                    "decoder.layers.*.mlp.router.weight": f"{pfx}.layers.*.mlp.gate.weight",
                    "decoder.layers.*.mlp.experts.linear_fc1.weight": f"{pfx}.layers.*.mlp.experts.gate_up_proj",
                    "decoder.layers.*.mlp.experts.linear_fc2.weight": f"{pfx}.layers.*.mlp.experts.down_proj",
                    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": f"{pfx}.layers.*.mlp.shared_expert.down_proj.weight",
                }
            )

        replicated_param_mappings = {
            "decoder.layers.*.self_attention.input_layernorm.weight": f"{pfx}.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_attn.A_log": f"{pfx}.layers.*.linear_attn.A_log",
            "decoder.layers.*.self_attention.linear_attn.conv1d.weight": f"{pfx}.layers.*.linear_attn.conv1d.weight",
            "decoder.layers.*.self_attention.linear_attn.dt_bias": f"{pfx}.layers.*.linear_attn.dt_bias",
            "decoder.layers.*.self_attention.linear_attn.in_proj_a.weight": f"{pfx}.layers.*.linear_attn.in_proj_a.weight",
            "decoder.layers.*.self_attention.linear_attn.in_proj_b.weight": f"{pfx}.layers.*.linear_attn.in_proj_b.weight",
            "decoder.layers.*.self_attention.linear_attn.in_proj_qkv.weight": f"{pfx}.layers.*.linear_attn.in_proj_qkv.weight",
            "decoder.layers.*.self_attention.linear_attn.in_proj_z.weight": f"{pfx}.layers.*.linear_attn.in_proj_z.weight",
            "decoder.layers.*.self_attention.linear_attn.norm.weight": f"{pfx}.layers.*.linear_attn.norm.weight",
            "decoder.layers.*.self_attention.linear_attn.out_proj.weight": f"{pfx}.layers.*.linear_attn.out_proj.weight",
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in auto_param_mappings.items()
        ]
        mapping_list.extend(
            [
                ReplicatedMapping(megatron_param=megatron_param, hf_param=hf_param)
                for megatron_param, hf_param in replicated_param_mappings.items()
            ]
        )

        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q=f"{pfx}.layers.*.self_attn.q_proj.weight",
                k=f"{pfx}.layers.*.self_attn.k_proj.weight",
                v=f"{pfx}.layers.*.self_attn.v_proj.weight",
            )
        )

        if is_moe:
            mapping_list.append(
                Qwen35GroupedExpertGateUpMapping(
                    megatron_param="decoder.layers.*.mlp.experts.weight1",
                    hf_param=f"{pfx}.layers.*.mlp.experts.gate_up_proj",
                )
            )
            mapping_list.append(
                Qwen35GroupedExpertDownMapping(
                    megatron_param="decoder.layers.*.mlp.experts.weight2",
                    hf_param=f"{pfx}.layers.*.mlp.experts.down_proj",
                )
            )
            mapping_list.append(
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate=f"{pfx}.layers.*.mlp.shared_expert.gate_proj.weight",
                    up=f"{pfx}.layers.*.mlp.shared_expert.up_proj.weight",
                )
            )
            mapping_list.append(
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.gate_weight",
                    hf_param=f"{pfx}.layers.*.mlp.shared_expert_gate.weight",
                )
            )
        else:
            mapping_list.append(
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate=f"{pfx}.layers.*.mlp.gate_proj.weight",
                    up=f"{pfx}.layers.*.mlp.up_proj.weight",
                )
            )

        return MegatronMappingRegistry(*mapping_list)
