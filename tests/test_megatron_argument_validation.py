import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


def load_arguments_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    arguments_mod = types.ModuleType("megatron.training.arguments")
    tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
    tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
    transformers_mod = types.ModuleType("transformers")

    arguments_mod.parse_args = lambda *args, **kwargs: None
    arguments_mod.validate_args = lambda args: args
    tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "arguments.py"
    module_name = "test_megatron_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_slime_arguments_module(monkeypatch):
    router_pkg_mod = types.ModuleType("sglang_router")
    router_launch_mod = types.ModuleType("sglang_router.launch_router")
    sglang_arguments_mod = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_external_mod = types.ModuleType("slime.backends.sglang_utils.external")
    logging_utils_mod = types.ModuleType("slime.utils.logging_utils")

    router_launch_mod.RouterArgs = object
    sglang_arguments_mod.sglang_parse_args = lambda *args, **kwargs: None
    sglang_arguments_mod.validate_args = lambda args: args
    sglang_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_arguments_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", sglang_external_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "utils" / "arguments.py"
    module_name = "test_slime_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_sglang_arguments_module(monkeypatch):
    sglang_mod = types.ModuleType("sglang")
    srt_mod = types.ModuleType("sglang.srt")
    server_args_mod = types.ModuleType("sglang.srt.server_args")
    router_pkg_mod = types.ModuleType("sglang_router")
    router_launch_mod = types.ModuleType("sglang_router.launch_router")
    http_utils_mod = types.ModuleType("slime.utils.http_utils")

    class ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--data-parallel-size", type=int, default=1)
            parser.add_argument("--pipeline-parallel-size", type=int, default=1)
            parser.add_argument("--expert-parallel-size", type=int, default=1)

    class RouterArgs:
        @staticmethod
        def add_cli_args(parser, *, use_router_prefix=False, exclude_host_port=False):
            prefix = "router-" if use_router_prefix else ""
            parser.add_argument(f"--{prefix}policy", dest="router_policy", default="cache_aware")
            parser.add_argument(
                f"--{prefix}max-concurrent-requests",
                dest="router_max_concurrent_requests",
                type=int,
                default=-1,
            )
            parser.add_argument(f"--{prefix}queue-size", dest="router_queue_size", type=int, default=100)
            parser.add_argument(
                f"--{prefix}queue-timeout-secs",
                dest="router_queue_timeout_secs",
                type=int,
                default=60,
            )

    server_args_mod.ServerArgs = ServerArgs
    router_launch_mod.RouterArgs = RouterArgs
    http_utils_mod._wrap_ipv6 = lambda host: host

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_mod)
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.http_utils", http_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "sglang_utils" / "arguments.py"
    module_name = "test_sglang_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_initialize_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    config_mod = types.ModuleType("megatron.core.config")
    num_micro_mod = types.ModuleType("megatron.core.num_microbatches_calculator")
    training_mod = types.ModuleType("megatron.training")
    global_vars_mod = types.ModuleType("megatron.training.global_vars")

    class MpuStub:
        initialize_model_parallel = staticmethod(lambda *args, **kwargs: None)
        get_pipeline_model_parallel_rank = staticmethod(lambda: 0)
        get_data_parallel_rank = staticmethod(lambda with_context_parallel=False: 0)

    tensor_parallel_mod = types.SimpleNamespace(model_parallel_cuda_manual_seed=lambda *args, **kwargs: None)

    core_mod.mpu = MpuStub
    core_mod.tensor_parallel = tensor_parallel_mod
    config_mod.set_experimental_flag = lambda *args, **kwargs: None
    num_micro_mod.init_num_microbatches_calculator = lambda *args, **kwargs: None
    global_vars_mod._build_tokenizer = lambda *args, **kwargs: None
    global_vars_mod.set_args = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.config", config_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.num_microbatches_calculator", num_micro_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", global_vars_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "initialize.py"
    module_name = "test_megatron_initialize_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_qwen3_6_args(**overrides):
    values = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_qwen3_6_hf_config():
    text_config = types.SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000},
    )
    return types.SimpleNamespace(text_config=text_config)


def make_allgather_cp_args(**overrides):
    values = dict(
        allgather_cp=True,
        context_parallel_size=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_megatron_initialize_fills_newer_tokenizer_defaults(monkeypatch):
    module = load_initialize_module(monkeypatch)

    args = types.SimpleNamespace()
    module._set_megatron_arg_defaults(args)
    assert args.tokenizer_special_tokens is None
    assert args.tokenizer_hf_no_use_fast is False
    assert args.tokenizer_hf_no_include_special_tokens is False
    assert args.tokenizer_sentencepiece_legacy is False
    assert args.tiktoken_pattern is None
    assert args.tiktoken_num_special_tokens == 1000
    assert args.tokenizer_metadata is None
    assert args.special_tokens is None
    assert args.tokenizer_prompt_format is None
    assert args.image_tag_type is None
    assert args.force_system_message is False
    assert args.sft_tokenizer_prompt_format is None
    assert args.hybrid_layer_pattern is None
    assert args.moe_latent_size is None
    assert args.te_precision_config_file is None
    assert args.ddp_param_name_patterns_for_fp32_local_accumulation == []
    assert args.megatron_fsdp_main_params_dtype is module.torch.float32
    assert args.megatron_fsdp_main_grads_dtype is None
    assert args.megatron_fsdp_grad_comm_dtype is None

    args = types.SimpleNamespace(
        tokenizer_special_tokens=["<extra>"],
        tokenizer_hf_no_use_fast=True,
        tokenizer_hf_no_include_special_tokens=True,
        tokenizer_sentencepiece_legacy=True,
        tiktoken_pattern="v2",
        tiktoken_num_special_tokens=7,
        tokenizer_metadata="metadata.json",
        special_tokens={"bos": "<s>"},
        tokenizer_prompt_format="prompt",
        image_tag_type="plain",
        force_system_message=True,
        sft_tokenizer_prompt_format="sft",
        hybrid_layer_pattern="M-M",
        moe_latent_size=128,
        te_precision_config_file="precision.yaml",
        ddp_param_name_patterns_for_fp32_local_accumulation=["linear_q_down_proj"],
        megatron_fsdp_main_params_dtype=module.torch.bfloat16,
        megatron_fsdp_main_grads_dtype=module.torch.float32,
        megatron_fsdp_grad_comm_dtype=module.torch.float16,
    )
    module._set_megatron_arg_defaults(args)
    assert args.tokenizer_special_tokens == ["<extra>"]
    assert args.tokenizer_hf_no_use_fast is True
    assert args.tokenizer_hf_no_include_special_tokens is True
    assert args.tokenizer_sentencepiece_legacy is True
    assert args.tiktoken_pattern == "v2"
    assert args.tiktoken_num_special_tokens == 7
    assert args.tokenizer_metadata == "metadata.json"
    assert args.special_tokens == {"bos": "<s>"}
    assert args.tokenizer_prompt_format == "prompt"
    assert args.image_tag_type == "plain"
    assert args.force_system_message is True
    assert args.sft_tokenizer_prompt_format == "sft"
    assert args.hybrid_layer_pattern == "M-M"
    assert args.moe_latent_size == 128
    assert args.te_precision_config_file == "precision.yaml"
    assert args.ddp_param_name_patterns_for_fp32_local_accumulation == ["linear_q_down_proj"]
    assert args.megatron_fsdp_main_params_dtype is module.torch.bfloat16
    assert args.megatron_fsdp_main_grads_dtype is module.torch.float32
    assert args.megatron_fsdp_grad_comm_dtype is module.torch.float16


@pytest.mark.unit
def test_sglang_parse_args_preserves_router_cli_values(monkeypatch):
    module = load_sglang_arguments_module(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--rollout-num-gpus-per-engine",
            "8",
            "--sglang-pp-size",
            "1",
            "--router-policy",
            "round_robin",
            "--router-max-concurrent-requests",
            "20",
            "--router-queue-size",
            "40",
            "--router-queue-timeout-secs",
            "1800",
        ],
    )

    args = module.sglang_parse_args()

    assert args.router_policy == "round_robin"
    assert args.router_max_concurrent_requests == 20
    assert args.router_queue_size == 40
    assert args.router_queue_timeout_secs == 1800


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parallel_sizes", "expected"),
    [
        ({"sglang_dp_size": 1, "sglang_pp_size": 2, "sglang_ep_size": 4}, (1, 2, 4)),
        (
            {
                "sglang_data_parallel_size": 1,
                "sglang_pipeline_parallel_size": 2,
                "sglang_expert_parallel_size": 4,
            },
            (1, 2, 4),
        ),
    ],
)
def test_sglang_validate_accepts_current_and_legacy_parallel_size_names(
    monkeypatch, parallel_sizes, expected
):
    module = load_sglang_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        rollout_num_gpus_per_engine=8,
        sglang_enable_dp_attention=False,
        sglang_router_ip=None,
        prefill_num_servers=None,
        sglang_config=None,
        rollout_external=False,
        **parallel_sizes,
    )

    module.validate_args(args)

    assert (args.sglang_dp_size, args.sglang_pp_size, args.sglang_ep_size) == expected
    assert args.sglang_tp_size == 4


@pytest.mark.unit
def test_hf_validate_all_moe_skips_dense_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._hf_validate_args(make_qwen3_6_args(), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_moe_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    with pytest.raises(AssertionError, match="moe_intermediate_size"):
        module._hf_validate_args(make_qwen3_6_args(moe_ffn_hidden_size=256), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_dense_intermediate_size_when_moe_has_dense_layers(monkeypatch):
    module = load_arguments_module(monkeypatch)

    args = make_qwen3_6_args(moe_layer_freq=[0] + [1] * 39)

    with pytest.raises(AssertionError, match="intermediate_size"):
        module._hf_validate_args(args, make_qwen3_6_hf_config())


@pytest.mark.unit
def test_allgather_cp_rejects_non_dsa_cp_models(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args()
    hf_config = types.SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3")

    with pytest.raises(ValueError, match="only supported for DSA attention models"):
        module._validate_allgather_cp_supported(args, hf_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hf_config",
    [
        types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v3"),
        types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="glm"),
    ],
)
def test_allgather_cp_allows_dsa_architectures(monkeypatch, hf_config):
    module = load_arguments_module(monkeypatch)

    module._validate_allgather_cp_supported(make_allgather_cp_args(), hf_config)


@pytest.mark.unit
def test_allgather_cp_ignores_cp_size_one(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args(context_parallel_size=1)

    module._validate_allgather_cp_supported(args)


@pytest.mark.unit
def test_update_weight_disk_dir_required_for_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(update_weight_transport="disk", update_weight_disk_dir=None)

    with pytest.raises(ValueError, match="update-weight-disk-dir"):
        module.slime_validate_args(args)


def make_slime_validate_args(**overrides):
    values = dict(
        eval_config=None,
        eval_prompt_data=None,
        kl_coef=0,
        use_kl_loss=False,
        ref_load=None,
        use_opd=False,
        opd_type=None,
        opd_teacher_load=None,
        megatron_to_hf_mode="raw",
        load=None,
        hf_checkpoint="/tmp/hf",
        ref_ckpt_step=None,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        no_save_optim=False,
        no_save_rng=False,
        finetune=False,
        start_rollout_id=None,
        eval_interval=None,
        save_interval=None,
        save=None,
        kl_loss_coef=0,
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_max_tokens_per_gpu=None,
        balance_by_flops=False,
        balance_data=False,
        eps_clip_high=None,
        eps_clip=0.2,
        eval_reward_key=None,
        reward_key="reward",
        dump_details=None,
        save_debug_rollout_data=None,
        save_debug_train_data=None,
        load_debug_rollout_data=None,
        rollout_external_engine_addrs=None,
        debug_train_only=False,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        offload=False,
        offload_train=None,
        offload_rollout=None,
        debug_rollout_only=False,
        colocate=False,
        rollout_num_gpus=8,
        train_memory_margin_bytes=0,
        eval_function_path=None,
        rollout_function_path="custom.rollout",
        num_steps_per_rollout=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=None,
        grpo_std_normalization=True,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        rollout_global_dataset=False,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        custom_config_path=None,
        eval_max_context_len=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        train_backend="megatron",
        release_train=False,
        keep_old_actor=False,
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weights_trainable_only=False,
        update_weights_initial_full_sync=False,
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None,
        update_weight_mode="full",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=True, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_larger_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        colocate=True,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        rollout_num_gpus=12,
    )

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 12
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_without_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=False, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.actor_num_gpus_per_node == 8
    assert args.actor_num_nodes == 1
    assert args.offload_train is False
    assert args.offload_rollout is False


@pytest.mark.unit
def test_update_weight_delta_requires_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="nccl",
        update_weight_local_checkpoint_dir="/local/ckpt",
    )

    with pytest.raises(ValueError, match="requires --update-weight-transport=disk"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir="/local/ckpt",
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_requires_local_checkpoint_dir(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir=None,
    )

    with pytest.raises(ValueError, match="requires --update-weight-local-checkpoint-dir"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weights_trainable_only_requires_only_train_params(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(update_weights_trainable_only=True, only_train_params_name_list=None)

    with pytest.raises(ValueError, match="requires --only-train-params-name-list"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weights_trainable_only_requires_raw_conversion(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weights_trainable_only=True,
        only_train_params_name_list=["self_attention.linear_q_down_proj"],
        megatron_to_hf_mode="bridge",
    )

    with pytest.raises(ValueError, match="raw HF conversion"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weights_initial_full_sync_requires_trainable_only(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(update_weights_initial_full_sync=True)

    with pytest.raises(ValueError, match="requires --update-weights-trainable-only"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_release_train_trainable_only_requires_exact_training_state(monkeypatch):
    monkeypatch.setenv("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE", "1")
    monkeypatch.delenv("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE_TRAINING_STATE", raising=False)
    monkeypatch.delenv("SLIME_MEGATRON_TRAINABLE_ONLY_LOAD_TRAINING_STATE", raising=False)
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        release_train=True,
        save="/checkpoints/trainable",
        save_interval=1,
        update_weight_transport="disk",
        update_weight_disk_dir="/checkpoints/weights",
    )

    with pytest.raises(ValueError, match="requires exact training-state save/load"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_release_train_trainable_only_accepts_exact_training_state(monkeypatch):
    monkeypatch.setenv("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE", "1")
    monkeypatch.setenv("SLIME_MEGATRON_TRAINABLE_ONLY_SAVE_TRAINING_STATE", "1")
    monkeypatch.setenv("SLIME_MEGATRON_TRAINABLE_ONLY_LOAD_TRAINING_STATE", "1")
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        release_train=True,
        save="/checkpoints/trainable",
        save_interval=1,
        update_weight_transport="disk",
        update_weight_disk_dir="/checkpoints/weights",
    )

    module.slime_validate_args(args)

    assert args.offload_train is False
    assert args.offload_rollout is False


@pytest.mark.unit
def test_release_train_validation_reads_exact_state_from_train_env(monkeypatch):
    for name in (
        "SLIME_MEGATRON_TRAINABLE_ONLY_SAVE",
        "SLIME_MEGATRON_TRAINABLE_ONLY_SAVE_TRAINING_STATE",
        "SLIME_MEGATRON_TRAINABLE_ONLY_LOAD_TRAINING_STATE",
    ):
        monkeypatch.delenv(name, raising=False)
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        release_train=True,
        save="/checkpoints/trainable",
        save_interval=1,
        update_weight_transport="disk",
        update_weight_disk_dir="/checkpoints/weights",
        train_env_vars={
            "SLIME_MEGATRON_TRAINABLE_ONLY_SAVE": "1",
            "SLIME_MEGATRON_TRAINABLE_ONLY_SAVE_TRAINING_STATE": "true",
            "SLIME_MEGATRON_TRAINABLE_ONLY_LOAD_TRAINING_STATE": "yes",
        },
    )

    module.slime_validate_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
