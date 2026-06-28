from slime.utils.ray_env import collect_ray_runtime_env_passthrough, format_env_key_list


def test_collects_sglang_and_low_level_env_vars():
    env = {
        "SGLANG_DSA_FUSE_TOPK": "0",
        "SGLANG_DSA_TOPK_BACKEND": "torch",
        "NCCL_IB_DISABLE": "0",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "MLP_SKIP_SORT_RDMA": "true",
        "UNRELATED": "ignore",
    }

    result = collect_ray_runtime_env_passthrough(env)

    assert result["SGLANG_DSA_FUSE_TOPK"] == "0"
    assert result["SGLANG_DSA_TOPK_BACKEND"] == "torch"
    assert result["NCCL_IB_DISABLE"] == "0"
    assert result["TORCH_NCCL_AVOID_RECORD_STREAMS"] == "1"
    assert result["MLP_SKIP_SORT_RDMA"] == "true"
    assert "UNRELATED" not in result


def test_explicit_passthrough_and_secret_filtering():
    env = {
        "SLIME_RAY_RUNTIME_ENV_PASSTHROUGH": "CUSTOM_FLAG,CUSTOM_TOKEN",
        "CUSTOM_FLAG": "yes",
        "CUSTOM_TOKEN": "secret",
        "SGLANG_API_KEY": "secret",
    }

    result = collect_ray_runtime_env_passthrough(env)

    assert result["CUSTOM_FLAG"] == "yes"
    assert "CUSTOM_TOKEN" not in result
    assert "SGLANG_API_KEY" not in result
    assert format_env_key_list(result) == "CUSTOM_FLAG"
