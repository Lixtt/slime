import importlib
import sys
import types

import numpy as np
import pytest


def _install_rollout_import_stubs():
    torch = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        @property
        def shape(self):
            return self.value.shape

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def reshape(self, *shape):
            return FakeTensor(self.value.reshape(*shape))

        def view(self, *shape):
            return self.reshape(*shape)

        def mean(self, dim=None, keepdim=False):
            return FakeTensor(np.mean(self.value, axis=dim, keepdims=keepdim))

        def std(self, dim=None, keepdim=False):
            axis = dim
            if axis is None:
                ddof = 1 if self.value.size > 1 else 0
            else:
                size = self.value.shape[axis]
                ddof = 1 if size > 1 else 0
            return FakeTensor(np.std(self.value, axis=axis, ddof=ddof, keepdims=keepdim))

        def flatten(self):
            return FakeTensor(self.value.flatten())

        def tolist(self):
            return self.value.tolist()

        def numel(self):
            return self.value.size

        def __sub__(self, other):
            other_value = other.value if isinstance(other, FakeTensor) else other
            return FakeTensor(self.value - other_value)

        def __truediv__(self, other):
            other_value = other.value if isinstance(other, FakeTensor) else other
            return FakeTensor(self.value / other_value)

        def __add__(self, other):
            other_value = other.value if isinstance(other, FakeTensor) else other
            return FakeTensor(self.value + other_value)

        def __radd__(self, other):
            return self.__add__(other)

    torch.Tensor = FakeTensor
    torch.dtype = type("dtype", (), {})
    torch.Size = type("Size", (tuple,), {})
    torch.long = "long"
    torch.int = "int"
    torch.int32 = "int32"
    torch.float = "float"
    torch.float32 = "float32"
    torch.as_tensor = lambda value, dtype=None: FakeTensor(value)
    torch.tensor = lambda value, dtype=None: FakeTensor(value)
    torch.is_tensor = lambda value: isinstance(value, FakeTensor)
    torch.load = lambda *args, **kwargs: None
    torch.zeros_like = lambda value: FakeTensor(np.zeros_like(value.value))
    sys.modules["torch"] = torch

    ray = types.ModuleType("ray")
    ray.remote = lambda obj=None, **kwargs: obj if obj is not None else (lambda cls: cls)
    ray.get = lambda value: value
    ray.put = lambda value: value
    sys.modules["ray"] = ray
    sys.modules["ray.util"] = types.ModuleType("ray.util")
    scheduling = types.ModuleType("ray.util.scheduling_strategies")
    scheduling.PlacementGroupSchedulingStrategy = object
    sys.modules["ray.util.scheduling_strategies"] = scheduling

    sys.modules["sglang"] = types.ModuleType("sglang")
    sys.modules["sglang.srt"] = types.ModuleType("sglang.srt")
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    sys.modules["sglang.srt.constants"] = constants

    engine = types.ModuleType("slime.backends.sglang_utils.sglang_engine")
    engine.SGLangEngine = type("SGLangEngine", (), {})
    sys.modules["slime.backends.sglang_utils.sglang_engine"] = engine

    logging_utils = types.ModuleType("slime.utils.logging_utils")
    logging_utils.configure_logger = lambda *args, **kwargs: None
    logging_utils.init_tracking = lambda *args, **kwargs: None
    sys.modules["slime.utils.logging_utils"] = logging_utils

    health_monitor = types.ModuleType("slime.utils.health_monitor")
    health_monitor.RolloutHealthMonitor = type("RolloutHealthMonitor", (), {})
    sys.modules["slime.utils.health_monitor"] = health_monitor

    http_utils = types.ModuleType("slime.utils.http_utils")
    http_utils._wrap_ipv6 = lambda value: value
    http_utils.find_available_port = lambda *args, **kwargs: 12345
    http_utils.get_host_info = lambda *args, **kwargs: ("host", "127.0.0.1")
    http_utils.init_http_client = lambda *args, **kwargs: None
    sys.modules["slime.utils.http_utils"] = http_utils

    metric_utils = types.ModuleType("slime.utils.metric_utils")
    metric_utils.MetricChecker = type("MetricChecker", (), {})
    metric_utils.compute_pass_rate = lambda *args, **kwargs: {}
    metric_utils.compute_rollout_step = lambda *args, **kwargs: 0
    metric_utils.compute_statistics = lambda *args, **kwargs: {}
    metric_utils.dict_add_prefix = lambda data, prefix: {f"{prefix}{key}": value for key, value in data.items()}
    metric_utils.has_repetition = lambda *args, **kwargs: False
    sys.modules["slime.utils.metric_utils"] = metric_utils

    misc = types.ModuleType("slime.utils.misc")
    misc.Box = type("Box", (dict,), {})
    misc.decode_int32_meta_array = lambda *args, **kwargs: None
    misc.group_by = lambda items, key: {}
    misc.load_function = lambda path: None
    sys.modules["slime.utils.misc"] = misc

    seqlen_balancing = types.ModuleType("slime.utils.seqlen_balancing")
    seqlen_balancing.get_seqlen_balanced_partitions = lambda *args, **kwargs: []
    seqlen_balancing.first_fit_pack = lambda lengths, max_tokens_per_bin: [[i] for i, _ in enumerate(lengths)]
    seqlen_balancing.expand_bins_by_splitting = lambda bins, target_count, lengths: None
    sys.modules["slime.utils.seqlen_balancing"] = seqlen_balancing

    ray_utils = types.ModuleType("slime.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    ray_utils.Lock = type("Lock", (), {})
    ray_utils.add_default_ray_env_vars = lambda env_vars: env_vars
    sys.modules["slime.ray.utils"] = ray_utils


def test_dynamic_global_batch_keeps_short_rollout_until_dummy_padding():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.args = types.SimpleNamespace(
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        disable_rollout_trim_samples=False,
        global_batch_size=4,
        num_steps_per_rollout=1,
        dynamic_history=False,
        use_dynamic_global_batch_size=True,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=100,
        micro_batch_size=1,
        balance_data=False,
    )
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    manager.generate_rollout = object()
    manager.data_source = object()
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None

    def fake_call_rollout_fn(fn, args, rollout_id, data_source, evaluation):
        return RolloutFnTrainOutput(
            samples=[
                [
                    Sample(
                        group_index=0,
                        index=0,
                        tokens=[1, 2],
                        response_length=1,
                        loss_mask=[1],
                        rollout_log_probs=[0.0],
                        reward=1.0,
                    )
                ]
            ],
            metrics={"ok": 1},
        )

    rollout.call_rollout_fn = fake_call_rollout_fn

    data, metrics = manager._get_rollout_data(rollout_id=0)
    assert len(data) == 1
    assert metrics == {"ok": 1}
    assert manager._dynamic_global_batch_size == 2

    train_data = manager._convert_samples_to_train_data(data)
    assert len(train_data["tokens"]) == 2
    assert train_data["group_ids"] == [0, -1]
    assert train_data["loss_masks"] == [[1], [0]]
    assert manager._dynamic_global_batch_size == 2

    rollout_data_refs = manager._split_train_data_by_dp(train_data)
    assert len(rollout_data_refs) == 2
    assert rollout_data_refs[0]["global_batch_sizes"] == [2]
    assert rollout_data_refs[1]["global_batch_sizes"] == [2]


def test_reward_normalization_groups_by_canonical_sample_group_id_even_for_full_batches():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.custom_reward_post_process_func = None
    manager.args = types.SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    samples = [
        Sample(group_index=0, index=0, reward=1.0),
        Sample(group_index=1, index=1, reward=10.0),
        Sample(group_index=0, index=2, reward=3.0),
        Sample(group_index=1, index=3, reward=14.0),
    ]

    raw_rewards, normalized_rewards = manager._post_process_rewards(samples)

    assert raw_rewards == [1.0, 10.0, 3.0, 14.0]
    assert normalized_rewards == [-1.0, -2.0, 1.0, 2.0]


def test_reward_normalization_uses_prompt_group_and_not_trajectory_id():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.custom_reward_post_process_func = None
    manager.args = types.SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    samples = [
        Sample(group_id=10, group_index=0, index=0, reward=1.0),
        Sample(group_id=20, group_index=0, index=1, reward=10.0),
        Sample(group_id=30, group_index=1, index=2, reward=3.0),
        Sample(group_id=40, group_index=1, index=3, reward=14.0),
    ]

    raw_rewards, normalized_rewards = manager._post_process_rewards(samples)

    assert raw_rewards == [1.0, 10.0, 3.0, 14.0]
    assert normalized_rewards == [-4.5, 4.5, -5.5, 5.5]


def test_reward_normalization_counts_compacted_trajectory_once_and_broadcasts_advantage():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.custom_reward_post_process_func = None
    manager.args = types.SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        reward_key=None,
    )
    samples = [
        Sample(group_id=10, group_index=0, index=0, reward=1.0),
        Sample(group_id=10, group_index=0, index=0, reward=1.0),
        Sample(group_id=11, group_index=0, index=1, reward=3.0),
    ]

    raw_rewards, normalized_rewards = manager._post_process_rewards(samples)

    assert raw_rewards == [1.0, 1.0, 3.0]
    assert normalized_rewards == [-1.0, -1.0, 1.0]


def test_reward_normalization_rejects_trajectory_crossing_prompt_groups():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.custom_reward_post_process_func = None
    manager.args = types.SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        reward_key=None,
    )
    samples = [
        Sample(group_id=10, group_index=0, index=0, reward=1.0),
        Sample(group_id=10, group_index=1, index=0, reward=1.0),
    ]

    with pytest.raises(ValueError, match="crosses prompt groups"):
        manager._post_process_rewards(samples)


def test_reward_normalization_ignores_removed_padding_samples():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.custom_reward_post_process_func = None
    manager.args = types.SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=4,
        rollout_batch_size=1,
        reward_key=None,
    )
    samples = [
        Sample(group_index=0, index=0, reward=1.0),
        Sample(group_index=0, index=1, reward=3.0),
        Sample(group_index=0, index=-1, reward=-1.0, remove_sample=True),
        Sample(group_index=0, index=-2, reward=-1.0, remove_sample=True),
    ]

    raw_rewards, normalized_rewards = manager._post_process_rewards(samples)

    assert raw_rewards == [1.0, 3.0, -1.0, -1.0]
    assert normalized_rewards == [-1.0, 1.0, 0.0, 0.0]


def test_convert_samples_keeps_train_metadata_from_sample_metadata():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.train_parallel_config = {
        "dp_size": 1,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    manager.args = types.SimpleNamespace(
        use_dynamic_global_batch_size=False,
        disable_rollout_trim_samples=False,
        global_batch_size=2,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        grpo_std_normalization=False,
    )
    samples = [
        Sample(
            group_index=0,
            index=0,
            tokens=[1, 2],
            response_length=1,
            loss_mask=[1],
            reward=1.0,
            metadata={"train_metadata": {"sample_group_index": 0, "eligible_for_rl": True}},
        ),
        Sample(
            group_index=0,
            index=1,
            tokens=[3, 4],
            response_length=1,
            loss_mask=[1],
            reward=0.0,
            train_metadata={"sample_group_index": 0, "eligible_for_rl": False},
        ),
    ]

    train_data = manager._convert_samples_to_train_data(samples)

    assert train_data["group_ids"] == [0, 1]
    assert train_data["metadata"] == [
        {"sample_group_index": 0, "eligible_for_rl": True},
        {"sample_group_index": 0, "eligible_for_rl": False},
    ]


def test_five_prompts_times_eight_trajectories_do_not_inject_dummy_rows():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.train_parallel_config = {
        "dp_size": 1,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    manager.args = types.SimpleNamespace(
        use_dynamic_global_batch_size=True,
        disable_rollout_trim_samples=False,
        num_steps_per_rollout=1,
        global_batch_size=40,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=8,
        rollout_batch_size=5,
        grpo_std_normalization=False,
    )
    samples = [
        Sample(
            group_index=prompt_index,
            index=trajectory_index,
            tokens=[1, 2],
            response_length=1,
            loss_mask=[1],
            reward=1.0,
        )
        for prompt_index in range(5)
        for trajectory_index in range(prompt_index * 8, (prompt_index + 1) * 8)
    ]

    train_data = manager._convert_samples_to_train_data(samples)

    assert len(train_data["tokens"]) == 40
    assert train_data["group_ids"] == list(range(40))
    assert manager._dynamic_global_batch_size == 40
    assert not any(sample.metadata.get("dummy_removed_sample") for sample in samples)


def test_compacted_rows_share_trajectory_loss_denominator():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.train_parallel_config = {
        "dp_size": 1,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    manager.args = types.SimpleNamespace(
        use_dynamic_global_batch_size=False,
        disable_rollout_trim_samples=False,
        global_batch_size=2,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        grpo_std_normalization=False,
    )
    samples = [
        Sample(group_id=10, group_index=0, index=0, tokens=[1, 2], response_length=1, loss_mask=[1], reward=1.0),
        Sample(
            group_id=10,
            group_index=0,
            index=0,
            tokens=[3, 4, 5],
            response_length=2,
            loss_mask=[1, 0],
            reward=1.0,
        ),
        Sample(group_id=11, group_index=0, index=1, tokens=[6, 7], response_length=1, loss_mask=[1], reward=0.0),
    ]

    train_data = manager._convert_samples_to_train_data(samples)

    assert train_data["group_ids"] == [10, 10, 11]
    assert train_data["rollout_mask_sums"] == [2, 2, 1]


def _segmented_ppo_samples(Sample, *, rewards=(1.0, 1.0), segment_indices=(0, 1), segment_count=2):
    return [
        Sample(
            group_id=10,
            group_index=0,
            index=0,
            reward=reward,
            metadata={
                "trajectory": {
                    "id": 10,
                    "segment_index": segment_index,
                    "segment_count": segment_count,
                }
            },
        )
        for reward, segment_index in zip(rewards, segment_indices, strict=True)
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"gamma": 0.99},
        {"lambd": 0.95},
        {"kl_coef": 0.01},
    ],
)
def test_segmented_ppo_accepts_trajectory_chained_gae_semantics(overrides):
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    samples = _segmented_ppo_samples(Sample)
    values = {"advantage_estimator": "ppo", "gamma": 1.0, "lambd": 1.0, "kl_coef": 0.0}
    values.update(overrides)

    segment_indices, segment_counts = rollout._validate_segmented_ppo_semantics(
        types.SimpleNamespace(**values), samples, [1.0, 1.0], [10, 10]
    )

    assert segment_indices == [0, 1]
    assert segment_counts == [2, 2]


def test_segmented_ppo_rejects_inconsistent_terminal_rewards():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    samples = _segmented_ppo_samples(Sample, rewards=(1.0, 0.0))
    args = types.SimpleNamespace(advantage_estimator="ppo", gamma=1.0, lambd=1.0, kl_coef=0.0)

    with pytest.raises(ValueError, match="inconsistent terminal rewards"):
        rollout._validate_segmented_ppo_semantics(args, samples, [1.0, 0.0], [10, 10])


@pytest.mark.parametrize(
    ("sample_kwargs", "message"),
    [
        ({"segment_count": 3}, "is incomplete"),
        ({"segment_indices": (0, 0)}, "invalid segment indices"),
    ],
)
def test_segmented_ppo_rejects_incomplete_or_duplicate_segments(sample_kwargs, message):
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    samples = _segmented_ppo_samples(Sample, **sample_kwargs)
    args = types.SimpleNamespace(advantage_estimator="ppo", gamma=1.0, lambd=1.0, kl_coef=0.0)

    with pytest.raises(ValueError, match=message):
        rollout._validate_segmented_ppo_semantics(args, samples, [1.0, 1.0], [10, 10])


def test_single_row_ppo_does_not_require_segmented_trajectory_constraints():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    sample = Sample(group_id=10, group_index=0, index=0, reward=1.0)
    args = types.SimpleNamespace(advantage_estimator="ppo", gamma=0.99, lambd=0.95, kl_coef=0.1)

    rollout._validate_segmented_ppo_semantics(args, [sample], [1.0], [10])


def test_ppo_padding_ignores_stale_source_segment_metadata():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    sample = Sample(
        group_id=None,
        index=-1,
        reward=-1.0,
        remove_sample=True,
        metadata={
            "trajectory": {
                "id": 10,
                "segment_index": 4,
                "segment_count": 5,
            }
        },
    )
    args = types.SimpleNamespace(advantage_estimator="ppo", gamma=0.99, lambd=0.95, kl_coef=0.1)

    segment_indices, segment_counts = rollout._validate_segmented_ppo_semantics(
        args,
        [sample],
        [-1.0],
        [0],
    )

    assert segment_indices == [0]
    assert segment_counts == [1]


def test_trajectory_prefix_trimming_keeps_all_compacted_rows():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.utils.types import Sample

    samples = [
        Sample(group_id=10, group_index=0, index=0),
        Sample(group_id=10, group_index=0, index=0),
        Sample(group_id=11, group_index=0, index=1),
        Sample(group_id=12, group_index=1, index=2),
    ]

    trimmed = rollout._trim_to_trajectory_prefix(samples, 2)

    assert trimmed == samples[:3]
    assert rollout._count_distinct_trajectories(trimmed) == 2


def test_dynamic_global_batch_infers_missing_train_parallel_config():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")

    manager = object.__new__(rollout.RolloutManager)
    manager.args = types.SimpleNamespace(
        global_batch_size=8,
        world_size=40,
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=5,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        microbatch_group_size_per_vp_stage=None,
    )

    assert not hasattr(manager, "train_parallel_config")
    assert manager._compute_dynamic_global_batch_size(12, target_steps=8) == 1
    assert manager.train_parallel_config == {
        "dp_size": 1,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }


def test_empty_rollout_batch_is_padded_with_dummy_samples():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.rollout.base_types import RolloutFnTrainOutput

    manager = object.__new__(rollout.RolloutManager)
    manager.args = types.SimpleNamespace(
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        disable_rollout_trim_samples=False,
        global_batch_size=4,
        num_steps_per_rollout=1,
        dynamic_history=False,
        use_dynamic_global_batch_size=True,
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=False,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=100,
        micro_batch_size=1,
        balance_data=False,
    )
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    manager.generate_rollout = object()
    manager.data_source = object()
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None

    def fake_call_rollout_fn(fn, args, rollout_id, data_source, evaluation):
        return RolloutFnTrainOutput(samples=[], metrics={"empty": 1})

    rollout.call_rollout_fn = fake_call_rollout_fn

    data, metrics = manager._get_rollout_data(rollout_id=0)
    assert data == []
    assert metrics == {"empty": 1}

    train_data = manager._convert_samples_to_train_data(data)
    assert len(train_data["tokens"]) == 2
    assert train_data["group_ids"] == [-1, -2]
    assert train_data["sample_indices"] == [-1, -2]
    assert train_data["loss_masks"] == [[0], [0]]
    assert train_data["raw_reward"] == [0.0, 0.0]
    assert manager._dynamic_global_batch_size == 2

    rollout_data_refs = manager._split_train_data_by_dp(train_data)
    assert len(rollout_data_refs) == 2
    assert rollout_data_refs[0]["global_batch_sizes"] == [2]
    assert rollout_data_refs[1]["global_batch_sizes"] == [2]


def test_incomplete_untrimmed_rollout_is_padded_to_global_batch_groups():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.args = types.SimpleNamespace(
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        disable_rollout_trim_samples=True,
        global_batch_size=4,
        num_steps_per_rollout=1,
        dynamic_history=False,
        use_dynamic_global_batch_size=False,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        micro_batch_size=1,
        balance_data=False,
    )
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    manager.generate_rollout = object()
    manager.data_source = object()
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None

    def fake_call_rollout_fn(fn, args, rollout_id, data_source, evaluation):
        return RolloutFnTrainOutput(
            samples=[
                [
                    Sample(
                        group_index=0,
                        index=0,
                        tokens=[1, 2],
                        response_length=1,
                        loss_mask=[1],
                        rollout_log_probs=[0.0],
                        reward=0.5,
                    )
                ]
            ],
            metrics={"partial": 1},
        )

    rollout.call_rollout_fn = fake_call_rollout_fn

    data, metrics = manager._get_rollout_data(rollout_id=0)
    assert len(data) == 1
    assert metrics == {"partial": 1}

    train_data = manager._convert_samples_to_train_data(data)
    assert len(train_data["tokens"]) == 4
    assert train_data["group_ids"] == [0, -1, -2, -3]
    assert train_data["loss_masks"] == [[1], [0], [0], [0]]

    rollout_data_refs = manager._split_train_data_by_dp(train_data)
    assert len(rollout_data_refs) == 2
    assert rollout_data_refs[0]["global_batch_sizes"] == [4]
    assert rollout_data_refs[1]["global_batch_sizes"] == [4]


def test_rollout_trim_samples_defaults_to_enabled_when_arg_missing():
    _install_rollout_import_stubs()
    rollout = importlib.import_module("slime.ray.rollout")
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.utils.types import Sample

    manager = object.__new__(rollout.RolloutManager)
    manager.args = types.SimpleNamespace(
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        global_batch_size=4,
        num_steps_per_rollout=1,
        dynamic_history=False,
    )
    manager.train_parallel_config = {"dp_size": 2}
    manager.generate_rollout = object()
    manager.data_source = object()

    def fake_call_rollout_fn(fn, args, rollout_id, data_source, evaluation):
        return RolloutFnTrainOutput(
            samples=[
                [
                    Sample(
                        group_index=i,
                        index=i,
                        tokens=[1, 2],
                        response_length=1,
                        loss_mask=[1],
                        reward=1.0,
                    )
                ]
                for i in range(5)
            ],
            metrics={"ok": 1},
        )

    rollout.call_rollout_fn = fake_call_rollout_fn

    data, metrics = manager._get_rollout_data(rollout_id=0)

    assert len(data) == 4
    assert metrics == {"ok": 1}
