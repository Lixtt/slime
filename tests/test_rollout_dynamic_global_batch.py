import importlib
import sys
import types


def _install_rollout_import_stubs():
    torch = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

    torch.Tensor = FakeTensor
    torch.dtype = type("dtype", (), {})
    torch.Size = type("Size", (tuple,), {})
    torch.long = "long"
    torch.int = "int"
    torch.int32 = "int32"
    torch.float32 = "float32"
    torch.as_tensor = lambda value, dtype=None: FakeTensor(value)
    torch.is_tensor = lambda value: isinstance(value, FakeTensor)
    torch.load = lambda *args, **kwargs: None
    torch.zeros_like = lambda value: value
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
