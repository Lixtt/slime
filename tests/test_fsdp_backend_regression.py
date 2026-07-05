import argparse
import importlib
import sys
import types
import ast
from argparse import Namespace
from pathlib import Path

SLIME_ROOT = Path(__file__).resolve().parents[1]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))


def _install_minimal_ray(monkeypatch):
    ray = types.ModuleType("ray")
    ray.get = lambda value: value
    ray.put = lambda value: value

    class _RemoteActorFactory:
        def options(self, **_kwargs):
            return self

        def remote(self, *_args, **_kwargs):
            class _ActorHandle:
                get_master_addr_and_port = types.SimpleNamespace(remote=lambda: ("127.0.0.1", 12345))

            return _ActorHandle()

    def remote(*args, **_kwargs):
        if args and isinstance(args[0], type):
            return _RemoteActorFactory()
        return lambda _cls: _RemoteActorFactory()

    ray.remote = remote
    monkeypatch.setitem(sys.modules, "ray", ray)

    ray_util = types.ModuleType("ray.util")
    placement_group_mod = types.ModuleType("ray.util.placement_group")
    placement_group_mod.PlacementGroup = type("PlacementGroup", (), {})
    scheduling_mod = types.ModuleType("ray.util.scheduling_strategies")
    scheduling_mod.PlacementGroupSchedulingStrategy = lambda **kwargs: ("strategy", kwargs)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", placement_group_mod)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", scheduling_mod)


def _install_minimal_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.dtype = type("dtype", (), {})
    torch.Size = type("Size", (tuple,), {})
    monkeypatch.setitem(sys.modules, "torch", torch)


def _install_arguments_import_stubs(monkeypatch):
    router_pkg = types.ModuleType("sglang_router")
    launch_router_mod = types.ModuleType("sglang_router.launch_router")

    class RouterArgs:
        @staticmethod
        def add_cli_args(parser, **_kwargs):
            return parser

    launch_router_mod.RouterArgs = RouterArgs
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router_mod)

    sglang_args_mod = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_args_mod.sglang_parse_args = lambda: Namespace()
    sglang_args_mod.validate_args = lambda _args: None
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_args_mod)

    logging_utils_mod = types.ModuleType("slime.utils.logging_utils")
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils_mod)


def test_train_backend_fsdp_dispatches_to_fsdp_actor(monkeypatch):
    _install_minimal_ray(monkeypatch)
    ray_utils = types.ModuleType("slime.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    ray_utils.add_default_ray_env_vars = lambda env_vars: env_vars
    monkeypatch.setitem(sys.modules, "slime.ray.utils", ray_utils)

    fsdp_pkg = types.ModuleType("slime.backends.fsdp_utils")
    fsdp_pkg.__path__ = []
    fsdp_actor_mod = types.ModuleType("slime.backends.fsdp_utils.actor")

    class FSDPTrainRayActor:
        pass

    fsdp_actor_mod.FSDPTrainRayActor = FSDPTrainRayActor
    monkeypatch.setitem(sys.modules, "slime.backends.fsdp_utils", fsdp_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.fsdp_utils.actor", fsdp_actor_mod)

    selected = []

    def remote(*args, **_kwargs):
        if args and isinstance(args[0], type):
            selected.append(args[0])
            return _RemoteActorFactory()

        def wrap(cls):
            selected.append(cls)
            return _RemoteActorFactory()

        return wrap

    class _RemoteActorFactory:
        def options(self, **_kwargs):
            return self

        def remote(self, *_args, **_kwargs):
            class _ActorHandle:
                get_master_addr_and_port = types.SimpleNamespace(remote=lambda: ("127.0.0.1", 12345))

            return _ActorHandle()

    sys.modules.pop("slime.ray.actor_group", None)
    actor_group = importlib.import_module("slime.ray.actor_group")
    monkeypatch.setattr(actor_group.ray, "remote", remote)

    args = Namespace(
        train_backend="fsdp",
        offload_train=False,
        use_routing_replay=False,
        train_env_vars={},
    )
    actor_group.RayTrainGroup(args, 1, 1, (object(), [0], [0]), num_gpus_per_actor=0.4)

    assert selected == [FSDPTrainRayActor]


def test_pre_parse_mode_accepts_fsdp(monkeypatch):
    _install_arguments_import_stubs(monkeypatch)
    sys.modules.pop("slime.utils.arguments", None)
    arguments = importlib.import_module("slime.utils.arguments")
    monkeypatch.setattr(sys, "argv", ["prog", "--train-backend", "fsdp"])

    assert arguments._pre_parse_mode().train_backend == "fsdp"


def test_slime_extra_args_include_true_on_policy_default(monkeypatch):
    _install_arguments_import_stubs(monkeypatch)
    sys.modules.pop("slime.utils.arguments", None)
    arguments = importlib.import_module("slime.utils.arguments")

    parser = argparse.ArgumentParser()
    arguments.get_slime_extra_args_provider()(parser)

    assert parser.parse_args(["--rollout-batch-size", "1"]).true_on_policy_mode is False
    assert parser.parse_args(["--rollout-batch-size", "1", "--true-on-policy-mode"]).true_on_policy_mode is True


def test_slime_extra_args_include_fsdp_lora_defaults(monkeypatch):
    _install_arguments_import_stubs(monkeypatch)
    sys.modules.pop("slime.utils.arguments", None)
    arguments = importlib.import_module("slime.utils.arguments")

    parser = argparse.ArgumentParser()
    arguments.get_slime_extra_args_provider()(parser)

    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    assert defaults.attn_implementation == "flash_attention_2"
    assert defaults.gradient_checkpointing is False
    assert defaults.use_lora is False
    assert defaults.lora_rank == 8
    assert defaults.use_megatron_lora is False
    assert defaults.megatron_lora_save_adapter_only is True

    parsed = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--attn-implementation",
            "eager",
            "--gradient-checkpointing",
            "--use-lora",
            "--lora-rank",
            "128",
            "--lora-alpha",
            "256",
            "--lora-target-modules",
            "q_proj,v_proj",
        ]
    )
    assert parsed.attn_implementation == "eager"
    assert parsed.gradient_checkpointing is True
    assert parsed.use_lora is True
    assert parsed.lora_rank == 128
    assert parsed.lora_alpha == 256
    assert parsed.lora_target_modules == "q_proj,v_proj"

    megatron_parsed = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--use-megatron-lora",
            "--lora-target-modules",
            "linear_q_down_proj,linear_kv_down_proj",
            "--no-megatron-lora-save-adapter-only",
            "--megatron-lora-adapter-load",
            "/tmp/adapter",
            "--megatron-lora-include-experts",
        ]
    )
    assert megatron_parsed.use_megatron_lora is True
    assert megatron_parsed.lora_target_modules == "linear_q_down_proj,linear_kv_down_proj"
    assert megatron_parsed.megatron_lora_save_adapter_only is False
    assert megatron_parsed.megatron_lora_adapter_load == "/tmp/adapter"
    assert megatron_parsed.megatron_lora_include_experts is True


def test_runtime_lora_backend_switches_are_explicit():
    source = (SLIME_ROOT.parent / "a3s-code-rl/scripts/runtime/launch_a3s_code_rl.sh").read_text()

    assert "--use-lora" in source
    assert "--use-megatron-lora" in source
    assert "TRAIN_BACKEND=fsdp" not in source
    assert "Megatron USE_LORA=1 cannot be combined with ONLY_TRAIN_PARAMS_NAME_LIST" in source


def test_fsdp_actor_uses_current_rollout_manager_update_api():
    source = (SLIME_ROOT / "slime/backends/fsdp_utils/actor.py").read_text()

    assert "get_updatable_engines_and_lock" in source
    assert "clear_updatable_num_new_engines" in source
    assert "_unpack_rollout_engine_info" in source


def test_fsdp_actor_reports_non_megatron_parallel_defaults():
    source = (SLIME_ROOT / "slime/backends/fsdp_utils/actor.py").read_text()

    assert '"cp_size": 1' in source
    assert '"vpp_size": 1' in source
    assert '"microbatch_group_size_per_vp_stage": 1' in source


def test_fsdp_actor_train_accepts_actor_group_external_data_kwarg():
    source = (SLIME_ROOT / "slime/backends/fsdp_utils/actor.py").read_text()
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "FSDPTrainRayActor":
            train_defs = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "train"]
            break
    else:
        raise AssertionError("FSDPTrainRayActor class not found")

    assert len(train_defs) == 1
    args = [arg.arg for arg in train_defs[0].args.args]
    assert args[:4] == ["self", "rollout_id", "rollout_data_ref", "external_data"]


def test_fsdp_actor_policy_loss_matches_current_two_value_api():
    source = (SLIME_ROOT / "slime/backends/fsdp_utils/actor.py").read_text()
    module = ast.parse(source)

    assignments = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name) or call.func.id != "compute_policy_loss":
            continue
        assert len(node.targets) == 1
        target = node.targets[0]
        assert isinstance(target, ast.Tuple)
        assignments.append([item.id for item in target.elts if isinstance(item, ast.Name)])

    assert assignments == [["pg_loss", "pg_clipfrac"]]
    assert "ratio_safety_clipfrac" not in source


def test_dataset_len_and_micro_batch_helper(monkeypatch):
    _install_minimal_ray(monkeypatch)
    _install_minimal_torch(monkeypatch)
    timer_mod = types.ModuleType("slime.utils.timer")
    timer_mod.Timer = type("Timer", (), {})
    monkeypatch.setitem(sys.modules, "slime.utils.timer", timer_mod)
    httpx_mod = types.ModuleType("httpx")
    httpx_mod.AsyncClient = type("AsyncClient", (), {})
    monkeypatch.setitem(sys.modules, "httpx", httpx_mod)

    sys.modules.pop("slime.utils.data", None)
    data = importlib.import_module("slime.utils.data")

    dataset = object.__new__(data.Dataset)
    dataset.samples = [object(), object(), object()]

    assert len(dataset) == 3
    assert data.get_minimum_num_micro_batch_size([2, 2, 3], max_tokens_per_gpu=4) == 2
