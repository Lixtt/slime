from pathlib import Path


def test_initial_weight_update_does_not_resume_kv_before_sync():
    source = Path("slime/train.py").read_text()
    before_initial_update = source.split("actor_model.update_weights()", 1)[0]

    assert "rollout_manager.onload_weights.remote()" in before_initial_update
    assert "onload_kv" not in before_initial_update


def test_initial_quality_gate_runs_before_rollout_offload_and_actor_creation():
    source = Path("slime/train.py").read_text()
    quality_gate = '"pre_initial_update",\n        debug_train_only=args.debug_train_only,'
    rollout_offload = "rollout_manager.offload.remote()"
    actor_creation = "create_training_models(args, pgs, rollout_manager)"

    assert source.index(quality_gate) < source.index(rollout_offload) < source.index(actor_creation)


def test_initial_weight_update_resumes_kv_before_post_update_gate():
    source = Path("slime/train.py").read_text()
    initial_update_tail = source.split("actor_model.update_weights()", 1)[1]
    before_post_initial_gate = initial_update_tail.split(
        'run_rollout_generation_quality_gate(rollout_manager, "post_initial_update")',
        1,
    )[0]

    assert "rollout_manager.onload_kv.remote()" in before_post_initial_gate


def test_training_loop_still_resumes_kv_after_rollout_offload():
    source = Path("slime/train.py").read_text()
    train_loop = source.split("for rollout_id in range", 1)[1]

    assert "rollout_manager.offload.remote()" in train_loop
    assert "rollout_manager.onload_kv.remote()" in train_loop


def test_training_loop_runs_pre_update_quality_gate_before_rollout_offload():
    source = Path("slime/train.py").read_text()
    train_loop = source.split("for rollout_id in range", 1)[1]
    generate = "rollout_manager.generate.remote(rollout_id)"
    quality_gate = 'f"pre_update_{rollout_id}",\n            debug_train_only=args.debug_train_only,'
    rollout_offload = "rollout_manager.offload.remote()"

    assert train_loop.index(generate) < train_loop.index(quality_gate) < train_loop.index(rollout_offload)


def test_rollout_manager_creation_does_not_hide_initial_offload():
    source = Path("slime/slime/ray/placement_group.py").read_text()
    create_body = source.split("def create_rollout_manager(args, pg):", 1)[1]

    assert "rollout_manager.offload.remote()" not in create_body


def test_rollout_offload_pauses_generation_before_memory_release():
    source = Path("slime/slime/ray/rollout.py").read_text()
    offload_body = source.split("    def offload(self):", 1)[1].split("\n    def onload", 1)[0]

    assert "engine.pause_generation.remote()" in offload_body
    assert "engine.release_memory_occupation.remote()" in offload_body
    assert offload_body.index("pause_generation") < offload_body.index("release_memory_occupation")


def test_rollout_onload_continues_generation_after_kv_cache_resume():
    source = Path("slime/slime/ray/rollout.py").read_text()
    onload_body = source.split("    def onload(self, tags: list[str] | None = None):", 1)[1].split(
        "\n    def onload_weights_from_disk",
        1,
    )[0]

    assert "engine.resume_memory_occupation.remote(tags=tags)" in onload_body
    assert "GPU_MEMORY_TYPE_KV_CACHE" in onload_body
    assert "GPU_MEMORY_TYPE_CUDA_GRAPH" in onload_body
    assert "engine.continue_generation.remote()" in onload_body
    assert onload_body.index("resume_memory_occupation") < onload_body.index("continue_generation")


def test_sglang_engine_actor_has_startup_retries():
    source = Path("slime/slime/ray/rollout.py").read_text()
    start_engines_body = source.split("    def start_engines", 1)[1].split("\n    def offload", 1)[0]

    assert 'SLIME_SGLANG_ENGINE_MAX_RESTARTS", 2' in start_engines_body
    assert 'SLIME_SGLANG_ENGINE_MAX_TASK_RETRIES", 2' in start_engines_body
    assert "max_restarts=max_restarts" in start_engines_body
    assert "max_task_retries=max_task_retries" in start_engines_body


def test_megatron_update_does_not_resume_all_paused_actor_weights_before_sync():
    source = Path("slime/slime/backends/megatron_utils/actor.py").read_text()
    update_body = source.split("    def update_weights(self) -> None:", 1)[1].split(
        "\n    def load_other_checkpoint",
        1,
    )[0]

    assert "_resume_tms_train_weights_for_update" not in source
    assert "train_weight_context" not in update_body
    assert "with offload_context, lora_context:" in update_body
    assert "self.weight_updater.update_weights()" in update_body
