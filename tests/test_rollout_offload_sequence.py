from pathlib import Path


def test_initial_weight_update_does_not_resume_kv_before_sync():
    source = Path("slime/train.py").read_text()
    before_initial_update = source.split("actor_model.update_weights()", 1)[0]

    assert "rollout_manager.onload_weights.remote()" in before_initial_update
    assert "onload_kv" not in before_initial_update


def test_initial_weight_update_runs_quality_gate_after_weight_onload():
    source = Path("slime/train.py").read_text()
    before_initial_update = source.split("actor_model.update_weights()", 1)[0]

    assert 'run_rollout_generation_quality_gate(rollout_manager, "pre_initial_update")' in before_initial_update
    assert before_initial_update.index("rollout_manager.onload_weights.remote()") < before_initial_update.index(
        '"pre_initial_update"'
    )


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


def test_training_loop_runs_pre_update_quality_gate_after_weight_onload():
    source = Path("slime/train.py").read_text()
    train_loop = source.split("for rollout_id in range", 1)[1]
    before_loop_update = train_loop.split("actor_model.update_weights()", 1)[0]

    assert 'run_rollout_generation_quality_gate(rollout_manager, f"pre_update_{rollout_id}")' in before_loop_update
    assert before_loop_update.index("rollout_manager.onload_weights.remote()") < before_loop_update.index(
        'f"pre_update_{rollout_id}"'
    )


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


def test_distributed_weight_update_streams_from_actor_cpu_backup():
    source = Path("slime/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py").read_text()

    assert "self.weights_getter = weights_getter" in source
    assert "def _iter_named_params_for_current_update" in source
    assert "local_weights = self.weights_getter()" in source
    assert "Missing CPU actor weight backup" in source
    assert "Refusing to read the paused Megatron model tensor" in source
