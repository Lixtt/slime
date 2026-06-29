from pathlib import Path


def test_initial_weight_update_does_not_resume_kv_before_sync():
    source = Path("slime/train.py").read_text()
    before_initial_update = source.split("actor_model.update_weights()", 1)[0]

    assert "rollout_manager.onload_weights.remote()" in before_initial_update
    assert "onload_kv" not in before_initial_update


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
