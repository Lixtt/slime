from pathlib import Path


def test_initial_weight_update_does_not_resume_kv_twice():
    source = Path("slime/train.py").read_text()
    initial_update_tail = source.split("actor_model.update_weights()", 1)[1]
    before_post_initial_gate = initial_update_tail.split(
        'run_rollout_generation_quality_gate(rollout_manager, "post_initial_update")',
        1,
    )[0]

    assert "onload_kv" not in before_post_initial_gate


def test_training_loop_still_resumes_kv_after_rollout_offload():
    source = Path("slime/train.py").read_text()
    train_loop = source.split("for rollout_id in range", 1)[1]

    assert "rollout_manager.offload.remote()" in train_loop
    assert "rollout_manager.onload_kv.remote()" in train_loop
