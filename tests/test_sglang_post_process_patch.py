from pathlib import Path


SLIME_ROOT = Path(__file__).resolve().parents[1]


def test_fp8_online_post_process_skips_kv_cache_quant_methods():
    for relpath in (
        "docker/patch/latest/sglang.patch",
        "docker/patch/v0.5.12.post1/sglang.patch",
    ):
        text = (SLIME_ROOT / relpath).read_text()

        assert "BaseKVCacheMethod" in text
        assert "should_run_quant_post_process(quant_method)" in text
        assert ") and should_run_quant_post_process(quant_method):" in text


def test_latest_scheduler_tensor_update_and_post_process_cover_draft_worker():
    text = (SLIME_ROOT / "docker/patch/latest/sglang.patch").read_text()

    assert "self.tp_worker.update_weights_from_tensor(recv_req)" in text
    assert "self.draft_worker.update_weights_from_tensor" in text
    assert "self.tp_worker.post_process_weights(recv_req)" in text
    assert "self.draft_worker.post_process_weights(recv_req)" in text
