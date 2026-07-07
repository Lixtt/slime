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


def test_latest_flashmla_kv_rebuilds_metadata_for_actual_q_rows():
    text = (SLIME_ROOT / "docker/patch/latest/sglang.patch").read_text()

    assert "Rebuilding FlashMLA-KV metadata for actual q rows" in text
    assert "flashmla_metadata.num_splits.shape[0] != q_rows + 1" in text
    assert "cache_seqlens = cache_seqlens[:q_rows].contiguous()" in text
    assert "tile_scheduler_metadata=flashmla_metadata.flashmla_metadata" in text
    assert "num_splits=flashmla_metadata.num_splits" in text
