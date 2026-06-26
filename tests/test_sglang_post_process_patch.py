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
