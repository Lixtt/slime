import subprocess
from pathlib import Path


SLIME_ROOT = Path(__file__).resolve().parents[1]
LATEST_PATCH = SLIME_ROOT / "docker/patch/latest/sglang.patch"


def test_latest_sglang_patch_is_well_formed():
    subprocess.run(
        ["git", "apply", "--numstat", str(LATEST_PATCH)],
        cwd=SLIME_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_fp8_online_post_process_skips_kv_cache_quant_methods():
    latest_text = LATEST_PATCH.read_text()
    assert "BaseKVCacheMethod" in latest_text
    assert "not isinstance(quant_method, BaseKVCacheMethod)" in latest_text

    legacy_text = (
        SLIME_ROOT / "docker/patch/v0.5.12.post1/sglang.patch"
    ).read_text()
    assert "BaseKVCacheMethod" in legacy_text
    assert "should_run_quant_post_process(quant_method)" in legacy_text
    assert ") and should_run_quant_post_process(quant_method):" in legacy_text


def test_latest_scheduler_tensor_update_and_post_process_cover_draft_worker():
    text = LATEST_PATCH.read_text()

    assert "self.tp_worker.update_weights_from_tensor(recv_req)" in text
    assert "self.draft_worker.update_weights_from_tensor" in text
    assert "self.tp_worker.post_process_weights(recv_req)" in text
    assert "self.draft_worker.post_process_weights(recv_req)" in text


def test_latest_builds_use_one_pinned_sglang_patch():
    dockerfile = (SLIME_ROOT / "docker/Dockerfile").read_text()
    build_conda = (SLIME_ROOT / "build_conda.sh").read_text()

    assert "lmsysorg/sglang:${SGLANG_IMAGE_TAG}" in dockerfile
    assert "SGLANG_IMAGE_TAG=v0.5.14-cu129" in dockerfile
    assert "SGLANG_VERSION=\"v0.5.14\"" in build_conda
    assert "sglang-kernel==0.4.4" in build_conda
    assert 'ray[default]>=2.55.1' in build_conda
    for obsolete_patch in (
        "sglang-top_p.patch",
        "sglang-release_hicache.patch",
        "sglang-pull_weights.patch",
    ):
        assert obsolete_patch not in dockerfile
        assert not (SLIME_ROOT / "docker/patch/latest" / obsolete_patch).exists()
