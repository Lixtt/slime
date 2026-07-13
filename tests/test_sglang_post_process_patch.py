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


def test_latest_tensor_update_single_dispatches_and_cleans_exact_owner():
    text = LATEST_PATCH.read_text()

    assert "Use one tensor deserialization owner" in text
    assert "self.tp_worker.update_weights_from_tensor" not in text
    assert "self.draft_worker.update_weights_from_tensor" not in text
    assert "self.draft_worker.draft_runner.update_weights_from_tensor" in text
    assert "self.target_worker.model_runner.update_weights_from_tensor" in text
    assert "Release the CUDA IPC mapping in the TP process" in text
    assert "Release the CUDA IPC mapping after EAGLE draft and target loads" in text
    assert "torch.cuda.synchronize()" in text
    assert "reductions.shared_cache.clear()" in text
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


def test_build_conda_prefix_mode_does_not_bootstrap_micromamba():
    text = (SLIME_ROOT / "build_conda.sh").read_text()
    prefix_branch = text.split(
        "else\n  # Preserve the standalone bootstrap", maxsplit=1
    )[0]

    assert 'SLIME_ENV_PREFIX="${SLIME_ENV_PREFIX:-}"' in prefix_branch
    assert 'ENV_SELECTOR=(-p "${SLIME_ENV_PREFIX}")' in prefix_branch
    assert 'conda activate "${SLIME_ENV_PREFIX}"' in prefix_branch
    assert 'set +u\n  eval "$("${CONDA_EXE}" shell.bash hook)"' in prefix_branch
    assert 'conda activate "${SLIME_ENV_PREFIX}"\n  set -u' in prefix_branch
    assert "micro.mamba.pm" not in prefix_branch
    assert "source ~/.bashrc" not in prefix_branch


def test_build_conda_native_build_parallelism_is_overridable():
    text = (SLIME_ROOT / "build_conda.sh").read_text()

    assert 'BUILD_MAX_JOBS="${BUILD_MAX_JOBS:-$(nproc)}"' in text
    assert 'MAX_JOBS="${FA2_MAX_JOBS}"' in text
    assert 'MAX_JOBS="${FA3_MAX_JOBS}"' in text
    assert "MAX_JOBS=64" not in text
    assert "MAX_JOBS=96" not in text


def test_build_conda_replaces_cuda13_variants_before_cu129_torch():
    text = (SLIME_ROOT / "build_conda.sh").read_text()

    assert 'SLIME_BUILD_TMPDIR="${SLIME_BUILD_TMPDIR:-}"' in text
    assert 'export TMPDIR="${SLIME_BUILD_TMPDIR}"' in text
    assert "awk -F'==' '/-cu13(==|$)/ {print $1}'" in text
    assert 'pip uninstall -y "${cuda13_packages[@]}"' in text
    cu129_torch = (
        "torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0"
    )
    assert cu129_torch in text
    reinstall = "pip install --force-reinstall cuda-python==12.9"
    assert reinstall in text
    assert text.index("pip uninstall -y") < text.index(cu129_torch)
    assert text.index(cu129_torch) < text.index(reinstall)


def test_build_conda_validates_native_runtime_and_conda_abi_first():
    text = (SLIME_ROOT / "build_conda.sh").read_text()

    assert (
        'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib'
        '${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
    ) in text
    for module in (
        '"flash_attn_3._C"',
        '"hopper.flash_attn_interface"',
        '"sgl_kernel"',
        '"sglang"',
        '"torch_memory_saver"',
        '"transformer_engine.pytorch"',
    ):
        assert module in text
    assert "torch_memory_saver_hook_mode_preload_cu12.abi3.so" in text
    assert 'canonicalize_name(name).endswith("-cu13")' in text


def test_build_conda_uses_cached_retries_for_autoregressive_sglang_runtime():
    text = (SLIME_ROOT / "build_conda.sh").read_text()

    assert 'SLIME_BUILD_CACHE_DIR="${SLIME_BUILD_CACHE_DIR:-}"' in text
    assert 'export PIP_CACHE_DIR="${SLIME_BUILD_CACHE_DIR}"' in text
    assert 'PIP_INSTALL_RETRIES="${PIP_INSTALL_RETRIES:-4}"' in text
    assert "pip install attempt ${attempt}/${PIP_INSTALL_RETRIES} failed" in text
    assert 'pip install -e "python"' in text
    assert 'pip install -e "python[all]"' not in text
