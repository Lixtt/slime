#!/bin/bash

set -euxo pipefail

SLIME_ENV_PREFIX="${SLIME_ENV_PREFIX:-}"
SLIME_ENV_NAME="${SLIME_ENV_NAME:-slime}"
BUILD_MAX_JOBS="${BUILD_MAX_JOBS:-$(nproc)}"
FA2_MAX_JOBS="${FA2_MAX_JOBS:-${BUILD_MAX_JOBS}}"
FA3_MAX_JOBS="${FA3_MAX_JOBS:-${BUILD_MAX_JOBS}}"
SLIME_BUILD_TMPDIR="${SLIME_BUILD_TMPDIR:-}"

if [[ -n "${SLIME_BUILD_TMPDIR}" ]]; then
  mkdir -p "${SLIME_BUILD_TMPDIR}"
  export TMPDIR="${SLIME_BUILD_TMPDIR}"
fi
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

env_install() {
  "${ENV_MANAGER_BIN}" install "${ENV_SELECTOR[@]}" "$@"
}

if [[ -n "${SLIME_ENV_PREFIX}" ]]; then
  # Prefix mode is intended for shared clusters and CI: use an existing conda
  # installation without downloading micromamba or changing shell dotfiles.
  CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
  if [[ -z "${CONDA_EXE}" || ! -x "${CONDA_EXE}" ]]; then
    echo "SLIME_ENV_PREFIX requires an executable CONDA_EXE" >&2
    exit 2
  fi
  ENV_MANAGER_BIN="${CONDA_EXE}"
  ENV_SELECTOR=(-p "${SLIME_ENV_PREFIX}")
  if [[ ! -x "${SLIME_ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_EXE}" create "${ENV_SELECTOR[@]}" python=3.12 pip -c conda-forge -y
  fi
  # Third-party conda activate hooks may read optional variables without
  # default expansions (CUDA's hook reads NVCC_PREPEND_FLAGS, for example).
  set +u
  eval "$("${CONDA_EXE}" shell.bash hook)"
  conda activate "${SLIME_ENV_PREFIX}"
  set -u
else
  # Preserve the standalone bootstrap used by the upstream development image.
  yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
  export PS1=tmp
  mkdir -p /root/.cargo/
  touch /root/.cargo/env
  set +u
  source ~/.bashrc
  set -u

  # The installer may write the `nodefaults` meta-tag as a real channel.
  if [[ -f ~/.condarc ]]; then
    sed -i '/^\s*-\s*nodefaults\s*$/d' ~/.condarc
  fi

  ENV_MANAGER_BIN="$(command -v micromamba)"
  ENV_SELECTOR=(-n "${SLIME_ENV_NAME}")
  "${ENV_MANAGER_BIN}" create "${ENV_SELECTOR[@]}" python=3.12 pip -c conda-forge -y
  set +u
  micromamba activate "${SLIME_ENV_NAME}"
  set -u
fi
export CUDA_HOME="$CONDA_PREFIX"

# Keep these in sync with docker/Dockerfile:
#   - SGLANG_IMAGE_TAG (ARG)            -> SGLANG_VERSION below
#   - MEGATRON_COMMIT (ARG)             -> MEGATRON_COMMIT below
#   - PATCH_VERSION (ARG, default "latest") -> PATCH_VERSION below
export SGLANG_VERSION="v0.5.14"
export SGLANG_COMMIT="49e384ce9d304648e9959666ecb8ce8cd98d0deb"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="latest"

export BASE_DIR=${BASE_DIR:-"/root"}
export SGLANG_DIR=${SGLANG_DIR:-"$BASE_DIR/sglang-v0.5.14-slime"}
export SLIME_DIR=${SLIME_DIR:-"$BASE_DIR/slime"}
export SLIME_REPO_URL=${SLIME_REPO_URL:-"https://github.com/Lixtt/slime.git"}
cd $BASE_DIR

if [ ! -d "$SLIME_DIR/.git" ]; then
  git clone "$SLIME_REPO_URL" "$SLIME_DIR"
fi
if [ -n "$(git -C "$SLIME_DIR" status --porcelain --untracked-files=no)" ]; then
  echo "Refusing to install from a dirty Slime checkout: $SLIME_DIR" >&2
  exit 1
fi
SGLANG_PATCH="$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch"

# install cuda 12.9 as it's the default cuda version for torch
env_install \
  cuda=12.9.1 \
  cuda-nvtx=12.9.79 \
  cuda-nvtx-dev=12.9.79 \
  nccl \
  -c nvidia/label/cuda-12.9.1 \
  -c nvidia \
  -c conda-forge \
  -y
env_install -c conda-forge cudnn -y
# sglang's editable install builds a Rust extension (sglang-grpc via
# setuptools-rust), so the conda env needs a working rustc + cargo.
env_install -c conda-forge rust -y

# install sglang. The Dockerfile starts FROM lmsysorg/sglang:v0.5.14-cu129
# which already has sglang installed with cu129-built native kernels; we have
# to install it ourselves here. Two follow-up steps clean up the cu13 spill:
#   1. force-reinstall torch / sglang-kernel / sgl-deep-gemm to their +cu129
#      wheels (pypi defaults are cu13);
#   2. uninstall the cu13 nvidia-* runtime libs sglang dragged in, then
#      reinstall the cu12 equivalents to repair the `site-packages/nvidia/*`
#      shared dirs (pip uninstall stomps libs co-owned across cu12/cu13).
if [ ! -d "$SGLANG_DIR/.git" ]; then
  cd $BASE_DIR
  git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
fi
cd "$SGLANG_DIR"
git checkout --detach ${SGLANG_COMMIT}
if [ -n "$(git status --porcelain --untracked-files=no)" ] \
  && ! git apply --reverse --check "$SGLANG_PATCH" 2>/dev/null; then
  echo "Refusing to install from an unexpectedly dirty SGLang checkout: $SGLANG_DIR" >&2
  exit 1
fi
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.4 sgl-deep-gemm==0.1.3 \
  --index-url https://docs.sglang.ai/whl/cu129/
pip uninstall -y \
  cuda-bindings \
  cuda-core \
  cuda-python \
  cuda-toolkit \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 \
  nvidia-cufft-cu12 \
  nvidia-cufile-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple
pip install --force-reinstall cuda-python==12.9


pip install cmake ninja

# flash attn 2 (matches Dockerfile)
MAX_JOBS="${FA2_MAX_JOBS}" pip -v install flash-attn==2.8.3 --no-build-isolation

# flash attn 3 (matches Dockerfile and TransformerEngine 2.16 CP APIs)
if [ ! -d "$BASE_DIR/flash-attention/.git" ]; then
  git clone https://github.com/Dao-AILab/flash-attention.git "$BASE_DIR/flash-attention"
fi
cd "$BASE_DIR/flash-attention"
git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c
git submodule update --init
cd hopper
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${FA3_MAX_JOBS}" pip -v install . --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install flash-linear-attention==0.4.2
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
# tilelang (matches Dockerfile)
pip install tilelang==0.1.8 -f https://tile-ai.github.io/whl/nightly/cu128/

pip install --no-build-isolation "transformer_engine[pytorch]==2.16.1"

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
# --no-build-isolation: TMS's setup.py needs to find nvcc + headers + the
# installed torch to build its cu${TMS_CUDA_MAJOR} native hook; pip's default
# PEP 517 build venv hides them, so the wheel comes out python-only (~46KB)
# and sglang trips `Only hook_mode=preload supports pauseable CUDA Graph`
# because the preload .so was never compiled in.
pip install -v git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
  --no-cache-dir --force-reinstall --no-build-isolation
# matches Dockerfile (different fork/branch from older build_conda.sh)
pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# megatron
cd $BASE_DIR
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
# pre-install Megatron's build deps explicitly since we use --no-build-isolation
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
# --no-build-isolation: setup.py builds a C++ extension (megatron.core.datasets.helpers_cpp)
# that subprocess-shells `python3 -m pybind11`; without isolation pip uses the
# current env's python which already has pybind11 installed. Otherwise the ext
# is marked optional and silently skipped, which breaks GPT dataset loading.
cd $BASE_DIR/Megatron-LM && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

# install slime and apply patches

cd $SLIME_DIR
# Install slime's pure-python runtime deps first (wandb, ray, accelerate,
# transformers, etc.) from its requirements.txt, then install slime itself
# with --no-deps so pip doesn't re-resolve and stomp our pinned native libs
# (torch+cu129, sglang-kernel+cu129, ...). The Dockerfile does the same thing
# in two RUN layers (line ~71 + line ~124).
pip install -r requirements.txt
pip install "ray[default]>=2.55.1"
pip install -e . --no-deps

# int4_qat kernel (matches Dockerfile)
cd $SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat
pip install . --no-build-isolation

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"
# kernels 0.15.x trips a ValueError("Either a revision or a version must be
# specified") on `transformers.integrations.hub_kernels` import; pin to <0.15
# so `import sglang` works at runtime.
pip install "kernels<0.15.0"

# apply patch (matches Dockerfile: --3way + fail on conflicts)
cd "$SGLANG_DIR"
if git apply --reverse --check "$SGLANG_PATCH" 2>/dev/null; then
  echo "SGLang patch already applied"
elif git apply --check "$SGLANG_PATCH"; then
  git update-index --refresh || true
  git apply "$SGLANG_PATCH" --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "sglang patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "SGLang patch is neither applicable nor already applied" >&2
  exit 1
fi
cd $BASE_DIR/Megatron-LM
if git apply --reverse --check $SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch 2>/dev/null; then
  echo "Megatron patch already applied"
elif git apply --check $SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch; then
  git update-index --refresh || true
  git apply $SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "Megatron patch is neither applicable nor already applied" >&2
  exit 1
fi

python - <<'PY'
from importlib import metadata
from packaging.version import Version

expected_exact = {
    "sglang": "0.5.14",
    "sglang-kernel": "0.4.4",
    "torch": "2.11.0",
    "transformers": "5.8.1",
    "transformer-engine": "2.16.1",
}
for package, expected in expected_exact.items():
    actual = metadata.version(package)
    if Version(actual).base_version != expected:
        raise SystemExit(f"{package}: expected {expected}, got {actual}")
    print(f"{package}={actual}")

ray_version = metadata.version("ray")
if Version(ray_version) < Version("2.55.1"):
    raise SystemExit(f"ray: expected >=2.55.1, got {ray_version}")
print(f"ray={ray_version}")
PY
