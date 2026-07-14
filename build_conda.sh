#!/bin/bash

set -euxo pipefail

SLIME_ENV_PREFIX="${SLIME_ENV_PREFIX:-}"
SLIME_ENV_NAME="${SLIME_ENV_NAME:-slime}"
BUILD_MAX_JOBS="${BUILD_MAX_JOBS:-$(nproc)}"
FA2_MAX_JOBS="${FA2_MAX_JOBS:-${BUILD_MAX_JOBS}}"
FA3_MAX_JOBS="${FA3_MAX_JOBS:-${BUILD_MAX_JOBS}}"
FA3_DISABLE_SM80="${FA3_DISABLE_SM80:-1}"
SLIME_TORCH_CUDA_ARCH_LIST="${SLIME_TORCH_CUDA_ARCH_LIST:-9.0}"
FA2_CUDA_ARCHS="${FA2_CUDA_ARCHS:-90}"
SLIME_BUILD_TMPDIR="${SLIME_BUILD_TMPDIR:-}"
SLIME_BUILD_CACHE_DIR="${SLIME_BUILD_CACHE_DIR:-}"
PIP_INSTALL_RETRIES="${PIP_INSTALL_RETRIES:-4}"
PIP_RETRY_SLEEP_SEC="${PIP_RETRY_SLEEP_SEC:-10}"
NVIDIA_PYPI_INDEX_URL="${NVIDIA_PYPI_INDEX_URL:-https://pypi.nvidia.com}"
GENERAL_PYPI_INDEX_URL="${GENERAL_PYPI_INDEX_URL:-${PIP_INDEX_URL:-}}"

if [[ -n "${SLIME_BUILD_TMPDIR}" ]]; then
  mkdir -p "${SLIME_BUILD_TMPDIR}"
  export TMPDIR="${SLIME_BUILD_TMPDIR}"
fi
if [[ -n "${SLIME_BUILD_CACHE_DIR}" ]]; then
  mkdir -p "${SLIME_BUILD_CACHE_DIR}"
  export PIP_CACHE_DIR="${SLIME_BUILD_CACHE_DIR}"
  unset PIP_NO_CACHE_DIR
else
  export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
fi

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
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CUDA_TARGET_INCLUDE_DIR="${CUDA_TARGET_INCLUDE_DIR:-${CONDA_PREFIX}/targets/x86_64-linux/include}"
if [[ ! -f "${CUDA_TARGET_INCLUDE_DIR}/cuda_runtime.h" ]]; then
  echo "Missing CUDA runtime headers: ${CUDA_TARGET_INCLUDE_DIR}/cuda_runtime.h" >&2
  exit 2
fi
export CPATH="${CUDA_TARGET_INCLUDE_DIR}${CPATH:+:${CPATH}}"

# CUDA 12.9 rejects GCC 14, which current conda-forge activation hooks may
# select through x86_64-conda-linux-gnu-c++. Use the host's supported compiler
# for CUDA extensions while retaining conda's headers, libraries, and linker
# flags.
CUDA_HOST_CC="${CUDA_HOST_CC:-/usr/bin/gcc}"
CUDA_HOST_CXX="${CUDA_HOST_CXX:-/usr/bin/g++}"
if [[ ! -x "${CUDA_HOST_CC}" || ! -x "${CUDA_HOST_CXX}" ]]; then
  echo "Missing CUDA host compiler: CC=${CUDA_HOST_CC} CXX=${CUDA_HOST_CXX}" >&2
  exit 2
fi
cuda_host_cxx_major="$("${CUDA_HOST_CXX}" -dumpfullversion -dumpversion | cut -d. -f1)"
if [[ ! "${cuda_host_cxx_major}" =~ ^[0-9]+$ ]] || (( cuda_host_cxx_major >= 14 )); then
  echo "CUDA 12.9 requires a host C++ compiler older than GCC 14; got ${CUDA_HOST_CXX} major=${cuda_host_cxx_major}" >&2
  exit 2
fi
export CC="${CUDA_HOST_CC}"
export CXX="${CUDA_HOST_CXX}"
export NVCC_PREPEND_FLAGS="-I${CUDA_TARGET_INCLUDE_DIR} -ccbin=${CUDA_HOST_CXX}"
export TORCH_CUDA_ARCH_LIST="${SLIME_TORCH_CUDA_ARCH_LIST}"

# Retry complete pip install transactions. Streaming resets from cluster-local
# mirrors otherwise force operators to rerun this multi-hour build by hand.
pip() {
  local is_install=0
  local attempt
  [[ "${1:-}" == "install" || "${2:-}" == "install" ]] && is_install=1
  if (( ! is_install )); then
    "${CONDA_PREFIX}/bin/python" -m pip "$@"
    return
  fi
  for (( attempt = 1; attempt <= PIP_INSTALL_RETRIES; attempt++ )); do
    if "${CONDA_PREFIX}/bin/python" -m pip "$@"; then
      return 0
    fi
    if (( attempt < PIP_INSTALL_RETRIES )); then
      echo "pip install attempt ${attempt}/${PIP_INSTALL_RETRIES} failed; retrying in ${PIP_RETRY_SLEEP_SEC}s" >&2
      sleep "${PIP_RETRY_SLEEP_SEC}"
    fi
  done
  return 1
}

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

if [[ "${SLIME_BUILD_RESUME_AFTER_CU129_BASE:-0}" == "1" ]]; then
  # A verified resume must not revisit public conda channels just to resolve
  # already-installed CUDA packages. The exact Python/native package audit
  # below remains the authoritative cu129 base gate.
  "${CONDA_PREFIX}/bin/nvcc" --version | grep -q 'release 12\.9'
  command -v rustc >/dev/null
  command -v cargo >/dev/null
else
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
fi

# install sglang. The Dockerfile starts FROM lmsysorg/sglang:v0.5.14-cu129
# which already has sglang installed with cu129-built native kernels; we have
# to install it ourselves here. Match the official cu129 image order: remove
# dependency variants whose distribution names end in -cu13 first, then
# reinstall the complete cu129 Torch stack so shared `site-packages/nvidia/*`
# files cannot be removed after their cu12 owners are installed.
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
# Online RL needs SGLang's autoregressive runtime. The `all` extra adds
# diffusion, tracing, and HTTP/2 stacks that are part of the general Docker
# image but not the Slime runtime and greatly expand resolver/network failure.
if [[ "${SLIME_BUILD_RESUME_AFTER_CU129_BASE:-0}" == "1" ]]; then
  SGLANG_DIR="${SGLANG_DIR}" python - <<'PY'
import json
import os
from importlib import metadata
from pathlib import Path

import torch
from packaging.version import Version

expected = {
    "sgl-deep-gemm": "0.1.3",
    "sglang": "0.5.14",
    "sglang-kernel": "0.4.4",
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "torchvision": "0.26.0",
}
for package, version in expected.items():
    actual = metadata.version(package)
    if Version(actual).base_version != version:
        raise SystemExit(f"cannot resume: {package} expected {version}, got {actual}")
cuda_python_version = Version(metadata.version("cuda-python"))
if cuda_python_version.release[:2] != (12, 9):
    raise SystemExit(
        f"cannot resume: cuda-python expected 12.9.x, got {cuda_python_version}"
    )
if torch.version.cuda != "12.9":
    raise SystemExit(f"cannot resume: torch CUDA expected 12.9, got {torch.version.cuda}")
direct_url = json.loads(metadata.distribution("sglang").read_text("direct_url.json"))
actual_source = Path(direct_url["url"].removeprefix("file://")).resolve()
expected_source = (Path(os.environ["SGLANG_DIR"]) / "python").resolve()
if not direct_url.get("dir_info", {}).get("editable") or actual_source != expected_source:
    raise SystemExit(
        f"cannot resume: SGLang editable source expected {expected_source}, got {actual_source}"
    )
print("cu129_base_resume_check=ok")
PY
else
  if [[ "${SKIP_SGLANG_DEPENDENCY_RESOLUTION:-0}" != "1" ]]; then
    pip install -e "python" --extra-index-url https://download.pytorch.org/whl/cu129
  else
    # Resume a previously dependency-resolved build without letting SGLang's
    # cu13-oriented metadata replace the pinned cu129 Torch stack again.
    pip install -e "python" --no-deps
  fi
  mapfile -t cuda13_packages < <(
    pip list --format=freeze \
      | awk -F'==' '/-cu13(==|$)/ {print $1}'
  )
  if (( ${#cuda13_packages[@]} )); then
    pip uninstall -y "${cuda13_packages[@]}"
  fi
  torch_index_args=(--index-url https://download.pytorch.org/whl/cu129)
  if [[ -n "${GENERAL_PYPI_INDEX_URL}" ]]; then
    torch_index_args+=(--extra-index-url "${GENERAL_PYPI_INDEX_URL}")
  fi
  pip install --force-reinstall \
    torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
    "${torch_index_args[@]}"
  pip install --force-reinstall --no-deps \
    sglang-kernel==0.4.4 sgl-deep-gemm==0.1.3 \
    --index-url https://docs.sglang.ai/whl/cu129/
  pip install --force-reinstall cuda-python==12.9
fi


pip install cmake ninja

# flash attn 2 (matches Dockerfile)
# SGLang's dependency resolver may install flash-attn-4.  Its distribution
# metadata makes TransformerEngine select the FA4 API even after FA2 has
# overwritten the shared flash_attn package, so remove it before building FA2.
pip uninstall -y flash-attn-4 || true

FLASH_ATTENTION_FORCE_BUILD=TRUE \
  FLASH_ATTN_CUDA_ARCHS="${FA2_CUDA_ARCHS}" \
  MAX_JOBS="${FA2_MAX_JOBS}" \
  pip -v install flash-attn==2.8.3 --no-build-isolation

# flash attn 3 (matches Dockerfile and TransformerEngine 2.16 CP APIs)
if [ ! -d "$BASE_DIR/flash-attention/.git" ]; then
  git clone https://github.com/Dao-AILab/flash-attention.git "$BASE_DIR/flash-attention"
fi
cd "$BASE_DIR/flash-attention"
git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c
git submodule update --init
cd hopper
if python - <<'PY'
from importlib import metadata

import flash_attn_3._C

assert metadata.version("flash-attn-3") == "3.0.0"
print("flash-attn-3=3.0.0 already installed")
PY
then
  :
else
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_DISABLE_SM80="$( [[ "${FA3_DISABLE_SM80}" == "1" ]] && echo TRUE || echo FALSE )" \
    MAX_JOBS="${FA3_MAX_JOBS}" \
    pip -v install . --no-build-isolation
fi

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
pip install nvidia-cudnn-cu12==9.16.0.29 --index-url "${NVIDIA_PYPI_INDEX_URL}"
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
import importlib
from importlib import metadata
from pathlib import Path

import torch
from packaging.utils import canonicalize_name
from packaging.version import Version

expected_exact = {
    "flash-attn": "2.8.3",
    "flash-attn-3": "3.0.0",
    "nvidia-cutlass-dsl": "4.5.2",
    "nvidia-cutlass-dsl-libs-base": "4.5.2",
    "sgl-deep-gemm": "0.1.3",
    "sglang": "0.5.14",
    "sglang-kernel": "0.4.4",
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.8.1",
    "transformer-engine": "2.16.1",
}
for package, expected in expected_exact.items():
    actual = metadata.version(package)
    if Version(actual).base_version != expected:
        raise SystemExit(f"{package}: expected {expected}, got {actual}")
    print(f"{package}={actual}")

try:
    flash_attn_4_version = metadata.version("flash-attn-4")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit(
        "flash-attn-4 must be absent when using FA2 2.8.3; "
        f"found {flash_attn_4_version}"
    )

if torch.version.cuda != "12.9":
    raise SystemExit(f"torch: expected CUDA 12.9, got {torch.version.cuda}")

cuda_python_version = Version(metadata.version("cuda-python"))
if cuda_python_version.release[:2] != (12, 9):
    raise SystemExit(
        f"cuda-python: expected a 12.9.x release, got {cuda_python_version}"
    )
print(f"cuda-python={cuda_python_version}")

cuda13_packages = sorted(
    name
    for distribution in metadata.distributions()
    if (name := distribution.metadata.get("Name"))
    and canonicalize_name(name).endswith("-cu13")
)
if cuda13_packages:
    raise SystemExit(f"unexpected -cu13 distributions: {cuda13_packages}")

ray_version = metadata.version("ray")
if Version(ray_version) < Version("2.55.1"):
    raise SystemExit(f"ray: expected >=2.55.1, got {ray_version}")
print(f"ray={ray_version}")

required_modules = (
    "apex",
    "cuda.bindings.driver",
    "deep_gemm",
    "flash_attn",
    "flash_attn_3._C",
    "hopper.flash_attn_interface",
    "megatron.core",
    "sgl_kernel",
    "sglang",
    "slime",
    "torch_memory_saver",
    "transformer_engine.pytorch",
)
for module_name in required_modules:
    module = importlib.import_module(module_name)
    print(f"imported {module_name} from {getattr(module, '__file__', '<builtin>')}")

site_packages = Path(metadata.distribution("torch-memory-saver").locate_file(""))
tms_preload = site_packages / "torch_memory_saver_hook_mode_preload_cu12.abi3.so"
if not tms_preload.is_file() or tms_preload.stat().st_size == 0:
    raise SystemExit(f"missing native torch-memory-saver preload hook: {tms_preload}")
print(f"torch-memory-saver-preload={tms_preload}")
PY
