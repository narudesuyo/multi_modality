#!/bin/bash
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -g gf15
#PJM -j
#PJM -o flashattn_build.%j.log
#----------------------------#

set -euo pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "=== Job start: $(date) ==="
echo "=== Host: $(hostname) ==="
echo "=== PWD:  $(pwd) ==="

# ---- conda activate ----
. /work/04/gf15/f15016/apps/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

SCRIPT_DIR="/work/04/gf15/f15016/script/EgoHand/InternVideo/InternVideo2/multi_modality"
ENV_PREFIX="${SCRIPT_DIR}/.conda"

conda activate "${ENV_PREFIX}"

echo "=== python ==="
python -V
which python

# ---- pip cache (avoid /pjmhome permission issue) ----
export PIP_CACHE_DIR="${HOME}/.cache/pip"
mkdir -p "${PIP_CACHE_DIR}"

# ---- GPU check ----
echo "=== GPU check ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found (GPU nodeじゃない可能性)"
fi

# ---- toolchain ----
echo "=== Load GCC (need >=9) ==="
module avail gcc >/dev/null 2>&1 || true
module load gcc-toolset/10 || module load gcc/12.2.0

echo "gcc: $(command -v gcc)"
gcc --version | head -n 1
echo "g++: $(command -v g++)"
g++ --version | head -n 1

# ---- CUDA ----
# 12.1 に合わせたい場合は pjsub 時に CUDA_MOD=cuda/12.1 を渡すか、下を固定で書き換え
CUDA_MOD="${CUDA_MOD:-cuda/12.2}"
echo "=== Load CUDA: ${CUDA_MOD} ==="
module avail cuda || true
module load "${CUDA_MOD}"

NVCC_PATH="$(command -v nvcc || true)"
if [ -z "${NVCC_PATH}" ]; then
  echo "ERROR: nvcc not found after loading ${CUDA_MOD}"
  exit 1
fi
export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
export CUDA_HOME
echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version | tail -n 1

# ---- torch check ----
echo "=== torch check ==="
python - <<'PY' || true
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

# ---- build settings ----
export NINJA_STATUS="[%f/%t %p] "
export MAX_JOBS="${MAX_JOBS:-24}"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-v2.6.3}"
FLASH_ATTN_DIR="${FLASH_ATTN_DIR:-/tmp/flash-attention}"

echo "FLASH_ATTN_DIR=${FLASH_ATTN_DIR}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION}"

# ---- clone/update ----
if [ -d "${FLASH_ATTN_DIR}/.git" ]; then
  cd "${FLASH_ATTN_DIR}"
  git fetch --all
  git checkout "${FLASH_ATTN_VERSION}"
else
  rm -rf "${FLASH_ATTN_DIR}"
  git clone --depth 1 --branch "${FLASH_ATTN_VERSION}" \
    https://github.com/Dao-AILab/flash-attention.git "${FLASH_ATTN_DIR}"
  cd "${FLASH_ATTN_DIR}"
fi

# ---- build/install ----
pip install -U pip setuptools wheel ninja

echo "=== Install flash-attn ==="
pip install . --no-build-isolation -v

echo "=== Install fused_dense_lib ==="
cd "${FLASH_ATTN_DIR}/csrc/fused_dense_lib"
pip install . --no-build-isolation -v

echo "=== Install layer_norm ==="
cd "${FLASH_ATTN_DIR}/csrc/layer_norm"
pip install . --no-build-isolation -v

echo "=== Done: $(date) ==="