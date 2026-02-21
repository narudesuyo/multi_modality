#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gf15
#PJM -j
#PJM -o flashattn_build.%j.log
#----------------------------#

set -eu

echo "=== Job start: $(date) ==="
echo "=== Host: $(hostname) ==="
echo "=== PWD:  $(pwd) ==="

# 0) (必須) module 初期化されてない環境向け
# 環境によって不要/パスが違う場合あり。通らなければ消してOK。
if [ -f /etc/profile ]; then . /etc/profile; fi

# 1) conda activate
# あなたの環境の conda.sh に合わせて固定
. /work/04/gf15/f15016/apps/miniconda3/etc/profile.d/conda.sh
# conda function 有効化
eval "$(conda shell.sh hook)" 2>/dev/null || true

SCRIPT_DIR="/work/04/gf15/f15016/script/EgoHand/InternVideo/InternVideo2/multi_modality"
ENV_PREFIX="${SCRIPT_DIR}/.conda"

conda activate "${ENV_PREFIX}"

python -V || true
which python || true

# 2) GPU確認
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found (GPU nodeじゃない可能性)"
fi

# 3) toolchain + CUDA
# GCC >= 9
module avail gcc >/dev/null 2>&1 || true
module load gcc-toolset/10 || module load gcc/12.2.0

gcc --version | head -n 1 || true
g++ --version | head -n 1 || true

# CUDA module (存在するやつに固定)
CUDA_MOD="${CUDA_MOD:-cuda/12.2}"
module avail cuda || true
module load "${CUDA_MOD}"

NVCC_PATH="$(command -v nvcc || true)"
if [ -z "${NVCC_PATH}" ]; then
  echo "ERROR: nvcc not found after loading ${CUDA_MOD}"
  exit 1
fi
CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
export CUDA_HOME
echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version | tail -n 1 || true

# 4) torch check（失敗しても続行）
python - <<'PY' || true
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

# 5) flash-attn build
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-v2.6.3}"
# /tmp はジョブ中のみ。残したいなら work配下に
FLASH_ATTN_DIR="${FLASH_ATTN_DIR:-/tmp/flash-attention}"
MAX_JOBS="${MAX_JOBS:-24}"
export MAX_JOBS

echo "FLASH_ATTN_DIR=${FLASH_ATTN_DIR}"
echo "MAX_JOBS=${MAX_JOBS}"

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

pip install -U pip setuptools wheel ninja
pip install . --no-build-isolation -v

echo "--- fused_dense_lib ---"
cd "${FLASH_ATTN_DIR}/csrc/fused_dense_lib"
pip install . -v

echo "--- layer_norm ---"
cd "${FLASH_ATTN_DIR}/csrc/layer_norm"
pip install . -v

echo "=== Done: $(date) ==="