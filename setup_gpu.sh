#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda"

# ========= 0. Activate env =========
source /work/04/gf15/f15016/apps/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX}"

echo "=== [gpu] Checking GPU ==="
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found. Are you on a GPU node?"; exit 1; }
nvidia-smi || true

# ========= 1. Load CUDA module (match torch CUDA) =========
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")')"
TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
echo "=== [gpu] torch = ${TORCH_VER}, torch.version.cuda = ${TORCH_CUDA} ==="

if [ -z "${TORCH_CUDA}" ]; then
  echo "ERROR: torch.version.cuda is empty (CPU torch?). Install CUDA-enabled torch first."
  exit 1
fi

# Wisteriaは module がある想定
# まず候補表示（ログ用）
module avail cuda || true

# できるだけ torch の CUDA と同じバージョンをロード
# 例: 12.8 -> cuda/12.8 があればそれを使う
CUDA_MOD="cuda/${TORCH_CUDA}"
echo "=== [gpu] Loading module: ${CUDA_MOD} ==="
module load "${CUDA_MOD}" || {
  echo "WARNING: ${CUDA_MOD} not found. Load an available cuda/<ver> close to ${TORCH_CUDA} manually."
  echo "Try: module avail cuda"
  exit 1
}

# ========= 2. CUDA_HOME / nvcc =========
NVCC_PATH="$(command -v nvcc || true)"
if [ -z "${NVCC_PATH}" ]; then
  echo "ERROR: nvcc not found after loading CUDA module."
  exit 1
fi

export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
export CUDA_HOME
echo "=== [gpu] nvcc = ${NVCC_PATH} ==="
echo "=== [gpu] CUDA_HOME = ${CUDA_HOME} ==="
nvcc --version

# ========= 3. Build flash-attn =========
FLASH_ATTN_VERSION="v2.6.3"
FLASH_ATTN_DIR="/tmp/flash-attention"
export MAX_JOBS="${MAX_JOBS:-24}"

echo "=== [gpu] Cloning flash-attention ${FLASH_ATTN_VERSION} to ${FLASH_ATTN_DIR} ==="
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

echo "=== [gpu] Installing flash-attn (MAX_JOBS=${MAX_JOBS}) ==="
pip install -U pip setuptools wheel
pip install . --no-build-isolation

echo "=== [gpu] Installing CUDA extensions ==="

echo "--- [gpu] fused_dense_lib ---"
cd "${FLASH_ATTN_DIR}/csrc/fused_dense_lib"
pip install .

echo "--- [gpu] dropout_layer_norm ---"
cd "${FLASH_ATTN_DIR}/csrc/layer_norm"
pip install .

echo "=== [gpu] Done ==="