#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda"

# ========= 0. Activate env =========
# (パスはあなたの環境に合わせて固定)
source /work/04/gf15/f15016/apps/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX}"

echo "=== [gpu] Host: $(hostname) ==="
echo "=== [gpu] Python: $(python -V) ==="

# ========= 1. Check GPU =========
echo "=== [gpu] Checking GPU ==="
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found. Are you on a GPU node?"; exit 1; }
nvidia-smi || true

# ========= 2. Load toolchain (GCC >= 9 required by PyTorch extensions) =========
echo "=== [gpu] Loading GCC toolchain (need GCC>=9) ==="
module avail gcc >/dev/null 2>&1 || true

# Prefer gcc-toolset/10 for compatibility; fallback to gcc/12.2.0
module load gcc-toolset/10 || module load gcc/12.2.0

echo "=== [gpu] gcc: $(command -v gcc) ==="
gcc --version | head -n 1
echo "=== [gpu] g++: $(command -v g++) ==="
g++ --version | head -n 1

# ========= 3. Load CUDA (Wisteria has cuda/12.2 default; cuda/12.1 also exists) =========
# Set CUDA module explicitly (do NOT try to match torch.version.cuda automatically)
CUDA_MOD="${CUDA_MOD:-cuda/12.2}"
echo "=== [gpu] Loading CUDA module: ${CUDA_MOD} ==="
module avail cuda || true
module load "${CUDA_MOD}" || {
  echo "ERROR: Failed to load ${CUDA_MOD}. Check 'module avail cuda'."
  exit 1
}

# ========= 4. CUDA_HOME / nvcc =========
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
nvcc --version | tail -n 1

# ========= 5. Check torch (optional; don't fail hard) =========
echo "=== [gpu] torch check ==="
python - <<'PY' || true
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

# ========= 6. Build flash-attn =========
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-v2.6.3}"

# /tmp is OK for interactive builds, but may be cleared; you can override:
#   FLASH_ATTN_DIR=/work/gf15/f15016/tmp/flash-attention bash setup_gpu.sh
FLASH_ATTN_DIR="${FLASH_ATTN_DIR:-/tmp/flash-attention}"

export MAX_JOBS="${MAX_JOBS:-24}"

echo "=== [gpu] Preparing flash-attention ${FLASH_ATTN_VERSION} in ${FLASH_ATTN_DIR} ==="
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

echo "=== [gpu] Installing build deps ==="
pip install -U pip setuptools wheel ninja

echo "=== [gpu] Installing flash-attn (MAX_JOBS=${MAX_JOBS}) ==="
pip install . --no-build-isolation -v

echo "=== [gpu] Installing CUDA extensions ==="

echo "--- [gpu] fused_dense_lib ---"
cd "${FLASH_ATTN_DIR}/csrc/fused_dense_lib"
pip install . -v

echo "--- [gpu] dropout_layer_norm ---"
cd "${FLASH_ATTN_DIR}/csrc/layer_norm"
pip install . -v

echo "=== [gpu] Done ==="