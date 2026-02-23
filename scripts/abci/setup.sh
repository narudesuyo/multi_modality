#!/bin/bash
# ABCI environment setup for InternVideo2 multi-modality training.
#
# Usage:
#   1. Get an interactive GPU node:
#        qrsh -g <GROUP> -l rt_HG=1 -l h_rt=1:00:00
#
#   2. Run this script:
#        bash scripts/abci/setup.sh
#
#   Options (environment variables):
#     CUDA_MOD=cuda/12.2       CUDA module to load (default: cuda/12.2)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_PREFIX="${PROJECT_ROOT}/.conda"

echo "============================================"
echo " ABCI Setup: InternVideo2 multi-modality"
echo "============================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env:    ${ENV_PREFIX}"
echo ""

# ============================================================
# 1. Module setup (GCC + CUDA)
# ============================================================
echo "=== [1/3] Loading modules ==="

if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' command not found. Are you on ABCI?"
    exit 1
fi

module load gcc/12.2.0 2>/dev/null || module load gcc/11.4.0 2>/dev/null || {
    echo "WARN: Could not load GCC module, using system default"
}

CUDA_MOD="${CUDA_MOD:-cuda/12.2}"
module load "${CUDA_MOD}" || {
    echo "ERROR: Failed to load ${CUDA_MOD}."
    echo "Available CUDA modules:"
    module avail cuda 2>&1 || true
    exit 1
}

NVCC_PATH="$(command -v nvcc || true)"
if [ -z "${NVCC_PATH}" ]; then
    echo "ERROR: nvcc not found after loading ${CUDA_MOD}"
    exit 1
fi
CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
export CUDA_HOME
echo "GCC:       $(gcc --version | head -n1)"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "nvcc:      $(nvcc --version | tail -n1)"

# GPU check
if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    echo ""
else
    echo "WARN: nvidia-smi not found — are you on a GPU node?"
fi

# ============================================================
# 2. Conda environment
# ============================================================
echo "=== [2/3] Creating conda environment ==="

# ABCI provides conda via module or user install
if ! command -v conda >/dev/null 2>&1; then
    # Try common locations
    for p in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/apps/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "$p" ]; then
            . "$p"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found."
    echo "Install miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
    exit 1
fi

eval "$(conda shell.bash hook)" 2>/dev/null || true

if [ -d "${ENV_PREFIX}" ]; then
    echo "Conda env already exists at ${ENV_PREFIX}, activating..."
else
    echo "Creating new conda env from environment.yml..."
    conda env create -f "${PROJECT_ROOT}/environment.yml" --prefix "${ENV_PREFIX}" --override-channels --yes
fi

conda activate "${ENV_PREFIX}"
echo "Python: $(python -V) ($(which python))"

# ============================================================
# 3. Python dependencies (non-CUDA)
# ============================================================
echo "=== [3/3] Installing Python dependencies ==="

pip install -U pip setuptools wheel

cd "${PROJECT_ROOT}"
# Install the project itself + deps from pyproject.toml
# (flash_attn, apex may fail — that's OK)
pip install -e . --no-build-isolation 2>&1 | tail -5 || {
    echo "WARN: Some deps may have failed (flash_attn/apex expected). Continuing..."
}

# Install tyro for ABCI job submission script
pip install tyro

echo "torch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "CUDA:   $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "GPU ok: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"

# ============================================================
# Verification
# ============================================================
echo ""
echo "=== Verifying installation ==="

cd "${PROJECT_ROOT}"
python - <<'PY'
errors = []

def check(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, '__version__', '?')
        print(f"  OK  {name} == {ver}")
    except ImportError as e:
        print(f"  NG  {name}: {e}")
        errors.append(name)

print("--- Core ---")
check("torch")
check("torchvision")
check("torchaudio")
check("deepspeed")
check("transformers")
check("timm")
check("wandb")

print("--- GPU ---")
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Devices: {torch.cuda.device_count()}")

if errors:
    print(f"\nWARN: {len(errors)} package(s) failed: {', '.join(errors)}")
else:
    print("\nAll packages OK!")
PY

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " To activate later:"
echo "   conda activate ${ENV_PREFIX}"
echo ""
echo " To submit a training job:"
echo "   python scripts/abci/job.py --dry-run"
echo "============================================"
