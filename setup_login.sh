#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda"

echo "=== [login] Creating conda env at: ${ENV_PREFIX} ==="
conda env create -f "${SCRIPT_DIR}/environment.yml" --prefix "${ENV_PREFIX}" --yes

# conda activate in script
eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX}"

echo "=== [login] Python / pip ==="
python -V
pip -V

echo "=== [login] Install non-CUDA deps (safe on login node) ==="
# 例：CUDAコンパイル不要なものだけここで
# ※ deepspeed/apex/flash-attn などはGPUノード側でやる
pip install -U pip setuptools wheel

# ここはあなたの環境に合わせて調整
# 例：requirements_noncuda.txt を用意しておくのが運用ラク
if [ -f "${SCRIPT_DIR}/requirements_noncuda.txt" ]; then
  pip install -r "${SCRIPT_DIR}/requirements_noncuda.txt"
fi

echo "=== [login] Done. Next: run setup_gpu.sh on a GPU interactive node ==="
echo "conda activate ${ENV_PREFIX}"