#!/bin/bash
set -euo pipefail

# Flash Attention 2 and its CUDA extensions installer
# Ref: https://github.com/Dao-AILab/flash-attention

FLASH_ATTN_VERSION="v2.6.3"
FLASH_ATTN_DIR="/tmp/flash-attention"

# Use MAX_JOBS to control parallel compilation (default: nproc)
export MAX_JOBS=24

echo "=== Cloning flash-attention ${FLASH_ATTN_VERSION} to ${FLASH_ATTN_DIR} ==="
if [ -d "${FLASH_ATTN_DIR}" ]; then
    echo "Directory already exists, pulling latest..."
    cd "${FLASH_ATTN_DIR}"
    git fetch && git checkout "${FLASH_ATTN_VERSION}"
else
    git clone --depth 1 --branch "${FLASH_ATTN_VERSION}" \
        https://github.com/Dao-AILab/flash-attention.git "${FLASH_ATTN_DIR}"
    cd "${FLASH_ATTN_DIR}"
fi

echo "=== Installing flash-attn (MAX_JOBS=${MAX_JOBS}) ==="
pip install . --no-build-isolation

echo "=== Installing CUDA extensions ==="

# fused_dense_lib (FusedMLP)
echo "--- Installing fused_dense_lib ---"
cd "${FLASH_ATTN_DIR}/csrc/fused_dense_lib"
pip install .

# dropout_layer_norm (DropoutAddRMSNorm)
echo "--- Installing dropout_layer_norm ---"
cd "${FLASH_ATTN_DIR}/csrc/layer_norm"
pip install .

echo "=== Done ==="
