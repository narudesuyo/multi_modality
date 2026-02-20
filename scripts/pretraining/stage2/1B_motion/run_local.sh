#!/bin/bash
# Stage2 training: Video + Motion + Text
# Works with 1 GPU or multiple GPUs (auto-detected).
#
# Usage:
#   bash run_local.sh                    # use all available GPUs
#   bash run_local.sh --nproc_per_node 1 # force 1 GPU
#   CUDA_VISIBLE_DEVICES=0,1 bash run_local.sh  # select GPUs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$WORK_DIR"

CONDA_ENV="${WORK_DIR}/.conda"
export PATH="${CONDA_ENV}/bin:${PATH}"

export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}./"
# wandb: default online. Set WANDB_MODE=offline for local-only logging.
export WANDB_MODE="${WANDB_MODE:-online}"

# Suppress noisy debug/warning logs
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY="error"
export DATASETS_VERBOSITY="error"
export NCCL_DEBUG="WARN"
export DS_LOG_LEVEL="WARNING"
export TORCH_DISTRIBUTED_DEBUG="OFF"

# Auto-detect GPU count
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected GPUs: ${NUM_GPUS}"

JOB_NAME="stage2_motion_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${JOB_NAME}"

echo "Working dir: ${WORK_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Config:      ${SCRIPT_DIR}/config.py"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "$@" \
    tasks/pretrain.py \
    "${SCRIPT_DIR}/config.py" \
    output_dir "${OUTPUT_DIR}" \
    num_workers 0
