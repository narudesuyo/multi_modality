#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -g gf15
#PJM -j
#PJM -o stage2_train.%j.log
#PJM --mpi proc=1
#----------------------------#
set -eu

echo "=== Job start: $(date) ==="
echo "=== Host: $(hostname) ==="
echo "=== PWD:  $(pwd) ==="

# ---- conda activate ----
. /work/04/gf15/f15016/apps/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.sh hook)" 2>/dev/null || true

SCRIPT_DIR="/work/04/gf15/f15016/script/EgoHand/InternVideo/InternVideo2/multi_modality"
ENV_PREFIX="${SCRIPT_DIR}/.conda"

conda activate "${ENV_PREFIX}"

python -V || true
which python || true

# ---- GPU check ----
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found"
fi

# ---- toolchain + CUDA ----
module load gcc-toolset/10 || module load gcc/12.2.0
CUDA_MOD="${CUDA_MOD:-cuda/12.2}"
module load "${CUDA_MOD}"

NVCC_PATH="$(command -v nvcc || true)"
if [ -z "${NVCC_PATH}" ]; then
  echo "ERROR: nvcc not found after loading ${CUDA_MOD}"
  exit 1
fi
CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
export CUDA_HOME
echo "CUDA_HOME=${CUDA_HOME}"

# ---- training config ----
WORK_DIR="${SCRIPT_DIR}"
CONFIG_DIR="${WORK_DIR}/scripts/pretraining/stage2/1B_motion"
cd "${WORK_DIR}"

# ---- multi-node setup (PJM) ----
NNODES="${PJM_NODE_COUNT:-1}"
MASTER_ADDR=$(head -n 1 "${PJM_O_NODEINF}")
export MASTER_ADDR
export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}./"
export WANDB_MODE="${WANDB_MODE:-online}"

# Suppress noisy logs
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY="error"
export DATASETS_VERBOSITY="error"
export NCCL_DEBUG="WARN"
export DS_LOG_LEVEL="WARNING"
export TORCH_DISTRIBUTED_DEBUG="OFF"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "Detected GPUs per node: ${NUM_GPUS}"
echo "NNODES: ${NNODES}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

JOB_NAME="stage2_motion_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${CONFIG_DIR}/outputs/${JOB_NAME}"

echo "Working dir: ${WORK_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Config:      ${CONFIG_DIR}/config.py"

# mpirun で各ノードに torchrun を配布
# rendezvous (c10d) で node_rank を自動割り当て
mpirun -np "${NNODES}" --npernode 1 --bind-to none \
    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NUM_GPUS}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        tasks/pretrain.py \
        "${CONFIG_DIR}/config.py" \
        output_dir "${OUTPUT_DIR}" \
        num_workers 0

echo "=== Job end: $(date) ==="
