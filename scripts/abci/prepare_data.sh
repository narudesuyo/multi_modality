#!/bin/bash
# Full data preparation pipeline for Stage2 training on ABCI.
#
# Runs all data processing steps:
#   1. EgoExo4D: exo frames + kp3d extraction (prepare_atomic_clips.py)
#   2. Assembly101: frame symlinks + SMPL-H → kp3d (prepare_assembly101_clips.py)
#   3. BodyTokenize: VQ-VAE tokenization (inference_atomic.py)
#   4. Final annotation JSON generation (build_annotation_*.py)
#
# Usage (run on ABCI GPU node):
#   qrsh -g gch51606 -l rt_HG=1 -l h_rt=6:00:00
#   bash scripts/abci/prepare_data.sh
#
# Steps can be run selectively:
#   bash scripts/abci/prepare_data.sh --step 1      # EgoExo4D only
#   bash scripts/abci/prepare_data.sh --step 2      # Assembly101 only
#   bash scripts/abci/prepare_data.sh --step 3      # BodyTokenize only
#   bash scripts/abci/prepare_data.sh --step 4      # Annotation only
#
# Environment variables:
#   DATA_ROOT         (default: /groups/gch51606/takehiko.ohkawa/tmp_data)
#   BODYTOKENIZE_DIR  (default: /home/ach18478ho/code/tmp_code/BodyTokenize)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MOTION_SCRIPTS="${PROJECT_ROOT}/scripts/pretraining/stage2/1B_motion"

export DATA_ROOT="${DATA_ROOT:-/groups/gch51606/takehiko.ohkawa/tmp_data}"
BODYTOKENIZE_DIR="${BODYTOKENIZE_DIR:-/home/ach18478ho/code/tmp_code/BodyTokenize}"

# Parse --step argument (default: run all)
STEP="${1:-all}"
if [[ "${STEP}" == "--step" ]]; then
    STEP="${2:-all}"
fi

echo "============================================"
echo " Stage2 Data Preparation (ABCI)"
echo "============================================"
echo "PROJECT_ROOT:     ${PROJECT_ROOT}"
echo "DATA_ROOT:        ${DATA_ROOT}"
echo "BODYTOKENIZE_DIR: ${BODYTOKENIZE_DIR}"
echo "Step:             ${STEP}"
echo ""

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_ROOT}"

run_step() {
    local step_num="$1"
    [[ "${STEP}" == "all" || "${STEP}" == "${step_num}" ]]
}

# ============================================================
# Step 1: EgoExo4D — exo JPG frames + kp3d extraction
# ============================================================
if run_step 1; then
    echo "=== [Step 1] EgoExo4D: exo frames + kp3d ==="

    for split in train val; do
        echo "--- ${split} ---"
        python "${MOTION_SCRIPTS}/prepare_atomic_clips.py" \
            --split "${split}" --skip-existing
    done

    echo "[Step 1] Done."
    echo ""
fi

# ============================================================
# Step 2: Assembly101 — frame symlinks + SMPL-H → kp3d
# ============================================================
if run_step 2; then
    echo "=== [Step 2] Assembly101: frames + kp3d ==="

    for split in train validation; do
        echo "--- ${split} ---"
        python "${MOTION_SCRIPTS}/prepare_assembly101_clips.py" \
            --split "${split}" --device cuda --skip-existing
    done

    echo "[Step 2] Done."
    echo ""
fi

# ============================================================
# Step 3: BodyTokenize — VQ-VAE tokenization (requires GPU)
# ============================================================
if run_step 3; then
    echo "=== [Step 3] BodyTokenize: VQ-VAE tokenization ==="

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "WARN: No GPU detected. BodyTokenize needs a GPU node."
        echo "  Use: qrsh -g gch51606 -l rt_HG=1 -l h_rt=2:00:00"
    fi

    cd "${BODYTOKENIZE_DIR}"

    # EgoExo4D train
    echo "--- EgoExo4D train ---"
    python inference_atomic.py \
        --data-root "${DATA_ROOT}/EgoExo4D/processed/train" \
        --motion-dir "${DATA_ROOT}/EgoExo4D/processed/train/motion_atomic" \
        --output-dir "${DATA_ROOT}/EgoExo4D/processed/train/tok_pose_atomic_40" \
        --annotation-json "${MOTION_SCRIPTS}/annotation_atomic_train_intermediate.json"

    # EgoExo4D val
    echo "--- EgoExo4D val ---"
    python inference_atomic.py \
        --data-root "${DATA_ROOT}/EgoExo4D/processed/val" \
        --motion-dir "${DATA_ROOT}/EgoExo4D/processed/val/motion_atomic" \
        --output-dir "${DATA_ROOT}/EgoExo4D/processed/val/tok_pose_atomic_40" \
        --annotation-json "${MOTION_SCRIPTS}/annotation_atomic_val_intermediate.json"

    # Assembly101 train
    echo "--- Assembly101 train ---"
    python inference_atomic.py \
        --data-root "${DATA_ROOT}/Assembly101/processed/train" \
        --motion-dir "${DATA_ROOT}/Assembly101/processed/train/motion_atomic" \
        --output-dir "${DATA_ROOT}/Assembly101/processed/train/tok_pose_atomic_40" \
        --annotation-json "${MOTION_SCRIPTS}/annotation_assembly101_train_intermediate.json"

    # Assembly101 validation
    echo "--- Assembly101 validation ---"
    python inference_atomic.py \
        --data-root "${DATA_ROOT}/Assembly101/processed/validation" \
        --motion-dir "${DATA_ROOT}/Assembly101/processed/validation/motion_atomic" \
        --output-dir "${DATA_ROOT}/Assembly101/processed/validation/tok_pose_atomic_40" \
        --annotation-json "${MOTION_SCRIPTS}/annotation_assembly101_validation_intermediate.json"

    cd "${PROJECT_ROOT}"

    echo "[Step 3] Done."
    echo ""
fi

# ============================================================
# Step 4: Final annotation JSON generation
# ============================================================
if run_step 4; then
    echo "=== [Step 4] Build final annotation JSONs ==="

    cd "${PROJECT_ROOT}"

    # EgoExo4D
    echo "--- EgoExo4D train ---"
    python "${MOTION_SCRIPTS}/build_annotation_atomic.py" --split train
    echo "--- EgoExo4D val ---"
    python "${MOTION_SCRIPTS}/build_annotation_atomic.py" --split val

    # Assembly101
    echo "--- Assembly101 train ---"
    python "${MOTION_SCRIPTS}/build_annotation_assembly101.py" --split train
    echo "--- Assembly101 validation ---"
    python "${MOTION_SCRIPTS}/build_annotation_assembly101.py" --split validation

    echo "[Step 4] Done."
    echo ""
fi

echo "============================================"
echo " Data preparation complete!"
echo ""
echo " Next: submit training job"
echo "   python scripts/abci/job.py --use-bf16 --dry-run"
echo "============================================"
