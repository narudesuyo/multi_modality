#!/bin/bash
# Transfer data from local machine to ABCI.
#
# Usage:
#   bash scripts/abci/rsync_to_abci.sh              # actual transfer
#   bash scripts/abci/rsync_to_abci.sh --dry-run     # preview only
#
# Environment variables (override defaults):
#   ABCI_HOST       SSH host alias for ABCI (default: abci)
#   ABCI_DATA_ROOT  Remote data root (default: /groups/gch51606/takehiko.ohkawa/tmp_data)
#   LOCAL_DATA_ROOT Local data root (default: /work/narus/data)
#
# Transfer contents (~156 GB total):
#   - ee4d/                        ~96 GB   (EgoExo4D motion data)
#   - description/                 ~75 MB   (atomic descriptions)
#   - 1B_ft_k710_f8.pth           ~2 GB    (Stage1 pretrained weights)
#   - BodyTokenize/                ~146 MB  (VQ-VAE ckpt + config + stats)
#   - Assembly101/ (raw)           ~58 GB   (video + motion + text)

set -euo pipefail

# ---- Configuration ----
ABCI_HOST="${ABCI_HOST:-abci}"
ABCI_DATA_ROOT="${ABCI_DATA_ROOT:-/groups/gch51606/takehiko.ohkawa/tmp_data}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/work/narus/data}"

# Local paths
LOCAL_EGOHAND="/home/narus/2026/EgoHand"
LOCAL_STAGE1="${LOCAL_EGOHAND}/InternVideo/InternVideo2/multi_modality/scripts/pretraining/stage1"
LOCAL_BODYTOKENIZE="${LOCAL_EGOHAND}/BodyTokenize"
LOCAL_ASSEMBLY101="/home/share/datasets/Assembly101"

# Common rsync flags
RSYNC_OPTS="-avhP --stats"

# Pass --dry-run through
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN MODE ==="
fi

echo "============================================"
echo " rsync local → ABCI"
echo "============================================"
echo "ABCI_HOST:      ${ABCI_HOST}"
echo "ABCI_DATA_ROOT: ${ABCI_DATA_ROOT}"
echo "LOCAL_DATA_ROOT: ${LOCAL_DATA_ROOT}"
echo ""

# ---- 1. ee4d/ (~96 GB) ----
echo "=== [1/5] ee4d/ ==="
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_DATA_ROOT}/ee4d/" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/ee4d/"

# ---- 2. description/ (~75 MB) ----
echo "=== [2/5] description/ ==="
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_DATA_ROOT}/description/" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/description/"

# ---- 3. Stage1 weights (~2 GB) ----
echo "=== [3/5] 1B_ft_k710_f8.pth ==="
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_STAGE1}/1B_ft_k710_f8.pth" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/1B_ft_k710_f8.pth"

# ---- 4. BodyTokenize ckpt + config + statistics (~146 MB) ----
echo "=== [4/5] BodyTokenize/ ==="
# Create flat structure on ABCI: BodyTokenize/{ckpt_best.pt, config.yaml, statistics/}
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_BODYTOKENIZE}/ckpt_vq/ckpt_best.pt" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/BodyTokenize/ckpt_best.pt"
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_BODYTOKENIZE}/ckpt_vq/config.yaml" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/BodyTokenize/config.yaml"
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_BODYTOKENIZE}/ckpt_vq/statistics/" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/BodyTokenize/statistics/"

# ---- 5. Assembly101 raw data (~58 GB) ----
echo "=== [5/5] Assembly101/ ==="
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_ASSEMBLY101}/video/" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/Assembly101/video/"
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_ASSEMBLY101}/motion/" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/Assembly101/motion/"
rsync ${RSYNC_OPTS} ${DRY_RUN} \
    "${LOCAL_ASSEMBLY101}/text/" \
    "${ABCI_HOST}:${ABCI_DATA_ROOT}/Assembly101/text/"

echo ""
echo "============================================"
echo " Transfer complete!"
echo "============================================"
