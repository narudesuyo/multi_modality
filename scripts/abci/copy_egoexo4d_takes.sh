#!/bin/bash
# Copy required EgoExo4D takes from ABCI shared dataset to DATA_ROOT.
#
# This script:
#   1. Extracts the list of take_names needed from ee4d motion data (takes.json)
#   2. Generates take_list_{train,val}.json
#   3. Copies only relevant takes from the ABCI shared EgoExo4D dataset
#
# Usage (run on ABCI):
#   bash scripts/abci/copy_egoexo4d_takes.sh
#   bash scripts/abci/copy_egoexo4d_takes.sh --dry-run
#
# Environment variables:
#   DATA_ROOT       Target data root (default: /groups/gch51606/takehiko.ohkawa/tmp_data)
#   SRC_TAKES_DIR   Source takes directory (default: /groups/gch51606/dataset/egoexo4d/takes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/groups/gch51606/takehiko.ohkawa/tmp_data}"
SRC_TAKES_DIR="${SRC_TAKES_DIR:-/groups/gch51606/dataset/egoexo4d/takes}"
DST_TAKES_DIR="${DATA_ROOT}/EgoExo4D/takes"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="true"
    echo "=== DRY RUN MODE ==="
fi

echo "============================================"
echo " Copy EgoExo4D takes for Stage2"
echo "============================================"
echo "DATA_ROOT:     ${DATA_ROOT}"
echo "SRC_TAKES_DIR: ${SRC_TAKES_DIR}"
echo "DST_TAKES_DIR: ${DST_TAKES_DIR}"
echo ""

# ---- Step 1: Generate take lists from ee4d motion data ----
echo "=== [1/2] Generating take lists ==="

python3 - "${DATA_ROOT}" "${DST_TAKES_DIR}" "${DRY_RUN}" <<'PYEOF'
import json
import os
import sys
from pathlib import Path

data_root = sys.argv[1]
dst_takes_dir = sys.argv[2]
dry_run = sys.argv[3] == "true"

# Load takes.json from ee4d to get uuid -> take_name mapping
takes_json = os.path.join(data_root, "ee4d", "ee4d_motion_uniegomotion", "takes.json")
print(f"Loading {takes_json} ...")
with open(takes_json, "r") as f:
    takes = json.load(f)
uuid_to_take = {t["take_uid"]: t["take_name"] for t in takes}
print(f"  Total takes in takes.json: {len(uuid_to_take)}")

# Load atomic descriptions to find which takes are in each split
egoexo4d_dir = os.path.join(data_root, "EgoExo4D")
os.makedirs(egoexo4d_dir, exist_ok=True)

for split in ["train", "val"]:
    desc_path = os.path.join(data_root, "description", f"atomic_descriptions_{split}.json")
    if not os.path.isfile(desc_path):
        print(f"  WARN: {desc_path} not found, skipping {split}")
        continue

    with open(desc_path, "r") as f:
        desc_data = json.load(f)

    # Collect unique take_names from annotations
    take_names = set()
    annotations = desc_data.get("annotations", desc_data) if isinstance(desc_data, dict) else desc_data
    if isinstance(annotations, dict):
        annotations = annotations.get("annotations", [])

    for ann in annotations:
        take_uid = ann.get("take_uid", "")
        take_name = uuid_to_take.get(take_uid)
        if take_name:
            take_names.add(take_name)

    take_names = sorted(take_names)
    print(f"  {split}: {len(take_names)} unique takes")

    # Write take_list JSON
    take_list_path = os.path.join(egoexo4d_dir, f"take_list_{split}.json")
    with open(take_list_path, "w") as f:
        json.dump(take_names, f, indent=2)
    print(f"  Wrote {take_list_path}")

# Also count how many takes exist in the source dir
src_dir = dst_takes_dir.replace(data_root, "").strip("/")
print(f"\n  Take lists ready. Next step will copy takes to {dst_takes_dir}")
PYEOF

# ---- Step 2: Copy takes ----
echo ""
echo "=== [2/2] Copying takes ==="

for split in train val; do
    TAKE_LIST="${DATA_ROOT}/EgoExo4D/take_list_${split}.json"
    if [[ ! -f "${TAKE_LIST}" ]]; then
        echo "WARN: ${TAKE_LIST} not found, skipping ${split}"
        continue
    fi

    # Read take names from JSON list
    TAKE_NAMES=$(python3 -c "import json; print('\n'.join(json.load(open('${TAKE_LIST}'))))")
    TOTAL=$(echo "${TAKE_NAMES}" | wc -l)
    COUNT=0

    echo "${split}: ${TOTAL} takes to copy"

    mkdir -p "${DST_TAKES_DIR}"

    while IFS= read -r take_name; do
        COUNT=$((COUNT + 1))
        SRC="${SRC_TAKES_DIR}/${take_name}"
        DST="${DST_TAKES_DIR}/${take_name}"

        if [[ -d "${DST}" ]]; then
            # Already copied
            continue
        fi

        if [[ ! -d "${SRC}" ]]; then
            echo "  [${COUNT}/${TOTAL}] MISSING: ${take_name}"
            continue
        fi

        if [[ -n "${DRY_RUN}" ]]; then
            echo "  [${COUNT}/${TOTAL}] Would copy: ${take_name}"
        else
            echo "  [${COUNT}/${TOTAL}] Copying: ${take_name}"
            # Copy only frame_aligned_videos (the only subdir needed for exo frames)
            mkdir -p "${DST}/frame_aligned_videos"
            cp -a "${SRC}/frame_aligned_videos/." "${DST}/frame_aligned_videos/" 2>/dev/null || {
                echo "    WARN: frame_aligned_videos not found in ${take_name}, copying entire take"
                cp -a "${SRC}" "${DST}"
            }
        fi
    done <<< "${TAKE_NAMES}"
done

echo ""
echo "============================================"
echo " EgoExo4D takes copy complete!"
echo "============================================"
