#!/usr/bin/env python3
"""Build final annotation JSON for Assembly101 from intermediate + tok_pose.

Reads the intermediate annotation from prepare_assembly101_clips.py,
matches with tokenized pose files from BodyTokenize inference_atomic.py,
and generates the final annotation JSON for Stage2 training.

Usage:
    python build_annotation_assembly101.py
    python build_annotation_assembly101.py --split train
"""

import argparse
import json
import os
from pathlib import Path


_DATA_ROOT_BASE = os.environ.get("DATA_ROOT", "/work/narus/data")


def build_annotation(intermediate_json: str, output_path: str, data_root: str):
    with open(intermediate_json, "r") as f:
        entries = json.load(f)

    annotations = []
    stats = {
        "total_entries": len(entries),
        "matched": 0,
        "no_tok_pose": 0,
        "no_video": 0,
    }

    tok_pose_base = os.path.join(data_root, "tok_pose_atomic_40")

    for entry in entries:
        session_id = entry["session_id"]
        sample_id = entry["sample_id"]
        caption = entry["caption"]

        # Check video frames exist
        if "video" not in entry:
            stats["no_video"] += 1
            continue
        video_abs = os.path.join(data_root, entry["video"])
        if not os.path.isdir(video_abs):
            stats["no_video"] += 1
            continue

        # Find tokenized pose files for this sample
        tok_pose_dir = os.path.join(tok_pose_base, session_id)
        tok_pose_files = []
        if os.path.isdir(tok_pose_dir):
            for fname in sorted(os.listdir(tok_pose_dir)):
                if fname.startswith(f"{sample_id}_") and fname.endswith(".npz"):
                    tok_pose_files.append(fname)

        if not tok_pose_files:
            stats["no_tok_pose"] += 1
            continue

        # One annotation per tok_pose chunk
        for tok_fname in tok_pose_files:
            tok_rel = os.path.join("tok_pose_atomic_40", session_id, tok_fname)
            annotations.append({
                "video": entry["video"],
                "tok_pose": tok_rel,
                "caption": caption,
            })
            stats["matched"] += 1

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Total intermediate entries: {stats['total_entries']}")
    print(f"Matched (video+tok_pose):  {stats['matched']}")
    print(f"No tok_pose:               {stats['no_tok_pose']}")
    print(f"No video frames:           {stats['no_video']}")
    print(f"Total annotations:         {len(annotations)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--intermediate-json", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))

    if args.intermediate_json is None:
        args.intermediate_json = os.path.join(
            _here, f"annotation_assembly101_{args.split}_intermediate.json",
        )

    if args.output is None:
        args.output = os.path.join(
            _here, f"annotation_assembly101_{args.split}.json",
        )

    data_root = os.path.join(_DATA_ROOT_BASE, "Assembly101/processed", args.split)

    build_annotation(args.intermediate_json, args.output, data_root)
