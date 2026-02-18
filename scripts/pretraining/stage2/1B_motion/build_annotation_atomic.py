"""Build final annotation JSON for Stage2 training from atomic clips.

Reads the intermediate annotation from prepare_atomic_clips.py and matches
with tokenized pose files from BodyTokenize inference_atomic.py.

Generates entries for both ego and exo views (as separate training samples).

Usage:
    python build_annotation_atomic.py
    python build_annotation_atomic.py --output annotation_atomic_final.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


DATA_ROOT = "/large/naru/EgoHand/data/train/takes_clipped/egoexo"
TOK_POSE_DIR = os.path.join(DATA_ROOT, "tok_pose_atomic_40")


def build_annotation(intermediate_json: str, output_path: str):
    with open(intermediate_json, "r") as f:
        entries = json.load(f)

    annotations = []
    stats = {
        "total_entries": len(entries),
        "ego_matched": 0,
        "exo_matched": 0,
        "no_tok_pose": 0,
        "no_video": 0,
    }

    for entry in entries:
        take_name = entry["take_name"]
        sample_id = entry["sample_id"]
        caption = entry["caption"]

        # Find tokenized pose chunks for this sample
        tok_pose_dir = os.path.join(TOK_POSE_DIR, take_name)
        tok_pose_files = []
        if os.path.isdir(tok_pose_dir):
            for fname in sorted(os.listdir(tok_pose_dir)):
                if fname.startswith(f"{sample_id}_") and fname.endswith(".npz"):
                    tok_pose_files.append(fname)

        if not tok_pose_files:
            stats["no_tok_pose"] += 1
            continue

        # Create entries for each tok_pose chunk x each view
        for tok_fname in tok_pose_files:
            tok_rel = os.path.join("tok_pose_atomic_40", take_name, tok_fname)

            # Ego view
            if "video_ego" in entry:
                video_path = os.path.join(DATA_ROOT, entry["video_ego"])
                if os.path.isfile(video_path):
                    annotations.append({
                        "video": entry["video_ego"],
                        "tok_pose": tok_rel,
                        "caption": caption,
                    })
                    stats["ego_matched"] += 1
                else:
                    stats["no_video"] += 1

            # Exo view
            if "video_exo" in entry:
                video_path = os.path.join(DATA_ROOT, entry["video_exo"])
                if os.path.isfile(video_path):
                    annotations.append({
                        "video": entry["video_exo"],
                        "tok_pose": tok_rel,
                        "caption": caption,
                    })
                    stats["exo_matched"] += 1
                else:
                    stats["no_video"] += 1

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Total intermediate entries: {stats['total_entries']}")
    print(f"Ego matched:    {stats['ego_matched']}")
    print(f"Exo matched:    {stats['exo_matched']}")
    print(f"No tok_pose:    {stats['no_tok_pose']}")
    print(f"No video file:  {stats['no_video']}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate-json", type=str,
                        default=os.path.join(os.path.dirname(__file__), "annotation_atomic_intermediate.json"),
                        help="Path to intermediate JSON from prepare_atomic_clips.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output annotation JSON path")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "annotation_atomic_final.json")

    build_annotation(args.intermediate_json, args.output)
