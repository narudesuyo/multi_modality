"""Build annotation JSON for Video + Motion (tok_pose) + Text training.

Scans video and tok_pose directories, matches clips with their pose chunks,
and generates annotation JSON with dummy captions.

Usage:
    python build_annotation.py --tok_pose_dir all --output annotation_all.json
    python build_annotation.py --tok_pose_dir 20  --output annotation_20.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


DATA_ROOT = "/large/naru/EgoHand/data/train/takes_clipped/egoexo"
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")

DUMMY_CAPTIONS = [
    "a person performing an activity",
    "a person doing something with their hands",
    "a person moving their body",
    "someone performing a task",
    "a human performing an action",
]


def build_annotation(tok_pose_dir_name, output_path):
    tok_pose_dir = os.path.join(DATA_ROOT, "tok_pose", tok_pose_dir_name)

    if not os.path.isdir(tok_pose_dir):
        raise FileNotFoundError(f"tok_pose dir not found: {tok_pose_dir}")

    annotations = []
    n_matched = 0
    n_no_video = 0

    # Scan all takes in tok_pose
    takes = sorted(os.listdir(tok_pose_dir))
    takes = [t for t in takes if os.path.isdir(os.path.join(tok_pose_dir, t))]

    for take_name in takes:
        take_pose_dir = os.path.join(tok_pose_dir, take_name)

        # Group pose chunks by their video clip (start___end)
        clip_to_chunks = defaultdict(list)
        for fname in sorted(os.listdir(take_pose_dir)):
            if not fname.endswith(".npz"):
                continue
            # Parse: <start>___<end>_<chunk>.npz
            stem = fname[:-4]  # remove .npz
            # Find last underscore to split chunk index
            last_underscore = stem.rfind("_")
            if last_underscore == -1:
                continue
            clip_key = stem[:last_underscore]  # e.g., "0___1215"
            clip_to_chunks[clip_key].append(fname)

        for clip_key, chunk_files in clip_to_chunks.items():
            # Check if corresponding video exists
            video_path = os.path.join(take_name, f"{clip_key}.mp4")
            video_full_path = os.path.join(VIDEO_DIR, video_path)

            if not os.path.isfile(video_full_path):
                n_no_video += len(chunk_files)
                continue

            # Create one annotation per chunk
            for i, chunk_fname in enumerate(sorted(chunk_files)):
                tok_pose_path = os.path.join(take_name, chunk_fname)
                caption = DUMMY_CAPTIONS[i % len(DUMMY_CAPTIONS)]

                annotations.append({
                    "video": video_path,
                    "tok_pose": tok_pose_path,
                    "caption": caption,
                })
                n_matched += 1

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Total annotations: {len(annotations)}")
    print(f"  Matched (video+pose): {n_matched}")
    print(f"  Skipped (no video):   {n_no_video}")
    print(f"  Takes scanned:        {len(takes)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok_pose_dir", type=str, default="20",
                        help="Subdirectory under tok_pose/ (e.g., 'all', '20')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__),
            f"annotation_{args.tok_pose_dir}.json"
        )

    build_annotation(args.tok_pose_dir, args.output)
