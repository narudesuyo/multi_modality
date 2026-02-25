#!/usr/bin/env python3
"""
Extract all frames from MP4 clips as JPG images for decord-free training.

Reads an annotation JSON (annotation_atomic_{split}.json), extracts every
frame from each referenced MP4 using ffmpeg, and writes an updated annotation
JSON whose "video" field points to a frame directory instead of an MP4.

Output structure:
  {DATA_ROOT}/EgoExo4D/processed/{split}/frames_atomic/{take_name}/{sample_id}_ego/
      img00001.jpg  img00002.jpg  ...

Usage:
    python extract_frames.py --split train
    python extract_frames.py --split val --workers 8
    python extract_frames.py --split train --quality 2 --skip-existing
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/work/narus/data"))

FFMPEG_TIMEOUT = 60  # seconds per clip


def extract_one(
    mp4_path: str,
    out_dir: str,
    quality: int = 2,
    skip_existing: bool = False,
) -> tuple[bool, str]:
    """Extract all frames from one MP4 as JPGs.

    Returns (ok, message).
    """
    if skip_existing and os.path.isdir(out_dir):
        files = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
        if len(files) > 0:
            return True, f"skip ({len(files)} frames exist)"

    if not os.path.isfile(mp4_path):
        return False, "mp4 not found"

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-q:v", str(quality),  # JPEG quality (2=high, 5=medium)
        os.path.join(out_dir, "img%05d.jpg"),
    ]

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
        )
        if r.returncode != 0:
            # Cleanup partial output
            _cleanup_dir(out_dir)
            msg = (r.stderr.strip() or r.stdout.strip())[-300:]
            return False, f"ffmpeg error: {msg}"

        # Count extracted frames
        n_frames = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
        if n_frames == 0:
            _cleanup_dir(out_dir)
            return False, "0 frames extracted"

        return True, f"ok ({n_frames} frames)"

    except subprocess.TimeoutExpired:
        _cleanup_dir(out_dir)
        return False, f"timeout ({FFMPEG_TIMEOUT}s)"
    except Exception as e:
        _cleanup_dir(out_dir)
        return False, str(e)


def _cleanup_dir(d: str):
    """Remove directory and its contents."""
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--quality", type=int, default=2,
                        help="JPEG quality (1=best, 5=good, 10=low). Default 2.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel extraction workers. 0=sequential. Default 0.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips whose frame dirs already exist.")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Seconds per clip for ffmpeg. Default 60.")
    args = parser.parse_args()

    global FFMPEG_TIMEOUT
    FFMPEG_TIMEOUT = args.timeout

    anno_path = HERE / f"annotation_atomic_{args.split}.json"
    anno_out_path = HERE / f"annotation_atomic_{args.split}_frames.json"
    bad_list_path = HERE / f"bad_frames_{args.split}.txt"

    data_root = DATA_ROOT / "EgoExo4D" / "processed" / args.split
    frames_base = data_root / "frames_atomic"

    print(f"Split:       {args.split}")
    print(f"Annotation:  {anno_path}")
    print(f"Output dir:  {frames_base}")
    print(f"Workers:     {args.workers}")
    print(f"Quality:     {args.quality}")
    print(f"Skip exist:  {args.skip_existing}")
    print()

    with open(anno_path) as f:
        annotations = json.load(f)

    print(f"Total annotation entries: {len(annotations)}")

    # Build extraction tasks: (mp4_abs, out_dir_abs, out_dir_rel, anno_index)
    tasks = []
    for i, ann in enumerate(annotations):
        video_rel = ann["video"]  # e.g. "videos_atomic/take/sample_ego.mp4"
        mp4_abs = str(data_root / video_rel)

        # videos_atomic/take/sample_ego.mp4 → frames_atomic/take/sample_ego/
        frames_rel = video_rel.replace("videos_atomic/", "frames_atomic/")
        frames_rel = frames_rel.rsplit(".", 1)[0]  # drop .mp4
        frames_abs = str(data_root / frames_rel)

        tasks.append((mp4_abs, frames_abs, frames_rel, i))

    # Deduplicate by output dir (same MP4 might appear in multiple annotations)
    seen = {}
    for mp4_abs, frames_abs, frames_rel, idx in tasks:
        if frames_abs not in seen:
            seen[frames_abs] = (mp4_abs, frames_abs, frames_rel, [idx])
        else:
            seen[frames_abs][3].append(idx)

    unique_tasks = list(seen.values())
    print(f"Unique video clips: {len(unique_tasks)}")
    print()

    # Extract frames
    results = {}  # frames_abs → (ok, msg)
    done = 0
    n_ok = 0
    n_bad = 0

    if args.workers <= 0:
        # Sequential execution
        for mp4_abs, frames_abs, frames_rel, indices in unique_tasks:
            ok, msg = extract_one(
                mp4_abs, frames_abs,
                quality=args.quality, skip_existing=args.skip_existing,
            )
            results[frames_abs] = (ok, msg)
            done += 1
            if ok:
                n_ok += 1
            else:
                n_bad += 1
            if done % 500 == 0 or not ok:
                status = "OK" if ok else "BAD"
                print(f"[{done:5d}/{len(unique_tasks)}] {status:3s} {frames_rel}  {msg}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for mp4_abs, frames_abs, frames_rel, indices in unique_tasks:
                fut = pool.submit(
                    extract_one, mp4_abs, frames_abs,
                    quality=args.quality, skip_existing=args.skip_existing,
                )
                futures[fut] = (frames_abs, frames_rel, indices)

            for fut in as_completed(futures):
                frames_abs, frames_rel, indices = futures[fut]
                ok, msg = fut.result()
                results[frames_abs] = (ok, msg)
                done += 1
                if ok:
                    n_ok += 1
                else:
                    n_bad += 1
                if done % 500 == 0 or not ok:
                    status = "OK" if ok else "BAD"
                    print(f"[{done:5d}/{len(unique_tasks)}] {status:3s} {frames_rel}  {msg}")

    print()
    print(f"Done: {n_ok} ok, {n_bad} bad out of {len(unique_tasks)} unique clips")

    # Build new annotation JSON & bad list
    new_annotations = []
    bad_lines = []

    for i, ann in enumerate(annotations):
        video_rel = ann["video"]
        frames_rel = video_rel.replace("videos_atomic/", "frames_atomic/")
        frames_rel = frames_rel.rsplit(".", 1)[0]
        frames_abs = str(data_root / frames_rel)

        ok, msg = results.get(frames_abs, (False, "not processed"))
        if ok:
            new_ann = dict(ann)
            new_ann["video"] = frames_rel  # point to frame directory
            new_annotations.append(new_ann)
        else:
            bad_lines.append(f"{video_rel}\t{msg}")

    with open(anno_out_path, "w") as f:
        json.dump(new_annotations, f, indent=2, ensure_ascii=False)

    with open(bad_list_path, "w") as f:
        for line in bad_lines:
            f.write(line + "\n")

    print()
    print(f"New annotation: {anno_out_path}  ({len(new_annotations)} entries)")
    print(f"Bad list:       {bad_list_path}  ({len(bad_lines)} entries)")
    print()
    if new_annotations:
        print(f"Next step: update config.py anno_path to use '{anno_out_path.name}'")


if __name__ == "__main__":
    main()
