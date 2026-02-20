#!/usr/bin/env python3
"""
Validate all video files in training annotation for decord compatibility.
Bad files (hang / error) are re-encoded with faststart, then a clean
annotation JSON is written.

Usage:
    python validate_videos.py [--fix] [--timeout 20]

Options:
    --fix       Re-encode bad files with ffmpeg instead of removing them.
                Without this flag, bad entries are simply dropped from the JSON.
    --timeout   Seconds to wait per video before declaring it bad (default 20).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
WORK_DIR = HERE / "../../../.."
PYTHON = str(WORK_DIR / ".conda/bin/python")
DATA_ROOT = Path("/large/naru/EgoHand/data/train/takes_clipped/egoexo")
ANNO_FILE = HERE / "annotation_atomic_train.json"
ANNO_OUT  = HERE / "annotation_atomic_train_validated.json"
BAD_LIST  = HERE / "bad_videos.txt"

# ── decord check (runs in subprocess so hangs are killable) ────────────────
_CHECK_SCRIPT = """
import sys, decord
decord.bridge.set_bridge('torch')
vr = decord.VideoReader(sys.argv[1])
n = len(vr)
frames = vr.get_batch([0, n // 2, n - 1])
print(f"OK frames={n} shape={list(frames.shape)}")
"""

def check_video(video_abs: Path, timeout: int) -> tuple[bool, str]:
    """Return (ok, message). Uses subprocess so decord hangs are killed."""
    try:
        r = subprocess.run(
            [PYTHON, "-c", _CHECK_SCRIPT, str(video_abs)],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, (r.stderr.strip() or r.stdout.strip())[-300:]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, str(e)


def fix_video(video_abs: Path) -> bool:
    """Re-encode with -movflags +faststart. Returns True on success."""
    tmp = video_abs.with_suffix(".tmp.mp4")
    ret = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_abs),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-movflags", "+faststart",
            str(tmp),
        ],
        capture_output=True, timeout=120,
    )
    if ret.returncode == 0 and tmp.exists():
        shutil.move(str(tmp), str(video_abs))
        return True
    tmp.unlink(missing_ok=True)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                        help="Re-encode bad files instead of removing them")
    parser.add_argument("--timeout", type=int, default=20,
                        help="Seconds per video (default 20)")
    args = parser.parse_args()

    with open(ANNO_FILE) as f:
        data = json.load(f)

    # Collect unique video paths
    unique_videos = sorted(set(anno["video"] for anno in data))
    print(f"Annotation entries : {len(data)}")
    print(f"Unique video files : {len(unique_videos)}")
    print(f"Timeout per video  : {args.timeout}s")
    print(f"Fix mode           : {'re-encode' if args.fix else 'drop from JSON'}")
    print()

    bad_videos = set()

    for i, rel_path in enumerate(unique_videos, 1):
        abs_path = DATA_ROOT / rel_path
        prefix = f"[{i:3d}/{len(unique_videos)}]"

        if not abs_path.exists():
            print(f"{prefix} MISSING  {rel_path}")
            bad_videos.add(rel_path)
            continue

        ok, msg = check_video(abs_path, args.timeout)
        if ok:
            print(f"{prefix} OK       {rel_path}  ({msg})")
        else:
            print(f"{prefix} BAD      {rel_path}  ({msg})")
            bad_videos.add(rel_path)

            if args.fix:
                print(f"           → re-encoding ...", end=" ", flush=True)
                if fix_video(abs_path):
                    # Verify the re-encoded file
                    ok2, msg2 = check_video(abs_path, args.timeout)
                    if ok2:
                        print(f"fixed ({msg2})")
                        bad_videos.discard(rel_path)
                    else:
                        print(f"still bad after re-encode: {msg2}")
                else:
                    print("ffmpeg failed")

    # ── summary ───────────────────────────────────────────────────────────
    print()
    print(f"{'='*60}")
    print(f"Bad / unfixable videos: {len(bad_videos)}")
    for v in sorted(bad_videos):
        print(f"  {v}")

    # Write bad list
    BAD_LIST.write_text("\n".join(sorted(bad_videos)) + "\n")
    print(f"\nBad file list saved to: {BAD_LIST}")

    # Write filtered annotation
    if bad_videos:
        clean = [anno for anno in data if anno["video"] not in bad_videos]
        dropped = len(data) - len(clean)
        with open(ANNO_OUT, "w") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        print(f"Clean annotation ({len(clean)} entries, dropped {dropped}) → {ANNO_OUT}")
        print()
        if not args.fix:
            print(f"Run with --fix to re-encode bad files instead of dropping them.")
    else:
        print("All videos OK — no changes needed.")


if __name__ == "__main__":
    main()
