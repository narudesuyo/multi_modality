#!/usr/bin/env python3
"""
Prepare atomic-description-aligned 4-second data for Stage2 training/val.

For each atomic description timestamp (±2 sec):
  1. Extract exo video frames (best_exo camera) as JPG directly (no MP4 intermediate)
  2. Extract corresponding motion kp3d from ee_{split}_joints_tips.pt
  3. Collect text descriptions

By default only exo view is processed (--exo-only). Use --include-ego to also
extract ego frames from pre-clipped videos.

Usage:
    python prepare_atomic_clips.py --split train --skip-existing
    python prepare_atomic_clips.py --split val --skip-existing
    python prepare_atomic_clips.py --split train --include-ego
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("DATA_ROOT", "/work/narus/data")

FPS = 30
HALF_WINDOW_SEC = 2.0
HALF_WINDOW_FRAMES = int(HALF_WINDOW_SEC * FPS)  # 60 frames

# ffmpeg settings (safe for most decoders)
ENC_CODEC = "libx264"
ENC_PRESET = "veryfast"      # you can change to ultrafast/fast
ENC_CRF = "23"               # lower = higher quality/larger
PIX_FMT = "yuv420p"
MOVFLAGS = "+faststart"

# Load-aware throttling: pause before spawning ffmpeg if system is busy.
# 36 physical cores (2x Xeon Gold 6254); threshold at ~80%.
LOAD_THRESHOLD = 40.0
LOAD_CHECK_INTERVAL = 10  # seconds to sleep between checks

FFMPEG_THREADS = 4

# Resize short side to this value (preserving aspect ratio).
# Set to 0 or None to disable resizing (keep original resolution).
FRAME_SHORT_SIDE = 320


def _wait_for_low_load() -> None:
    """Block until 1-min load average drops below LOAD_THRESHOLD."""
    while os.getloadavg()[0] > LOAD_THRESHOLD:
        time.sleep(LOAD_CHECK_INTERVAL)


# Use the same python as current interpreter for decord validation
PYTHON = sys.executable

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
def load_uuid_to_take_name(takes_json_path: str) -> dict:
    """Build mapping from take_uid -> take_name."""
    with open(takes_json_path, "r") as f:
        takes = json.load(f)
    return {t["take_uid"]: t["take_name"] for t in takes}


def list_ego_clips(take_name: str, clipped_video_dir: str) -> list[tuple[int, int, str]]:
    """Return [(start_frame, end_frame, full_path), ...] for a take's ego clips."""
    take_dir = os.path.join(clipped_video_dir, take_name)
    if not os.path.isdir(take_dir):
        return []
    clips = []
    for fname in sorted(os.listdir(take_dir)):
        if not fname.endswith(".mp4"):
            continue
        stem = fname[:-4]
        parts = stem.split("___")
        if len(parts) != 2:
            continue
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        clips.append((start, end, os.path.join(take_dir, fname)))
    return clips


def find_clip_for_frame(clips: list[tuple[int, int, str]], frame: int) -> Optional[tuple[int, int, str]]:
    """Find the clip that contains the given absolute frame number."""
    for start, end, path in clips:
        if start <= frame <= end:
            return start, end, path
    return None


def find_exo_video(take_name: str, cam_id: str, full_takes_dir: str) -> Optional[str]:
    """Find the exo camera video from full takes."""
    video_dir = os.path.join(full_takes_dir, take_name, "frame_aligned_videos")
    if not os.path.isdir(video_dir):
        return None
    video_path = os.path.join(video_dir, f"{cam_id}.mp4")
    if os.path.isfile(video_path):
        return video_path
    return None


_DECORD_CHECK_SCRIPT = r"""
import sys
import decord
decord.bridge.set_bridge('torch')
p = sys.argv[1]
vr = decord.VideoReader(p)
n = len(vr)
# touch frames to ensure decoding
idxs = [0, n//2, n-1] if n >= 3 else list(range(n))
frames = vr.get_batch(idxs)
print("OK", n, list(frames.shape))
"""


def validate_with_decord(video_path: str, timeout: int = 20) -> tuple[bool, str]:
    """Validate mp4 readability with decord in a killable subprocess."""
    try:
        r = subprocess.run(
            [PYTHON, "-c", _DECORD_CHECK_SCRIPT, video_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            return True, r.stdout.strip()
        msg = (r.stderr.strip() or r.stdout.strip())[-400:]
        return False, msg
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT>{timeout}s"
    except Exception as e:
        return False, str(e)


def clip_video_ffmpeg_safe(
    src_path: str,
    dst_path: str,
    start_sec: float,
    duration_sec: float,
    dry_run: bool = False,
    validate: bool = False,
    validate_timeout: int = 20,
    log_dir: Optional[str] = None,
) -> bool:
    """Clip a video segment using ffmpeg (re-encode; safe for decord)."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # optional: store ffmpeg stderr/stdout for debugging
    log_path = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, Path(dst_path).name + ".ffmpeg.log")

    # -ss BEFORE -i: keyframe-based seek (fast); minor frame offset is acceptable
    # for 4-second clips.
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.4f}",
        "-i", src_path,
        "-t", f"{duration_sec:.4f}",
        "-an",
        "-c:v", ENC_CODEC,
        "-preset", ENC_PRESET,
        "-crf", str(ENC_CRF),
        "-pix_fmt", PIX_FMT,
        "-movflags", MOVFLAGS,
        "-threads", str(FFMPEG_THREADS),
        dst_path,
    ]

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return True

    _wait_for_low_load()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if log_path is not None:
            with open(log_path, "w") as f:
                f.write("CMD:\n" + " ".join(cmd) + "\n\n")
                f.write("STDOUT:\n" + (r.stdout or "") + "\n\n")
                f.write("STDERR:\n" + (r.stderr or "") + "\n")

        if r.returncode != 0:
            return False

        if not os.path.exists(dst_path) or os.path.getsize(dst_path) < 1024:
            return False

        if validate:
            ok, msg = validate_with_decord(dst_path, timeout=validate_timeout)
            if not ok:
                # keep log; remove broken output
                try:
                    os.remove(dst_path)
                except OSError:
                    pass
                return False

        return True

    except subprocess.TimeoutExpired:
        # ffmpeg hung (rare), cleanup
        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)
        except OSError:
            pass
        return False


def extract_frames_ffmpeg(
    src_path: str,
    dst_dir: str,
    start_sec: float,
    duration_sec: float,
    quality: int = 2,
    dry_run: bool = False,
    log_dir: Optional[str] = None,
) -> bool:
    """Extract JPG frames directly from a video segment using ffmpeg.

    Output: dst_dir/img00001.jpg, img00002.jpg, ...
    Returns True if at least 1 frame was extracted.
    """
    os.makedirs(dst_dir, exist_ok=True)

    log_path = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, Path(dst_dir).name + ".ffmpeg.log")

    # -ss BEFORE -i: keyframe-based seek (fast).
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.4f}",
        "-i", src_path,
        "-t", f"{duration_sec:.4f}",
        "-an",
    ]
    if FRAME_SHORT_SIDE:
        cmd += ["-vf", f"scale=-2:{FRAME_SHORT_SIDE}"]
    cmd += [
        "-q:v", str(quality),
        "-threads", str(FFMPEG_THREADS),
        os.path.join(dst_dir, "img%05d.jpg"),
    ]

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return True

    _wait_for_low_load()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if log_path is not None:
            with open(log_path, "w") as f:
                f.write("CMD:\n" + " ".join(cmd) + "\n\n")
                f.write("STDOUT:\n" + (r.stdout or "") + "\n\n")
                f.write("STDERR:\n" + (r.stderr or "") + "\n")

        if r.returncode != 0:
            return False

        # Check at least 1 frame was extracted
        n_frames = len([f for f in os.listdir(dst_dir) if f.endswith(".jpg")])
        return n_frames > 0

    except subprocess.TimeoutExpired:
        import shutil
        shutil.rmtree(dst_dir, ignore_errors=True)
        return False


def find_motion_key_for_frame(
    motion_keys_by_take: dict[str, list[tuple[int, int, str]]],
    take_name: str,
    frame: int,
) -> Optional[tuple[int, int, str]]:
    """Find the motion key whose [start, end] range contains the given frame."""
    for start, end, key in motion_keys_by_take.get(take_name, []):
        if start <= frame <= end:
            return start, end, key
    return None


TARGET_MOTION_FRAMES = 41  # Fixed output length for kp3d slices


def extract_motion_slice(
    motion_data: dict,
    key: str,
    abs_frame_start: int,
    abs_frame_end: int,
) -> Optional[np.ndarray]:
    """Extract a slice of kp3d from motion data for the given absolute frame range.
    Always resamples to TARGET_MOTION_FRAMES frames via linear interpolation.
    Returns: kp3d array [TARGET_MOTION_FRAMES, 154, 3] or None if key not found.
    """
    item = motion_data.get(key)
    if item is None:
        return None

    kp3d = item["kp3d"]
    if torch.is_tensor(kp3d):
        kp3d = kp3d.detach().cpu().numpy()

    clip_start = int(item["start_idx"])
    clip_end = int(item["end_idx"])
    num_frames = int(kp3d.shape[0])

    total_clip_frames = clip_end - clip_start
    if total_clip_frames <= 0 or num_frames <= 1:
        return None

    # Proportional mapping from absolute frames to kp3d indices
    def to_idx(abs_f: int) -> float:
        x = (abs_f - clip_start) / float(total_clip_frames)
        return x * num_frames

    src_start = to_idx(abs_frame_start)
    src_end = to_idx(abs_frame_end)

    # clamp
    src_start = max(0.0, min(src_start, num_frames - 1.0))
    src_end = max(src_start + 0.01, min(src_end, float(num_frames)))

    # Resample to exactly TARGET_MOTION_FRAMES via linear interpolation
    sample_pts = np.linspace(src_start, src_end - 1e-6, TARGET_MOTION_FRAMES)
    T, J, C = kp3d.shape
    out = np.empty((TARGET_MOTION_FRAMES, J, C), dtype=np.float32)
    for i, pt in enumerate(sample_pts):
        lo = int(pt)
        hi = min(lo + 1, T - 1)
        alpha = pt - lo
        out[i] = (1.0 - alpha) * kp3d[lo] + alpha * kp3d[hi]

    return out


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="train",
                        help="Dataset split to process (default: train)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path for intermediate annotation JSON")
    parser.add_argument("--output-json-clean", type=str, default=None,
                        help="Path for cleaned (only fully-ok) annotation JSON")
    parser.add_argument("--bad-list", type=str, default=None,
                        help="Path to write bad sample list")
    parser.add_argument("--validate", action="store_true",
                        help="Validate output mp4 with decord (slower, safer)")
    parser.add_argument("--validate-timeout", type=int, default=20,
                        help="Seconds per output clip for decord validation (default 20)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output files already exist")
    parser.add_argument("--exo-only", action="store_true", default=True,
                        help="Only clip exo video, skip ego (default: True)")
    parser.add_argument("--include-ego", action="store_true",
                        help="Also clip ego video (overrides --exo-only)")
    parser.add_argument("--require-both-videos", action="store_true",
                        help="Only keep samples that have BOTH ego and exo videos")
    parser.add_argument("--require-motion", action="store_true",
                        help="Only keep samples that have motion_kp3d")
    parser.add_argument("--motion-only", action="store_true",
                        help="Re-generate motion kp3d only (skip video clipping)")
    parser.add_argument("--full-takes-dir", type=str, default=None,
                        help="Directory containing full takes (for exo camera). "
                             "Default: {DATA_ROOT}/EgoExo4D/takes")
    parser.add_argument("--clipped-video-dir", type=str, default=None,
                        help="Directory containing pre-clipped ego videos. "
                             "Default: {DATA_ROOT}/EgoExo4D/processed/{split}/videos")
    parser.add_argument("--output-base", type=str, default=None,
                        help="Output base directory. "
                             "Default: {DATA_ROOT}/EgoExo4D/processed/{split}")
    parser.add_argument("--frames-dir-name", type=str, default="frames_atomic",
                        help="Name of the frames subdirectory (default: frames_atomic). "
                             "Use e.g. 'frames_atomic_320' to save resized frames "
                             "separately from existing full-resolution frames.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers for processing takes (default: 1). "
                             "Set to e.g. 16 to parallelize ffmpeg across takes.")
    args = parser.parse_args()

    here = Path(__file__).parent

    if args.output_json is None:
        args.output_json = str(here / f"annotation_atomic_{args.split}_intermediate.json")

    if args.output_json_clean is None:
        args.output_json_clean = str(here / f"annotation_atomic_{args.split}.json")

    if args.bad_list is None:
        args.bad_list = str(here / f"bad_atomic_samples_{args.split}.txt")

    # ── paths derived from split ────────────────────────────────────────────
    atomic_desc_path = os.path.join(DATA_ROOT, "description", f"atomic_descriptions_{args.split}.json")
    takes_json_path = os.path.join(DATA_ROOT, "ee4d", "ee4d_motion_uniegomotion", "takes.json")
    motion_pt_path = os.path.join(
        DATA_ROOT, "ee4d", "ee4d_motion_uniegomotion", "uniegomotion", f"ee_{args.split}_joints_tips.pt"
    )

    # Pre-clipped ego videos (may not exist for train split)
    if args.clipped_video_dir is not None:
        clipped_video_dir = args.clipped_video_dir
    else:
        clipped_video_dir = os.path.join(DATA_ROOT, "EgoExo4D", "processed", args.split, "videos")

    # Full takes directory (same for all splits — raw data lives under EgoExo4D/)
    if args.full_takes_dir is not None:
        full_takes_dir = args.full_takes_dir
    else:
        full_takes_dir = os.path.join(DATA_ROOT, "EgoExo4D", "takes")

    if args.output_base is not None:
        output_base = args.output_base
    else:
        output_base = os.path.join(DATA_ROOT, "EgoExo4D", "processed", args.split)
    frames_dir_name = args.frames_dir_name
    output_frames_dir = os.path.join(output_base, frames_dir_name)
    output_motion_dir = os.path.join(output_base, "motion_atomic")
    ffmpeg_log_dir = os.path.join(output_base, "ffmpeg_logs_atomic")

    # Resolve exo-only vs include-ego
    exo_only = args.exo_only and not args.include_ego

    print(f"Split: {args.split}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Frames dir: {frames_dir_name}")
    print(f"Exo only: {exo_only}")
    print(f"Validate output: {args.validate} (timeout={args.validate_timeout}s)")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Require both videos: {args.require_both_videos}")
    print(f"Require motion: {args.require_motion}")
    print()

    # ── Load metadata ───────────────────────────────────────────────────────
    print("Loading atomic descriptions...")
    with open(atomic_desc_path, "r") as f:
        atomic_data = json.load(f)

    print("Loading takes.json...")
    uuid_to_take = load_uuid_to_take_name(takes_json_path)

    print("Loading motion data (this may take a while)...")
    if not args.dry_run:
        motion_data = torch.load(motion_pt_path, map_location="cpu", weights_only=False)
    else:
        motion_data = {}

    # Build take_name -> [(start, end, key), ...] index for motion data
    motion_keys_by_take: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    if not args.dry_run:
        for key in motion_data.keys():
            parts = key.rsplit("___", 2)
            if len(parts) == 3:
                take_name_m = parts[0]
                try:
                    s, e = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                motion_keys_by_take[take_name_m].append((s, e, key))
        for k in motion_keys_by_take:
            motion_keys_by_take[k].sort()

    # ── Pre-compute sample index offsets per take ──────────────────────────
    annotations = atomic_data["annotations"]
    take_items = list(annotations.items())

    sample_offsets: dict[str, int] = {}
    idx = 0
    for t_uid, a_list in take_items:
        sample_offsets[t_uid] = idx
        for anno in a_list:
            if anno.get("rejected", False):
                continue
            idx += len(anno.get("descriptions", []))

    # Disable load throttling in parallel mode (dedicated PBS node)
    if args.num_workers > 1:
        global LOAD_THRESHOLD
        LOAD_THRESHOLD = float('inf')
        print(f"[INFO]: Parallel mode: {args.num_workers} workers, load throttling disabled")

    results = []
    bad_samples = []  # store sample_id and reason

    stats = {
        "total_descriptions": 0,
        "no_take_mapping": 0,
        "no_ego_clip": 0,
        "ego_ok": 0,
        "no_exo_video": 0,
        "exo_ok": 0,
        "no_motion_key": 0,
        "motion_ok": 0,
        "skipped_existing": 0,
        "bad_output_clip": 0,
    }

    def process_take(take_uid_and_annos):
        """Process a single take: extract frames, motion, collect annotations."""
        take_uid, anno_list = take_uid_and_annos
        local_results = []
        local_bad = []
        local_stats = {k: 0 for k in stats}

        sample_idx = sample_offsets[take_uid]
        take_name = uuid_to_take.get(take_uid)
        if take_name is None:
            for anno in anno_list:
                local_stats["no_take_mapping"] += len(anno.get("descriptions", []))
            return local_results, local_bad, local_stats

        ego_clips = list_ego_clips(take_name, clipped_video_dir)

        for anno in anno_list:
            if anno.get("rejected", False):
                continue
            descriptions = anno.get("descriptions", [])

            for desc in descriptions:
                local_stats["total_descriptions"] += 1

                timestamp = float(desc["timestamp"])  # seconds
                text = desc["text"]
                best_exo = desc.get("best_exo") or {}
                exo_cam_id = best_exo.get("cam_id", "cam01") if best_exo else "cam01"

                center_frame = int(round(timestamp * FPS))
                frame_start = max(0, center_frame - HALF_WINDOW_FRAMES)
                frame_end = center_frame + HALF_WINDOW_FRAMES

                sample_id = f"{sample_idx:06d}"
                sample_idx += 1

                entry = {
                    "take_name": take_name,
                    "sample_id": sample_id,
                    "caption": text,
                    "timestamp": timestamp,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                }

                # Outputs (JPG frames directories)
                ego_frames_out = os.path.join(output_frames_dir, take_name, f"{sample_id}_ego")
                exo_frames_out = os.path.join(output_frames_dir, take_name, f"{sample_id}_exo")
                ego_rel = os.path.join(frames_dir_name, take_name, f"{sample_id}_ego")
                exo_rel = os.path.join(frames_dir_name, take_name, f"{sample_id}_exo")

                # ── Ego frames ──────────────────────────────────────────
                ego_ok = False
                if exo_only:
                    pass  # skip ego entirely
                elif args.motion_only:
                    if os.path.isdir(ego_frames_out) and any(
                        f.endswith(".jpg") for f in os.listdir(ego_frames_out)
                    ):
                        ego_ok = True
                        entry["video_ego"] = ego_rel
                        local_stats["ego_ok"] += 1
                else:
                    ego_result = find_clip_for_frame(ego_clips, center_frame)
                    if ego_result is not None:
                        clip_start, clip_end, clip_path = ego_result

                        rel_start_sec = max(0.0, (frame_start - clip_start) / FPS)
                        duration_sec = (frame_end - frame_start) / FPS

                        max_duration = (clip_end - clip_start) / FPS - rel_start_sec
                        duration_sec = max(0.0, min(duration_sec, max_duration))

                        if duration_sec >= 0.2:
                            if args.skip_existing and os.path.isdir(ego_frames_out) and any(
                                f.endswith(".jpg") for f in os.listdir(ego_frames_out)
                            ):
                                ego_ok = True
                                local_stats["skipped_existing"] += 1
                            else:
                                ego_ok = extract_frames_ffmpeg(
                                    clip_path, ego_frames_out, rel_start_sec, duration_sec,
                                    dry_run=args.dry_run,
                                    log_dir=ffmpeg_log_dir,
                                )
                            if ego_ok:
                                entry["video_ego"] = ego_rel
                                local_stats["ego_ok"] += 1
                            else:
                                local_stats["bad_output_clip"] += 1
                                local_bad.append(f"{take_name}/{sample_id} ego ffmpeg failed")
                        else:
                            local_stats["bad_output_clip"] += 1
                            local_bad.append(f"{take_name}/{sample_id} ego too_short duration={duration_sec:.3f}")
                    else:
                        local_stats["no_ego_clip"] += 1
                        local_bad.append(f"{take_name}/{sample_id} no_ego_clip")

                # ── Exo frames ──────────────────────────────────────────
                exo_ok = False
                if args.motion_only:
                    if os.path.isdir(exo_frames_out) and any(
                        f.endswith(".jpg") for f in os.listdir(exo_frames_out)
                    ):
                        exo_ok = True
                        entry["video_exo"] = exo_rel
                        local_stats["exo_ok"] += 1
                else:
                    exo_src = find_exo_video(take_name, exo_cam_id, full_takes_dir)
                    if exo_src is not None:
                        exo_start_sec = max(0.0, frame_start / FPS)
                        exo_duration = (frame_end - frame_start) / FPS
                        if exo_duration >= 0.2:
                            if args.skip_existing and os.path.isdir(exo_frames_out) and any(
                                f.endswith(".jpg") for f in os.listdir(exo_frames_out)
                            ):
                                exo_ok = True
                                local_stats["skipped_existing"] += 1
                            else:
                                exo_ok = extract_frames_ffmpeg(
                                    exo_src, exo_frames_out, exo_start_sec, exo_duration,
                                    dry_run=args.dry_run,
                                    log_dir=ffmpeg_log_dir,
                                )
                            if exo_ok:
                                entry["video_exo"] = exo_rel
                                local_stats["exo_ok"] += 1
                            else:
                                local_stats["bad_output_clip"] += 1
                                local_bad.append(f"{take_name}/{sample_id} exo ffmpeg failed cam={exo_cam_id}")
                        else:
                            local_stats["bad_output_clip"] += 1
                            local_bad.append(f"{take_name}/{sample_id} exo too_short duration={exo_duration:.3f}")
                    else:
                        local_stats["no_exo_video"] += 1
                        local_bad.append(f"{take_name}/{sample_id} no_exo_video cam={exo_cam_id}")

                # ── Motion slice ────────────────────────────────────────
                motion_ok = False
                if not args.dry_run:
                    motion_result = find_motion_key_for_frame(motion_keys_by_take, take_name, center_frame)
                    if motion_result is not None:
                        _, _, m_key = motion_result
                        kp3d_slice = extract_motion_slice(motion_data, m_key, frame_start, frame_end)
                        if kp3d_slice is not None and kp3d_slice.shape[0] > 0:
                            motion_out = os.path.join(output_motion_dir, take_name, f"{sample_id}_kp3d.npy")
                            motion_rel = os.path.join("motion_atomic", take_name, f"{sample_id}_kp3d.npy")
                            os.makedirs(os.path.dirname(motion_out), exist_ok=True)
                            np.save(motion_out, kp3d_slice)
                            entry["motion_kp3d"] = motion_rel
                            local_stats["motion_ok"] += 1
                            motion_ok = True
                        else:
                            local_stats["no_motion_key"] += 1
                            local_bad.append(f"{take_name}/{sample_id} motion_slice_empty")
                    else:
                        local_stats["no_motion_key"] += 1
                        local_bad.append(f"{take_name}/{sample_id} no_motion_key")
                else:
                    # dry-run: assume motion ok
                    motion_ok = True
                    entry["motion_kp3d"] = os.path.join("motion_atomic", take_name, f"{sample_id}_kp3d.npy")
                    local_stats["motion_ok"] += 1

                # Decide keep/drop
                keep = True
                if args.require_both_videos and not (ego_ok and exo_ok):
                    keep = False
                if args.require_motion and not motion_ok:
                    keep = False

                entry["_ego_ok"] = bool(ego_ok)
                entry["_exo_ok"] = bool(exo_ok)
                entry["_motion_ok"] = bool(motion_ok)

                local_results.append(entry)

        return local_results, local_bad, local_stats

    # ── Run processing (parallel or sequential) ───────────────────────────
    if args.num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for local_results, local_bad, local_stats in tqdm(
                executor.map(process_take, take_items),
                total=len(take_items), desc="Processing takes"
            ):
                results.extend(local_results)
                bad_samples.extend(local_bad)
                for k in stats:
                    stats[k] += local_stats[k]
    else:
        for item in tqdm(take_items, desc="Processing takes"):
            local_results, local_bad, local_stats = process_take(item)
            results.extend(local_results)
            bad_samples.extend(local_bad)
            for k in stats:
                stats[k] += local_stats[k]

    # ── Save intermediate annotation (keeps everything + ok flags) ────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Save clean annotation (only selected criteria) ───────────────────────
    clean = []
    for e in results:
        ego_ok = e.get("_ego_ok", False)
        exo_ok = e.get("_exo_ok", False)
        motion_ok = e.get("_motion_ok", False)
        keep = True
        if args.require_both_videos and not (ego_ok and exo_ok):
            keep = False
        if args.require_motion and not motion_ok:
            keep = False
        if keep:
            # remove internal flags
            e2 = dict(e)
            e2.pop("_ego_ok", None)
            e2.pop("_exo_ok", None)
            e2.pop("_motion_ok", None)
            clean.append(e2)

    with open(args.output_json_clean, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    # ── Save bad list ────────────────────────────────────────────────────────
    with open(args.bad_list, "w") as f:
        for line in bad_samples:
            f.write(line + "\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Intermediate JSON : {args.output_json}  (entries={len(results)})")
    print(f"Clean JSON        : {args.output_json_clean}  (entries={len(clean)})")
    print(f"Bad list          : {args.bad_list}  (lines={len(bad_samples)})")
    print("-" * 60)
    print(f"Total descriptions processed: {stats['total_descriptions']}")
    print(f"No take mapping:   {stats['no_take_mapping']}")
    print(f"Ego OK:            {stats['ego_ok']}")
    print(f"No ego clip:       {stats['no_ego_clip']}")
    print(f"Exo OK:            {stats['exo_ok']}")
    print(f"No exo video:      {stats['no_exo_video']}")
    print(f"Motion OK:         {stats['motion_ok']}")
    print(f"No motion key:     {stats['no_motion_key']}")
    print(f"Skipped existing:  {stats['skipped_existing']}")
    print(f"Bad output clips:  {stats['bad_output_clip']}")
    print("=" * 60)


if __name__ == "__main__":
    main()