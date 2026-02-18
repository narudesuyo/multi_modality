"""Prepare atomic-description-aligned 4-second clips for Stage2 training.

For each atomic description timestamp (±2 sec):
  1. Clip ego video from existing clipped videos
  2. Clip exo video from full takes (best_exo camera)
  3. Extract corresponding motion kp3d from ee_train_joints_tips.pt
  4. Collect text descriptions

Usage:
    python prepare_atomic_clips.py [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("DATA_ROOT", "/large/naru/EgoHand/data")
ATOMIC_DESC_PATH = os.path.join(DATA_ROOT, "description", "atomic_descriptions_train.json")
TAKES_JSON_PATH = os.path.join(DATA_ROOT, "ee4d", "ee4d_motion_uniegomotion", "takes.json")
MOTION_PT_PATH = os.path.join(
    DATA_ROOT, "ee4d", "ee4d_motion_uniegomotion", "uniegomotion", "ee_train_joints_tips.pt"
)

CLIPPED_VIDEO_DIR = os.path.join(DATA_ROOT, "train", "takes_clipped", "egoexo", "videos")
FULL_TAKES_DIR = os.path.join(DATA_ROOT, "train", "takes")

OUTPUT_BASE = os.path.join(DATA_ROOT, "train", "takes_clipped", "egoexo")
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_BASE, "videos_atomic")
OUTPUT_MOTION_DIR = os.path.join(OUTPUT_BASE, "motion_atomic")

FPS = 30
HALF_WINDOW_SEC = 2.0
HALF_WINDOW_FRAMES = int(HALF_WINDOW_SEC * FPS)  # 60


# ── helpers ────────────────────────────────────────────────────────────
def load_uuid_to_take_name(takes_json_path: str) -> dict:
    """Build mapping from take_uid -> take_name."""
    with open(takes_json_path, "r") as f:
        takes = json.load(f)
    return {t["take_uid"]: t["take_name"] for t in takes}


def list_ego_clips(take_name: str) -> list[tuple[int, int, str]]:
    """Return [(start_frame, end_frame, full_path), ...] for a take's ego clips."""
    take_dir = os.path.join(CLIPPED_VIDEO_DIR, take_name)
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


def find_clip_for_frame(clips: list[tuple[int, int, str]], frame: int):
    """Find the clip that contains the given absolute frame number."""
    for start, end, path in clips:
        if start <= frame <= end:
            return start, end, path
    return None


def find_exo_video(take_name: str, cam_id: str) -> str | None:
    """Find the exo camera video from full takes."""
    video_dir = os.path.join(FULL_TAKES_DIR, take_name, "frame_aligned_videos")
    if not os.path.isdir(video_dir):
        return None
    video_path = os.path.join(video_dir, f"{cam_id}.mp4")
    if os.path.isfile(video_path):
        return video_path
    return None


def clip_video_ffmpeg(
    src_path: str,
    dst_path: str,
    start_sec: float,
    duration_sec: float,
    dry_run: bool = False,
) -> bool:
    """Clip a video segment using ffmpeg."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.4f}",
        "-i", src_path,
        "-t", f"{duration_sec:.4f}",
        "-c:v", "copy",
        "-an",
        dst_path,
    ]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return True
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  [warn] ffmpeg failed for {dst_path}: {e}", file=sys.stderr)
        return False


def find_motion_key_for_frame(
    motion_keys_by_take: dict[str, list[tuple[int, int, str]]],
    take_name: str,
    frame: int,
):
    """Find the motion key whose [start, end] range contains the given frame."""
    for start, end, key in motion_keys_by_take.get(take_name, []):
        if start <= frame <= end:
            return start, end, key
    return None


def extract_motion_slice(
    motion_data: dict,
    key: str,
    abs_frame_start: int,
    abs_frame_end: int,
) -> np.ndarray | None:
    """Extract a slice of kp3d from motion data for the given absolute frame range.

    Returns: kp3d array [T_slice, 154, 3] or None if key not found.
    """
    item = motion_data.get(key)
    if item is None:
        return None

    kp3d = item["kp3d"]
    if torch.is_tensor(kp3d):
        kp3d = kp3d.numpy()

    clip_start = item["start_idx"]
    clip_end = item["end_idx"]
    num_frames = kp3d.shape[0]

    # Map absolute frames to motion array indices (proportional mapping)
    total_clip_frames = clip_end - clip_start
    if total_clip_frames <= 0:
        return None

    idx_start = int(round((abs_frame_start - clip_start) / total_clip_frames * num_frames))
    idx_end = int(round((abs_frame_end - clip_start) / total_clip_frames * num_frames))

    # Clamp to valid range
    idx_start = max(0, min(idx_start, num_frames - 1))
    idx_end = max(idx_start + 1, min(idx_end, num_frames))

    return kp3d[idx_start:idx_end].copy()


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path for intermediate annotation JSON")
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = os.path.join(os.path.dirname(__file__), "annotation_atomic_intermediate.json")

    # ── 1. Load metadata ───────────────────────────────────────────────
    print("Loading atomic descriptions...")
    with open(ATOMIC_DESC_PATH, "r") as f:
        atomic_data = json.load(f)

    print("Loading takes.json...")
    uuid_to_take = load_uuid_to_take_name(TAKES_JSON_PATH)

    print("Loading motion data (this may take a while)...")
    if not args.dry_run:
        motion_data = torch.load(MOTION_PT_PATH, map_location="cpu", weights_only=False)
    else:
        motion_data = {}

    # Build take_name -> [(start, end, key), ...] index for motion data
    motion_keys_by_take: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for key in (motion_data.keys() if not args.dry_run else []):
        parts = key.rsplit("___", 2)
        if len(parts) == 3:
            take_name_m = parts[0]
            try:
                s, e = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            motion_keys_by_take[take_name_m].append((s, e, key))

    # Sort each take's clips by start frame
    for k in motion_keys_by_take:
        motion_keys_by_take[k].sort()

    # ── 2. Process annotations ─────────────────────────────────────────
    annotations = atomic_data["annotations"]
    results = []
    stats = {
        "total_descriptions": 0,
        "no_take_mapping": 0,
        "no_ego_clip": 0,
        "no_exo_video": 0,
        "no_motion_key": 0,
        "ego_ok": 0,
        "exo_ok": 0,
        "motion_ok": 0,
    }

    sample_idx = 0

    for take_uid, anno_list in tqdm(annotations.items(), desc="Processing takes"):
        take_name = uuid_to_take.get(take_uid)
        if take_name is None:
            for anno in anno_list:
                stats["no_take_mapping"] += len(anno.get("descriptions", []))
            continue

        ego_clips = list_ego_clips(take_name)

        for anno in anno_list:
            if anno.get("rejected", False):
                continue
            descriptions = anno.get("descriptions", [])

            for desc in descriptions:
                stats["total_descriptions"] += 1
                timestamp = desc["timestamp"]  # seconds
                text = desc["text"]
                best_exo = desc.get("best_exo") or {}
                exo_cam_id = best_exo.get("cam_id", "cam01") if best_exo else "cam01"

                # Compute absolute frame range
                center_frame = int(round(timestamp * FPS))
                frame_start = max(0, center_frame - HALF_WINDOW_FRAMES)
                frame_end = center_frame + HALF_WINDOW_FRAMES

                # Output ID for this sample
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

                # ── Ego video clip ──
                ego_result = find_clip_for_frame(ego_clips, center_frame)
                if ego_result is not None:
                    clip_start, clip_end, clip_path = ego_result
                    # Compute relative time within the ego clip
                    rel_start_sec = max(0, (frame_start - clip_start) / FPS)
                    duration_sec = (frame_end - frame_start) / FPS
                    # Clamp duration to not exceed clip
                    max_duration = (clip_end - clip_start) / FPS - rel_start_sec
                    duration_sec = min(duration_sec, max_duration)

                    ego_out = os.path.join(OUTPUT_VIDEO_DIR, take_name, f"{sample_id}_ego.mp4")
                    ego_rel = os.path.join("videos_atomic", take_name, f"{sample_id}_ego.mp4")
                    ok = clip_video_ffmpeg(clip_path, ego_out, rel_start_sec, duration_sec, args.dry_run)
                    if ok:
                        entry["video_ego"] = ego_rel
                        stats["ego_ok"] += 1
                else:
                    stats["no_ego_clip"] += 1

                # ── Exo video clip ──
                exo_src = find_exo_video(take_name, exo_cam_id)
                if exo_src is not None:
                    # Exo is full-length video, use absolute timestamp
                    exo_start_sec = max(0, frame_start / FPS)
                    exo_duration = (frame_end - frame_start) / FPS

                    exo_out = os.path.join(OUTPUT_VIDEO_DIR, take_name, f"{sample_id}_exo.mp4")
                    exo_rel = os.path.join("videos_atomic", take_name, f"{sample_id}_exo.mp4")
                    ok = clip_video_ffmpeg(exo_src, exo_out, exo_start_sec, exo_duration, args.dry_run)
                    if ok:
                        entry["video_exo"] = exo_rel
                        stats["exo_ok"] += 1
                else:
                    stats["no_exo_video"] += 1

                # ── Motion extraction ──
                motion_result = find_motion_key_for_frame(
                    motion_keys_by_take, take_name, center_frame
                )
                if motion_result is not None and not args.dry_run:
                    m_start, m_end, m_key = motion_result
                    kp3d_slice = extract_motion_slice(
                        motion_data, m_key, frame_start, frame_end
                    )
                    if kp3d_slice is not None and kp3d_slice.shape[0] > 0:
                        motion_out = os.path.join(
                            OUTPUT_MOTION_DIR, take_name, f"{sample_id}_kp3d.npy"
                        )
                        motion_rel = os.path.join(
                            "motion_atomic", take_name, f"{sample_id}_kp3d.npy"
                        )
                        os.makedirs(os.path.dirname(motion_out), exist_ok=True)
                        np.save(motion_out, kp3d_slice)
                        entry["motion_kp3d"] = motion_rel
                        stats["motion_ok"] += 1
                    else:
                        stats["no_motion_key"] += 1
                elif motion_result is not None and args.dry_run:
                    _, _, m_key = motion_result
                    motion_rel = os.path.join(
                        "motion_atomic", take_name, f"{sample_id}_kp3d.npy"
                    )
                    entry["motion_kp3d"] = motion_rel
                    stats["motion_ok"] += 1
                else:
                    stats["no_motion_key"] += 1

                results.append(entry)

    # ── 3. Save intermediate annotation ────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_json}")
    print(f"Total descriptions processed: {stats['total_descriptions']}")
    print(f"  No take mapping:  {stats['no_take_mapping']}")
    print(f"  Ego clips OK:     {stats['ego_ok']}")
    print(f"  No ego clip:      {stats['no_ego_clip']}")
    print(f"  Exo clips OK:     {stats['exo_ok']}")
    print(f"  No exo video:     {stats['no_exo_video']}")
    print(f"  Motion OK:        {stats['motion_ok']}")
    print(f"  No motion key:    {stats['no_motion_key']}")
    print(f"Total entries:      {len(results)}")


if __name__ == "__main__":
    main()
