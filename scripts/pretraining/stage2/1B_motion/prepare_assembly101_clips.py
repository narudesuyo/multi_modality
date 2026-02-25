#!/usr/bin/env python3
"""Prepare Assembly101 clips for Stage2 mixed training.

For each annotation in the Assembly101 CSV:
  1. Symlink video frames to frames_atomic/ with img prefix
  2. Convert per-frame SMPL-H params (axis-angle JSON) to 3D keypoints
  3. Resample to 41 frames and save as kp3d .npy [41, 154, 3]
  4. Generate intermediate annotation JSON

Requirements:
  - smplx library: pip install smplx
  - SMPL-H body model: SMPLH_NEUTRAL.npz (or .pkl)
    Download from https://smpl-x.is.tue.mpg.de/
    Place at: {model-path}/smplh/SMPLH_NEUTRAL.npz

Usage:
    python prepare_assembly101_clips.py
    python prepare_assembly101_clips.py --limit 10         # test with 10 samples
    python prepare_assembly101_clips.py --skip-frames       # motion only
    python prepare_assembly101_clips.py --skip-motion       # frames only
    python prepare_assembly101_clips.py --device cuda       # GPU acceleration
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
DATA_ROOT = os.environ.get("DATA_ROOT", "/work/narus/data")
ASSEMBLY_ROOT = os.path.join(DATA_ROOT, "Assembly101")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "Assembly101/processed/train")
MODEL_PATH = os.environ.get(
    "SMPLH_MODEL_PATH",
    "/home/narus/2026/EgoHand/BodyTokenize/models/smplx",
)

TARGET_MOTION_FRAMES = 41

# ──────────────────────────────────────────────────────────────────────────────
# CSV parsing


def parse_csv(csv_path):
    """Parse Assembly101 annotation CSV.

    Returns list of dicts with keys:
        video_path, session_id, camera_id, start_frame, end_frame, text
    """
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = row["video"]  # e.g. "session_id/C10118_rgb"
            parts = video_path.split("/")
            session_id = parts[0]
            camera_id = parts[1] if len(parts) > 1 else ""
            samples.append({
                "video_path": video_path,
                "session_id": session_id,
                "camera_id": camera_id,
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "text": row["text"],
            })
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# SMPL-H forward kinematics


def setup_smplh_model(model_path, device="cpu"):
    """Create SMPL-H body model for forward kinematics."""
    import smplx

    # smplx.create looks for {model_path}/SMPLH_NEUTRAL.npz (or .pkl)
    # If the model is in a subdirectory like models/smplh/, pass models/smplh
    model = smplx.create(
        model_path=model_path,
        model_type="smplh",
        gender="male",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)
    model.eval()
    return model


def load_frame_params(motion_dir, session_id, frame_num):
    """Load SMPL-H parameters from a single per-frame JSON."""
    json_path = os.path.join(motion_dir, session_id, f"{frame_num:06d}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def smplh_params_to_joints(model, params_list, device="cpu"):
    """Run SMPL-H forward kinematics on a batch of per-frame params.

    Args:
        model: smplx SMPLH model
        params_list: list of dicts, each with betas/transl/global_orient/body_pose/...
        device: torch device

    Returns:
        joints: np.ndarray [T, 52, 3]
    """
    T = len(params_list)

    betas = torch.zeros(T, 10, device=device)
    transl = torch.zeros(T, 3, device=device)
    global_orient = torch.zeros(T, 3, device=device)
    body_pose = torch.zeros(T, 63, device=device)       # 21 joints × 3
    left_hand_pose = torch.zeros(T, 45, device=device)  # 15 joints × 3
    right_hand_pose = torch.zeros(T, 45, device=device)

    for i, p in enumerate(params_list):
        b = np.array(p["betas"], dtype=np.float32).flatten()
        betas[i, :len(b)] = torch.from_numpy(b[:10])

        transl[i] = torch.tensor(p["transl"], dtype=torch.float32)
        global_orient[i] = torch.tensor(p["global_orient"], dtype=torch.float32)

        bp = np.array(p["body_pose"], dtype=np.float32).flatten()
        body_pose[i, :min(len(bp), 63)] = torch.from_numpy(bp[:63])

        lhp = np.array(p["left_hand_pose"], dtype=np.float32).flatten()
        left_hand_pose[i, :min(len(lhp), 45)] = torch.from_numpy(lhp[:45])

        rhp = np.array(p["right_hand_pose"], dtype=np.float32).flatten()
        right_hand_pose[i, :min(len(rhp), 45)] = torch.from_numpy(rhp[:45])

    with torch.no_grad():
        output = model(
            betas=betas,
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )

    # SMPL-H output.joints: [T, J, 3] where J >= 52
    # Take first 52: 22 body + 15 LH + 15 RH
    joints = output.joints[:, :52, :].cpu().numpy()
    return joints.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Joint format conversion


def joints52_to_kp3d154(joints_52):
    """Map SMPL-H 52 joints → 154-joint kp3d format (matching EgoExo4D).

    SMPL-H:       [0:22] body, [22:37] LH, [37:52] RH
    EgoExo4D 154: [0:22] body, [22:25] jaw/eyes(zeros), [25:40] LH, [40:55] RH, [55:154] zeros
    """
    T = joints_52.shape[0]
    kp3d = np.zeros((T, 154, 3), dtype=np.float32)
    kp3d[:, 0:22, :] = joints_52[:, 0:22, :]     # body
    kp3d[:, 25:40, :] = joints_52[:, 22:37, :]   # left hand
    kp3d[:, 40:55, :] = joints_52[:, 37:52, :]   # right hand
    return kp3d


# ──────────────────────────────────────────────────────────────────────────────
# Motion resampling


def resample_motion(kp3d, target_frames=TARGET_MOTION_FRAMES):
    """Resample motion to fixed frame count via linear interpolation."""
    T = kp3d.shape[0]
    if T == target_frames:
        return kp3d
    if T == 0:
        return None
    if T == 1:
        return np.repeat(kp3d, target_frames, axis=0)

    sample_pts = np.linspace(0, T - 1, target_frames)
    J, C = kp3d.shape[1], kp3d.shape[2]
    out = np.empty((target_frames, J, C), dtype=np.float32)
    for i, pt in enumerate(sample_pts):
        lo = int(pt)
        hi = min(lo + 1, T - 1)
        alpha = pt - lo
        out[i] = (1.0 - alpha) * kp3d[lo] + alpha * kp3d[hi]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Frame linking


def link_frames(src_video_dir, start_frame, end_frame, dst_dir, use_symlink=True):
    """Link/copy Assembly101 frames to training format.

    Source: {src_video_dir}/{frame:06d}.jpg  (0-indexed)
    Dest:   {dst_dir}/img{N:05d}.jpg         (1-indexed)

    Returns number of frames linked.
    """
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for i, frame_num in enumerate(range(start_frame, end_frame + 1)):
        src = os.path.join(src_video_dir, f"{frame_num:06d}.jpg")
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, f"img{i + 1:05d}.jpg")
        if os.path.exists(dst):
            os.remove(dst)
        if use_symlink:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        n += 1
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Main


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--assembly-root", default=ASSEMBLY_ROOT)
    parser.add_argument("--output-root", default=None,
                        help="Output root dir. Default: {DATA_ROOT}/Assembly101/processed/{split}")
    parser.add_argument("--model-path", default=MODEL_PATH,
                        help="Directory containing smplh/ with SMPLH_NEUTRAL.npz")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N samples (0 = all)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--copy-frames", action="store_true",
                        help="Copy frames instead of symlinking")
    parser.add_argument("--skip-frames", action="store_true",
                        help="Skip frame linking (motion only)")
    parser.add_argument("--skip-motion", action="store_true",
                        help="Skip motion conversion (frames only)")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for SMPL-H forward kinematics")
    args = parser.parse_args()

    if args.output_root is None:
        args.output_root = os.path.join(DATA_ROOT, "Assembly101/processed", args.split)

    here = Path(__file__).parent
    csv_path = os.path.join(args.assembly_root, "text/v1", f"{args.split}.csv")
    motion_root = os.path.join(args.assembly_root, "motion/v1")
    video_root = os.path.join(args.assembly_root, "video")

    if args.output_json is None:
        args.output_json = str(
            here / f"annotation_assembly101_{args.split}_intermediate.json"
        )

    # ── Parse CSV ────────────────────────────────────────────────────────────
    samples = parse_csv(csv_path)
    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"Split:         {args.split}")
    print(f"Total samples: {len(samples)}")
    print(f"Assembly root: {args.assembly_root}")
    print(f"Output root:   {args.output_root}")
    print(f"Model path:    {args.model_path}")
    print(f"Device:        {args.device}")
    print()

    # ── Setup SMPL-H model ───────────────────────────────────────────────────
    smplh_model = None
    if not args.skip_motion:
        print("Loading SMPL-H model...")
        smplh_model = setup_smplh_model(args.model_path, args.device)
        print("  SMPL-H model loaded.")

    # ── Group by session for efficient motion loading ────────────────────────
    session_samples = defaultdict(list)
    for i, s in enumerate(samples):
        session_samples[s["session_id"]].append((i, s))

    results = []
    stats = defaultdict(int)
    stats["total"] = len(samples)

    for session_id in tqdm(sorted(session_samples.keys()), desc="Sessions"):
        session_data = session_samples[session_id]

        # Pre-load all motion JSONs for this session (if needed)
        session_motion_cache = {}
        if not args.skip_motion:
            motion_session_dir = os.path.join(motion_root, session_id)
            if os.path.isdir(motion_session_dir):
                for fname in os.listdir(motion_session_dir):
                    if fname.endswith(".json"):
                        frame_num = int(fname.replace(".json", ""))
                        fpath = os.path.join(motion_session_dir, fname)
                        with open(fpath) as f:
                            session_motion_cache[frame_num] = json.load(f)
                print(f"  {session_id}: loaded {len(session_motion_cache)} motion frames")

        for sample_idx, sample in session_data:
            sample_id = f"{sample_idx:06d}"

            entry = {
                "session_id": session_id,
                "sample_id": sample_id,
                "caption": sample["text"],
                "start_frame": sample["start_frame"],
                "end_frame": sample["end_frame"],
            }

            # ── Frame linking ────────────────────────────────────────────
            frame_ok = False
            if not args.skip_frames:
                dst_frames_dir = os.path.join(
                    args.output_root, "frames_atomic", session_id, sample_id,
                )
                frames_rel = os.path.join("frames_atomic", session_id, sample_id)

                if args.skip_existing and os.path.isdir(dst_frames_dir):
                    n = len([f for f in os.listdir(dst_frames_dir)
                             if f.startswith("img")])
                    if n > 0:
                        frame_ok = True
                        entry["video"] = frames_rel
                        stats["frame_skip"] += 1

                if not frame_ok:
                    src_dir = os.path.join(video_root, sample["video_path"])
                    n_frames = link_frames(
                        src_dir,
                        sample["start_frame"],
                        sample["end_frame"],
                        dst_frames_dir,
                        use_symlink=not args.copy_frames,
                    )
                    if n_frames > 0:
                        frame_ok = True
                        entry["video"] = frames_rel

                if frame_ok:
                    stats["frame_ok"] += 1
                else:
                    stats["frame_fail"] += 1

            # ── Motion conversion ────────────────────────────────────────
            motion_ok = False
            if not args.skip_motion:
                motion_out = os.path.join(
                    args.output_root, "motion_atomic", session_id,
                    f"{sample_id}_kp3d.npy",
                )
                motion_rel = os.path.join(
                    "motion_atomic", session_id, f"{sample_id}_kp3d.npy",
                )

                if args.skip_existing and os.path.exists(motion_out):
                    motion_ok = True
                    entry["motion_kp3d"] = motion_rel
                    stats["motion_skip"] += 1

                if not motion_ok:
                    # Collect per-frame params from cache
                    params_list = []
                    for fn in range(sample["start_frame"], sample["end_frame"] + 1):
                        p = session_motion_cache.get(fn)
                        if p is not None:
                            params_list.append(p)

                    if len(params_list) >= 2:
                        # FK in batches
                        all_joints = []
                        for bs in range(0, len(params_list), args.batch_size):
                            batch = params_list[bs : bs + args.batch_size]
                            joints = smplh_params_to_joints(
                                smplh_model, batch, args.device,
                            )
                            all_joints.append(joints)
                        joints_52 = np.concatenate(all_joints, axis=0)

                        # Map to 154-joint format
                        kp3d_154 = joints52_to_kp3d154(joints_52)

                        # Resample to 41 frames
                        kp3d = resample_motion(kp3d_154, TARGET_MOTION_FRAMES)

                        if kp3d is not None:
                            os.makedirs(os.path.dirname(motion_out), exist_ok=True)
                            np.save(motion_out, kp3d)
                            motion_ok = True
                            entry["motion_kp3d"] = motion_rel

                if motion_ok:
                    stats["motion_ok"] += 1
                else:
                    stats["motion_fail"] += 1

            entry["_frame_ok"] = frame_ok
            entry["_motion_ok"] = motion_ok
            results.append(entry)

    # ── Save intermediate JSON ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Intermediate JSON: {args.output_json}  ({len(results)} entries)")
    print(f"Total:       {stats['total']}")
    print(f"Frames OK:   {stats['frame_ok']}  (skip: {stats['frame_skip']})")
    print(f"Frames fail: {stats['frame_fail']}")
    print(f"Motion OK:   {stats['motion_ok']}  (skip: {stats['motion_skip']})")
    print(f"Motion fail: {stats['motion_fail']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
