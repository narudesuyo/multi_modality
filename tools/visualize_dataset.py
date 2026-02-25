"""Visualize paired samples from the Stage2 Motion dataset.

For each sample, saves:
  - video.mp4:    video clip (trimmed to the annotated segment)
  - kp3d.mp4:     3D skeleton animation from original kp3d
  - motion.mp4:   3D skeleton animation decoded from motion tokens (VQ-VAE recon)
  - caption.txt:  text caption
  - combined.mp4: video + kp3d + recon side-by-side with caption overlay

Usage:
    python tools/visualize_dataset.py \
        --ann scripts/pretraining/stage2/1B_motion/annotation_atomic_train.json \
        --data-root /work/narus/data/train/takes_clipped/egoexo \
        --output-dir ./vis_output \
        --num-samples 10
"""

import sys
import os
import argparse
import json

_BODY_TOKENIZE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../BodyTokenize")
)
sys.path.insert(0, _BODY_TOKENIZE_ROOT)

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from src.train.utils import build_model_from_args
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric
from src.evaluate.vis import visualize_two_motions



# ---------------------------------------------------------------------------
# Save video clip
# ---------------------------------------------------------------------------
def save_video_clip(video_path, save_path):
    """Copy the source video as-is to *save_path* using ffmpeg (fast, no re-encode)."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # fallback: just copy the file
        shutil.copy2(video_path, save_path)
        return

    subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-c", "copy", save_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Combined visualization (video + motion side-by-side + caption)
# ---------------------------------------------------------------------------
def _read_all_frames(video_path):
    """Read all frames from a video file. Returns list of BGR numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _wrap_text(text, max_chars=60):
    """Wrap text into multiple lines."""
    words = text.split()
    lines, line = [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > max_chars:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        lines.append(line)
    return lines


def save_combined_video(video_path, motion_paths, labels, caption, save_path, height=360):
    """Create side-by-side video (source | mot1 | mot2 ...) with labels and caption.

    Args:
        video_path: path to source video
        motion_paths: list of motion video paths to place right of the source
        labels: list of label strings for each panel (len = 1 + len(motion_paths))
        caption: text caption for bottom overlay
        save_path: output path
        height: resize height for all panels
    """
    vid_frames = _read_all_frames(video_path)
    if not vid_frames:
        return

    mot_frames_list = []
    for mp in motion_paths:
        mf = _read_all_frames(mp)
        if not mf:
            return
        mot_frames_list.append(mf)

    # Get source video fps for output
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    n_frames = len(vid_frames)

    def sample_frames(frames, n):
        indices = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in indices]

    mot_frames_list = [sample_frames(mf, n_frames) for mf in mot_frames_list]

    # Resize all to same height
    def resize_h(frame, h):
        oh, ow = frame.shape[:2]
        w = int(ow * h / oh)
        return cv2.resize(frame, (w, h))

    vid_frames = [resize_h(f, height) for f in vid_frames]
    mot_frames_list = [[resize_h(f, height) for f in mf] for mf in mot_frames_list]

    # Canvas width
    panel_widths = [vid_frames[0].shape[1]] + [mf[0].shape[1] for mf in mot_frames_list]
    total_w = sum(panel_widths)

    # Label + text area
    label_h = 30
    lines = _wrap_text(caption, max_chars=90)
    line_h = 28
    text_h = line_h * len(lines) + 20
    total_h = label_h + height + text_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, src_fps, (total_w, total_h))

    all_panels = [vid_frames] + mot_frames_list

    for frame_idx in range(n_frames):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        # Draw labels
        x_off = 0
        for pi, pw in enumerate(panel_widths):
            lbl = labels[pi] if pi < len(labels) else ""
            cv2.putText(canvas, lbl, (x_off + 8, label_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            x_off += pw
        # Place panels
        x_off = 0
        for pi, pw in enumerate(panel_widths):
            canvas[label_h:label_h + height, x_off:x_off + pw] = all_panels[pi][frame_idx]
            x_off += pw
        # Draw caption text
        for j, line in enumerate(lines):
            y = label_h + height + 20 + j * line_h
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
        writer.write(canvas)

    writer.release()


# ---------------------------------------------------------------------------
# kp3d -> 62 joints (same selection as inference_atomic.py)
# ---------------------------------------------------------------------------
def kp3d_to_joints(npy_path, include_fingertips=True):
    """Load raw kp3d (T, 154, 3) and select 62 (or 52) joints."""
    kp = np.load(npy_path).astype(np.float32)  # (T, 154, 3)
    if include_fingertips:
        joints = np.concatenate([kp[:, :22, :], kp[:, 25:55, :], kp[:, -10:, :]], axis=1)
    else:
        joints = np.concatenate([kp[:, :22, :], kp[:, 25:55, :]], axis=1)
    return joints  # (T, 62, 3) or (T, 52, 3)


def tok_pose_to_kp3d_path(tok_pose_rel):
    """Derive kp3d .npy path from tok_pose .npz relative path."""
    # tok_pose_atomic_40/{take}/{id}_0000.npz -> motion_atomic/{take}/{id}_kp3d.npy
    import re
    p = tok_pose_rel.replace("tok_pose_atomic_40/", "motion_atomic/")
    p = re.sub(r"_\d{4}\.npz$", "_kp3d.npy", p)
    return p


# ---------------------------------------------------------------------------
# VQ-VAE model loading
# ---------------------------------------------------------------------------
def load_vqvae(config_path, ckpt_path, device):
    cfg = OmegaConf.load(config_path)
    model = build_model_from_args(cfg, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_norm_stats(cfg, device):
    if not cfg.get("normalize", False):
        return None, None
    mean_path = cfg.get("mean_path", "./preprocess/statistics/tips/mean.npy")
    std_path = cfg.get("std_path", "./preprocess/statistics/tips/std.npy")
    if not os.path.isabs(mean_path):
        mean_path = os.path.join(_BODY_TOKENIZE_ROOT, mean_path)
    if not os.path.isabs(std_path):
        std_path = os.path.join(_BODY_TOKENIZE_ROOT, std_path)
    mean_full = torch.from_numpy(np.load(mean_path)).float().to(device)
    std_full = torch.from_numpy(np.load(std_path)).float().to(device)
    return mean_full, std_full


# ---------------------------------------------------------------------------
# Motion decode  (follows inference_atomic.py:384-416)
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_motion_to_joints(npz_path, model, cfg, mean_full, std_full, device):
    data = np.load(npz_path)
    idx = data["idx"]  # (1, T', 8) or (T', 8)
    if idx.ndim == 2:
        idx = idx[np.newaxis, ...]

    body_tok = cfg.get("body_tokens_per_t", 4)
    idxB = torch.from_numpy(idx[..., :body_tok]).long().to(device)
    idxH = torch.from_numpy(idx[..., body_tok:]).long().to(device)

    pr_n = model.decode_from_ids(idxH=idxH, idxB=idxB)

    body_dim = cfg.get("body_in_dim", 263)
    if mean_full is not None:
        mean_B, std_B = mean_full[:body_dim], std_full[:body_dim]
        mean_H, std_H = mean_full[body_dim:], std_full[body_dim:]
        pr_body = pr_n[..., :body_dim] * (std_B + 1e-8) + mean_B
        pr_hand = pr_n[..., body_dim:] * (std_H + 1e-8) + mean_H
    else:
        pr_body = pr_n[..., :body_dim]
        pr_hand = pr_n[..., body_dim:]

    include_ft = cfg.get("include_fingertips", True)
    pr_full = reconstruct_623_from_body_hand(pr_body, pr_hand, include_ft)

    joints_num = 62 if include_ft else 52
    joints = recover_from_ric(
        pr_full, joints_num,
        use_root_loss=cfg.get("use_root_loss", False),
        base_idx=cfg.get("base_idx", 15),
        hand_local=cfg.get("hand_local", True),
    )[0]  # (T, J, 3)

    return joints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", required=True, help="Annotation JSON path")
    p.add_argument("--data-root", required=True, help="Root dir for video files")
    p.add_argument("--motion-data-root", default=None,
                   help="Root dir for motion NPZ files (default: same as --data-root)")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--vqvae-config", default=os.path.join(_BODY_TOKENIZE_ROOT, "ckpt_vq/config.yaml"))
    p.add_argument("--vqvae-ckpt", default=os.path.join(_BODY_TOKENIZE_ROOT, "ckpt_vq/ckpt_best.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--motion-fps", type=int, default=10)
    p.add_argument("--view", default="all", choices=["all", "body", "hands", "lh", "rh"])
    return p.parse_args()


def main():
    args = parse_args()
    motion_data_root = args.motion_data_root or args.data_root

    with open(args.ann, "r") as f:
        annos = json.load(f)
    print(f"Loaded {len(annos)} annotations from {args.ann}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_vqvae(args.vqvae_config, args.vqvae_ckpt, device)
    mean_full, std_full = load_norm_stats(cfg, device)
    include_ft = cfg.get("include_fingertips", True)
    base_idx = cfg.get("base_idx", 15)

    n = min(args.num_samples, len(annos))

    for i in range(n):
        ann = annos[i]
        out_dir = os.path.join(args.output_dir, f"{i:04d}")
        os.makedirs(out_dir, exist_ok=True)

        caption = ann.get("caption", "")
        print(f"[{i+1}/{n}] {caption[:70]}")

        # --- Video clip ---
        video_path = os.path.join(args.data_root, ann["video"])
        try:
            save_video_clip(video_path, os.path.join(out_dir, "video.mp4"))
        except Exception as e:
            print(f"  [WARN] Video: {e}")

        # --- Original kp3d skeleton ---
        kp3d_rel = tok_pose_to_kp3d_path(ann["tok_pose"])
        kp3d_path = os.path.join(motion_data_root, kp3d_rel)
        try:
            kp3d_joints = kp3d_to_joints(kp3d_path, include_ft)
            visualize_two_motions(
                j_gt=kp3d_joints,
                j_pr=kp3d_joints,
                save_path=os.path.join(out_dir, "kp3d.mp4"),
                fps=args.motion_fps,
                view=args.view,
                rotate=False,
                include_fingertips=include_ft,
                only_gt=True,
                origin_align=True,
                base_idx=base_idx,
            )
        except Exception as e:
            print(f"  [WARN] kp3d: {e}")

        # --- VQ-VAE reconstructed motion ---
        npz_path = os.path.join(motion_data_root, ann["tok_pose"])
        try:
            joints = decode_motion_to_joints(
                npz_path, model, cfg, mean_full, std_full, device
            )
            visualize_two_motions(
                j_gt=joints,
                j_pr=joints,
                save_path=os.path.join(out_dir, "motion.mp4"),
                fps=args.motion_fps,
                view=args.view,
                rotate=False,
                include_fingertips=include_ft,
                only_gt=True,
                origin_align=True,
                base_idx=base_idx,
            )
        except Exception as e:
            print(f"  [WARN] Motion: {e}")

        # --- Caption ---
        with open(os.path.join(out_dir, "caption.txt"), "w") as f:
            f.write(caption + "\n")

        # --- Combined (video + kp3d + recon + caption) ---
        video_out = os.path.join(out_dir, "video.mp4")
        kp3d_out = os.path.join(out_dir, "kp3d.mp4")
        motion_out = os.path.join(out_dir, "motion.mp4")
        motion_paths = [p for p in [kp3d_out, motion_out] if os.path.exists(p)]
        if os.path.exists(video_out) and motion_paths:
            try:
                labels = ["Video"]
                if os.path.exists(kp3d_out):
                    labels.append("kp3d (GT)")
                if os.path.exists(motion_out):
                    labels.append("VQ-VAE recon")
                save_combined_video(
                    video_out, motion_paths, labels, caption,
                    os.path.join(out_dir, "combined.mp4"),
                )
            except Exception as e:
                print(f"  [WARN] Combined: {e}")

    print(f"\nDone. {n} samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
