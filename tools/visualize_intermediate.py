"""Visualize intermediate data from prepare_atomic_clips / prepare_assembly101_clips.

For each sample, saves a combined MP4 with:
  - Top row:    8 uniformly sampled video frames (matching training sampling)
  - Bottom row: 3D skeleton animation from kp3d (41 frames, 62 joints)
  - Caption overlay at the bottom

Usage:
    # EgoExo4D
    python tools/visualize_intermediate.py \
        --ann scripts/pretraining/stage2/1B_motion/annotation_atomic_train_intermediate.json \
        --data-root /work/narus/data/EgoExo4D/processed/train \
        --output-dir ./vis_egoexo4d \
        --num-samples 5

    # Assembly101
    python tools/visualize_intermediate.py \
        --ann scripts/pretraining/stage2/1B_motion/annotation_assembly101_train_intermediate.json \
        --data-root /work/narus/data/Assembly101/processed/train \
        --output-dir ./vis_assembly101 \
        --num-samples 5
"""

import argparse
import json
import os
import random

import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore", message="Tight layout not applied")


# ---------------------------------------------------------------------------
# Skeleton definition (52 joints + 10 fingertips = 62 joints)
# ---------------------------------------------------------------------------
# Body 0-21, left hand 22-36, right hand 37-51, fingertips 52-61
EDGES_52 = [
    # torso / spine
    (0, 1), (0, 2),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # legs
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    # arms
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    # left hand (wrist=20)
    (20, 22), (22, 23), (23, 24),   # index
    (20, 25), (25, 26), (26, 27),   # middle
    (20, 28), (28, 29), (29, 30),   # pinky
    (20, 31), (31, 32), (32, 33),   # ring
    (20, 34), (34, 35), (35, 36),   # thumb
    # right hand (wrist=21)
    (21, 37), (37, 38), (38, 39),   # index
    (21, 40), (40, 41), (41, 42),   # middle
    (21, 43), (43, 44), (44, 45),   # pinky
    (21, 46), (46, 47), (47, 48),   # ring
    (21, 49), (49, 50), (50, 51),   # thumb
]

# Fingertip edges (tips = joints 52-61)
EDGES_TIPS = [
    (36, 52),   # left thumb tip
    (24, 53),   # left index tip
    (27, 54),   # left middle tip
    (33, 55),   # left ring tip
    (30, 56),   # left pinky tip
    (51, 57),   # right thumb tip
    (39, 58),   # right index tip
    (42, 59),   # right middle tip
    (48, 60),   # right ring tip
    (45, 61),   # right pinky tip
]

# Colors for different body parts
COLOR_BODY = "#4477AA"
COLOR_LEFT_HAND = "#EE6677"
COLOR_RIGHT_HAND = "#228833"


def _get_edges(n_joints):
    """Return edge list appropriate for the joint count."""
    if n_joints >= 62:
        return EDGES_52 + EDGES_TIPS
    return EDGES_52


def _tip_color(idx):
    """Return hand color for a fingertip index."""
    # 52-56: left tips, 57-61: right tips
    return COLOR_LEFT_HAND if 52 <= idx <= 56 else COLOR_RIGHT_HAND


def _edge_color(i, j):
    """Return color based on which body part the edge belongs to."""
    if i >= 52 or j >= 52:
        tip_idx = i if i >= 52 else j
        return _tip_color(tip_idx)
    if (i >= 37 or j >= 37):
        return COLOR_RIGHT_HAND
    if (i >= 22 or j >= 22):
        return COLOR_LEFT_HAND
    return COLOR_BODY


def _filter_edges(edges, keep_idx):
    keep = set(keep_idx)
    return [(i, j) for (i, j) in edges if i in keep and j in keep]


def _hand_joint_indices(n_joints):
    left_hand = list(range(22, 37))
    right_hand = list(range(37, 52))
    wrists = [20, 21]
    tips = list(range(52, min(n_joints, 62)))
    return wrists + left_hand + right_hand + tips


# ---------------------------------------------------------------------------
# kp3d loading (same as visualize_dataset.py)
# ---------------------------------------------------------------------------
def kp3d_to_joints(npy_path, include_fingertips=True, auto_drop_fingertips=True):
    """Load raw kp3d (T, 154, 3) and select 62 (or 52) joints.

    Assembly101 kp3d is 154-joint with zero-padded fingertips. If tips are all
    zeros, drop them to avoid collapsed tip clusters in visualization.
    """
    kp = np.load(npy_path).astype(np.float32)  # (T, 154, 3)
    use_tips = include_fingertips
    if include_fingertips and auto_drop_fingertips:
        tips = kp[:, -10:, :]
        if np.allclose(tips, 0.0, atol=1e-6):
            use_tips = False
    if use_tips:
        joints = np.concatenate([kp[:, :22, :], kp[:, 25:55, :], kp[:, -10:, :]], axis=1)
    else:
        joints = np.concatenate([kp[:, :22, :], kp[:, 25:55, :]], axis=1)
    return joints  # (T, 62, 3) or (T, 52, 3)


# ---------------------------------------------------------------------------
# Frame reading (from image directory)
# ---------------------------------------------------------------------------
def read_all_frames_from_dir(frame_dir):
    """Read all images from a frame directory. Returns list of BGR numpy arrays."""
    img_names = sorted([f for f in os.listdir(frame_dir) if f.startswith("img")])
    if not img_names:
        img_names = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
    if not img_names:
        raise FileNotFoundError(f"No image files found in {frame_dir}")

    frames = []
    for name in img_names:
        img = cv2.imread(os.path.join(frame_dir, name))
        if img is not None:
            frames.append(img)
    return frames


def read_frames_from_dir(frame_dir, num_frames=8):
    """Read images from a frame directory and uniformly sample num_frames.

    Returns list of BGR numpy arrays and the selected indices.
    """
    img_names = sorted([f for f in os.listdir(frame_dir) if f.startswith("img")])
    if not img_names:
        # fallback: try all image files
        img_names = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
    if not img_names:
        raise FileNotFoundError(f"No image files found in {frame_dir}")

    vlen = len(img_names)
    # Uniform sampling (middle of each interval)
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(0, vlen, acc_samples + 1).astype(int)
    indices = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(acc_samples)]

    # Pad if fewer frames than requested
    if len(indices) < num_frames:
        indices = indices + [indices[-1]] * (num_frames - len(indices))

    frames = []
    for idx in indices:
        path = os.path.join(frame_dir, img_names[idx])
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Failed to read {path}")
        frames.append(img)

    return frames, indices


# ---------------------------------------------------------------------------
# Text wrapping (same as visualize_dataset.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3D skeleton rendering
# ---------------------------------------------------------------------------
def _axis_order_from_up(up_axis):
    """Return axis order (x_idx, y_idx, z_idx) for plotting, where z is up."""
    if up_axis == "x":
        return (1, 2, 0)  # plot y,z as x,y; x as z (up)
    if up_axis == "z":
        return (0, 1, 2)  # plot x,y as x,y; z as z (up)
    # default: y-up
    return (0, 2, 1)      # plot x,z as x,y; y as z (up)


def render_skeleton_frame(joints, frame_idx, edges, fig_size=(4, 4), dpi=80,
                          elev=15, azim=-60, x_range=None, y_range=None, z_range=None,
                          axis_order=(0, 2, 1), show_axes=False, axis_labels=("X", "Y", "Z")):
    """Render a single skeleton frame to a BGR numpy array.

    joints: (T, J, 3) array.
    axis_order defines which data axes map to (x, y, z) for plotting.
    """
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    j = joints[frame_idx]  # (J, 3)
    xs = j[:, axis_order[0]]
    ys = j[:, axis_order[1]]
    zs = j[:, axis_order[2]]

    # Draw edges
    for i_joint, j_joint in edges:
        if i_joint >= len(j) or j_joint >= len(j):
            continue
        color = _edge_color(i_joint, j_joint)
        ax.plot([xs[i_joint], xs[j_joint]],
                [ys[i_joint], ys[j_joint]],
                [zs[i_joint], zs[j_joint]],
                c=color, linewidth=1.5)

    # Draw joints
    ax.scatter(xs[:22], ys[:22], zs[:22], c=COLOR_BODY, s=8, depthshade=True)
    if len(j) > 22:
        ax.scatter(xs[22:37], ys[22:37], zs[22:37], c=COLOR_LEFT_HAND, s=4, depthshade=True)
        ax.scatter(xs[37:52], ys[37:52], zs[37:52], c=COLOR_RIGHT_HAND, s=4, depthshade=True)
    if len(j) > 52:
        left_tip = slice(52, 57)
        right_tip = slice(57, 62)
        ax.scatter(xs[left_tip], ys[left_tip], zs[left_tip], c=COLOR_LEFT_HAND, s=4, depthshade=True)
        ax.scatter(xs[right_tip], ys[right_tip], zs[right_tip], c=COLOR_RIGHT_HAND, s=4, depthshade=True)

    ax.view_init(elev=elev, azim=azim)

    if x_range is not None:
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(*z_range)

    if show_axes:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

    try:
        fig.tight_layout(pad=0)
    except Exception:
        pass
    fig.canvas.draw()

    # Convert to numpy BGR
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3]  # drop alpha
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img


def compute_axis_ranges(joints, margin=0.3, axis_order=(0, 2, 1), zoom=1.0, joint_idx=None):
    """Compute consistent axis ranges across all frames."""
    if joint_idx is not None:
        sel = joints[:, joint_idx, :]
    else:
        sel = joints
    xs = sel[..., axis_order[0]]
    ys = sel[..., axis_order[1]]
    zs = sel[..., axis_order[2]]

    def axis_range(vals):
        lo, hi = vals.min(), vals.max()
        mid = (lo + hi) / 2
        span = max(hi - lo, 0.1) / 2 + margin
        span = span / max(zoom, 1e-6)
        return (mid - span, mid + span)

    return axis_range(xs), axis_range(ys), axis_range(zs)


def render_skeleton_sequence(joints, edges, fig_size=(6, 6), dpi=100, axis_order=(0, 2, 1), zoom=1.0, joint_idx=None,
                             show_axes=False, axis_labels=("X", "Y", "Z")):
    """Render all skeleton frames. Returns list of BGR images."""
    x_range, y_range, z_range = compute_axis_ranges(joints, axis_order=axis_order, zoom=zoom, joint_idx=joint_idx)
    frames = []
    for t in range(len(joints)):
        img = render_skeleton_frame(
            joints, t, edges, fig_size=fig_size, dpi=dpi,
            x_range=x_range, y_range=y_range, z_range=z_range,
            axis_order=axis_order,
            show_axes=show_axes,
            axis_labels=axis_labels,
        )
        frames.append(img)
    return frames


def _detect_up_axis_from_frame(joints, frame_idx=0):
    """Heuristically detect up axis from joint spread in a single frame."""
    j = joints[frame_idx]
    ranges = j.max(axis=0) - j.min(axis=0)
    up_idx = int(np.argmax(ranges))
    return ["x", "y", "z"][up_idx]


# ---------------------------------------------------------------------------
# Detect dataset type from annotation entry
# ---------------------------------------------------------------------------
def get_video_field(ann):
    """Return the frame-directory relative path from the annotation entry."""
    if "video_exo" in ann:
        return ann["video_exo"]      # EgoExo4D
    elif "video" in ann:
        return ann["video"]          # Assembly101
    else:
        raise KeyError(f"No video/video_exo field in annotation: {list(ann.keys())}")


# ---------------------------------------------------------------------------
# Combined visualization
# ---------------------------------------------------------------------------
def make_combined_video(all_video_frames, skeleton_frames, caption, save_path,
                        fps=10, frame_height=360):
    """Create a combined MP4: left = video animation, right = skeleton animation.

    all_video_frames: list of ALL BGR images from the frame directory
    skeleton_frames: list of BGR skeleton images (one per kp3d frame)
    caption: text string
    """
    n_skel = len(skeleton_frames)

    # --- Left panel: video animation resampled to match skeleton frame count ---
    # Uniformly sample from all video frames to match skeleton frame count
    vid_indices = np.linspace(0, len(all_video_frames) - 1, n_skel).astype(int)
    sampled_vframes = [all_video_frames[i] for i in vid_indices]

    # Resize video frames to target height
    resized_vframes = []
    for f in sampled_vframes:
        h, w = f.shape[:2]
        new_w = int(w * frame_height / h)
        resized_vframes.append(cv2.resize(f, (new_w, frame_height)))
    vid_h = frame_height
    vid_w = resized_vframes[0].shape[1]

    # --- Right panel: skeleton animation ---
    skel_h_orig = skeleton_frames[0].shape[0]
    skel_w_orig = skeleton_frames[0].shape[1]
    skel_scale = vid_h / skel_h_orig
    new_skel_h = vid_h
    new_skel_w = int(skel_w_orig * skel_scale)

    resized_skel = [cv2.resize(f, (new_skel_w, new_skel_h)) for f in skeleton_frames]

    # --- Caption area ---
    canvas_w = vid_w + new_skel_w
    lines = _wrap_text(caption, max_chars=int(canvas_w / 12))
    line_h = 28
    caption_h = line_h * len(lines) + 20

    # --- Label area ---
    label_h = 30

    canvas_h = label_h + vid_h + caption_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (canvas_w, canvas_h))

    for t in range(n_skel):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Labels
        cv2.putText(canvas, f"Video (frame {t+1}/{n_skel})", (8, label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"kp3d (frame {t+1}/{n_skel})",
                    (vid_w + 8, label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Place video frame (left)
        canvas[label_h:label_h + vid_h, :vid_w] = resized_vframes[t]

        # Place skeleton (right)
        canvas[label_h:label_h + new_skel_h, vid_w:vid_w + new_skel_w] = resized_skel[t]

        # Caption
        y_off = label_h + vid_h
        for j, line in enumerate(lines):
            y = y_off + 20 + j * line_h
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)

        writer.write(canvas)

    writer.release()


def make_combined_video_multi(all_video_frames, skel_frames_list, labels, caption, save_path,
                              fps=10, frame_height=360):
    """Create combined MP4: left = video, right = multiple skeleton panels."""
    n_skel = len(skel_frames_list[0])
    vid_indices = np.linspace(0, len(all_video_frames) - 1, n_skel).astype(int)
    sampled_vframes = [all_video_frames[i] for i in vid_indices]

    resized_vframes = []
    for f in sampled_vframes:
        h, w = f.shape[:2]
        new_w = int(w * frame_height / h)
        resized_vframes.append(cv2.resize(f, (new_w, frame_height)))
    vid_h = frame_height
    vid_w = resized_vframes[0].shape[1]

    # Resize all skeleton panels to match height
    resized_skel_list = []
    skel_widths = []
    for skel_frames in skel_frames_list:
        skel_h_orig = skel_frames[0].shape[0]
        skel_w_orig = skel_frames[0].shape[1]
        skel_scale = vid_h / skel_h_orig
        new_skel_h = vid_h
        new_skel_w = int(skel_w_orig * skel_scale)
        skel_widths.append(new_skel_w)
        resized_skel_list.append([cv2.resize(f, (new_skel_w, new_skel_h)) for f in skel_frames])

    canvas_w = vid_w + sum(skel_widths)
    lines = _wrap_text(caption, max_chars=int(canvas_w / 12))
    line_h = 28
    caption_h = line_h * len(lines) + 20
    label_h = 30
    canvas_h = label_h + vid_h + caption_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (canvas_w, canvas_h))

    for t in range(n_skel):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Labels
        x_off = 0
        all_labels = ["Video"] + labels
        panel_widths = [vid_w] + skel_widths
        for lbl, pw in zip(all_labels, panel_widths):
            cv2.putText(canvas, f"{lbl} (frame {t+1}/{n_skel})",
                        (x_off + 8, label_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            x_off += pw

        # Place panels
        x_off = 0
        canvas[label_h:label_h + vid_h, x_off:x_off + vid_w] = resized_vframes[t]
        x_off += vid_w
        for pi, skel_frames in enumerate(resized_skel_list):
            pw = skel_widths[pi]
            canvas[label_h:label_h + vid_h, x_off:x_off + pw] = skel_frames[t]
            x_off += pw

        # Caption
        y_off = label_h + vid_h
        for j, line in enumerate(lines):
            y = y_off + 20 + j * line_h
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)

        writer.write(canvas)

    writer.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ann", required=True, help="Intermediate annotation JSON path")
    p.add_argument("--data-root", required=True,
                   help="Root dir where frame dirs and kp3d files live")
    p.add_argument("--output-dir", required=True, help="Output directory for MP4s")
    p.add_argument("--num-samples", type=int, default=5,
                   help="Number of samples to visualize")
    p.add_argument("--skeleton-fps", type=int, default=10,
                   help="FPS for skeleton animation in output video")
    p.add_argument("--frame-height", type=int, default=360,
                   help="Height of each video frame in the output")
    p.add_argument("--skip-bad", action="store_true",
                   help="Skip samples with _frame_ok=false or _motion_ok=false or _exo_ok=false")
    p.add_argument("--diverse", action="store_true",
                   help="Sample from different takes/sessions for diversity")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--up-axis", default="auto",
                   choices=["auto", "x", "y", "z"],
                   help="Up axis for kp3d. Use auto to infer per-sample.")
    p.add_argument("--no-axis-convert", action="store_true",
                   help="Plot kp3d as-is without axis reordering.")
    p.add_argument("--skel-zoom", type=float, default=1.0,
                   help="Zoom factor for skeleton view. >1.0 makes motion appear larger.")
    p.add_argument("--hands-view", action="store_true",
                   help="Also render a hands-only video (both hands).")
    p.add_argument("--hands-zoom", type=float, default=2.0,
                   help="Zoom factor for hands-only view.")
    p.add_argument("--show-axes", action="store_true",
                   help="Show axis labels in skeleton plots.")
    p.add_argument("--filter-take", type=str, default=None,
                   help="Only visualize samples from this take_name/session_id.")
    p.add_argument("--filter-sample", type=str, default=None,
                   help="Only visualize samples with this sample_id.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.ann, "r") as f:
        annos = json.load(f)
    print(f"Loaded {len(annos)} annotations from {args.ann}")

    if args.skip_bad:
        original_len = len(annos)
        annos = [
            a for a in annos
            if a.get("_frame_ok", True)
            and a.get("_motion_ok", True)
            and a.get("_exo_ok", True)
        ]
        print(f"After filtering bad samples: {len(annos)} / {original_len}")

    if args.filter_take is not None:
        annos = [a for a in annos if a.get("take_name", a.get("session_id")) == args.filter_take]
        print(f"After filter-take: {len(annos)}")
    if args.filter_sample is not None:
        annos = [a for a in annos if a.get("sample_id") == args.filter_sample]
        print(f"After filter-sample: {len(annos)}")

    if args.diverse:
        # Pick one sample per unique take/session for diversity
        random.seed(args.seed)
        by_source = {}
        for a in annos:
            key = a.get("take_name", a.get("session_id", "unknown"))
            by_source.setdefault(key, []).append(a)
        keys = list(by_source.keys())
        random.shuffle(keys)
        selected = []
        for k in keys:
            selected.append(random.choice(by_source[k]))
            if len(selected) >= args.num_samples:
                break
        annos = selected
        print(f"Diverse sampling: {len(annos)} from {len(by_source)} unique sources")

    n = min(args.num_samples, len(annos))
    if n == 0:
        print("No samples to visualize.")
        return

    for i in range(n):
        ann = annos[i]
        caption = ann.get("caption", "")
        sample_label = ann.get("sample_id", f"{i:04d}")
        take_or_session = ann.get("take_name", ann.get("session_id", "unknown"))
        out_name = f"{i:04d}_{take_or_session}_{sample_label}"
        out_path = os.path.join(args.output_dir, f"{out_name}.mp4")

        print(f"[{i+1}/{n}] {take_or_session}/{sample_label}: {caption[:70]}")

        # --- Video frames (read all) ---
        try:
            video_rel = get_video_field(ann)
            frame_dir = os.path.join(args.data_root, video_rel)
            all_video_frames = read_all_frames_from_dir(frame_dir)
            print(f"  Frames: {len(all_video_frames)} total from {frame_dir}")
        except Exception as e:
            print(f"  [SKIP] Video frames: {e}")
            continue

        # --- kp3d skeleton ---
        try:
            kp3d_path = os.path.join(args.data_root, ann["motion_kp3d"])
            joints = kp3d_to_joints(kp3d_path, include_fingertips=True)
            n_joints = joints.shape[1]
            edges = _get_edges(n_joints)

            if args.no_axis_convert:
                # Plot raw xyz without axis conversion
                for ax in (0, 1, 2):
                    joints[..., ax] -= joints[0, 15, ax]
                print(f"  kp3d: {joints.shape} (T={joints.shape[0]}, J={n_joints}), up=raw")
                axis_order = (0, 1, 2)
            else:
                # Determine up axis (per sample if auto)
                up_axis = args.up_axis
                if up_axis == "auto":
                    up_axis = _detect_up_axis_from_frame(joints, frame_idx=0)

                # Origin-align: center on head (joint 15) in horizontal plane
                up_idx = {"x": 0, "y": 1, "z": 2}[up_axis]
                horiz_axes = [0, 1, 2]
                horiz_axes.remove(up_idx)
                for ax in horiz_axes:
                    joints[..., ax] -= joints[0, 15, ax]

                print(f"  kp3d: {joints.shape} (T={joints.shape[0]}, J={n_joints}), up={up_axis}")
                axis_order = _axis_order_from_up(up_axis)

            skeleton_frames = render_skeleton_sequence(
                joints, edges, axis_order=axis_order, zoom=args.skel_zoom,
                show_axes=args.show_axes,
            )
        except Exception as e:
            print(f"  [SKIP] kp3d: {e}")
            continue

        # --- Combined video ---
        try:
            if args.hands_view:
                hand_idx = _hand_joint_indices(n_joints)
                hand_edges = _filter_edges(edges, hand_idx)
                hand_frames = render_skeleton_sequence(
                    joints, hand_edges, axis_order=axis_order, zoom=args.hands_zoom, joint_idx=hand_idx,
                    show_axes=args.show_axes,
                )
                make_combined_video_multi(
                    all_video_frames,
                    [skeleton_frames, hand_frames],
                    ["kp3d", "hands"],
                    caption,
                    out_path,
                    fps=args.skeleton_fps,
                    frame_height=args.frame_height,
                )
            else:
                make_combined_video(
                    all_video_frames, skeleton_frames, caption, out_path,
                    fps=args.skeleton_fps, frame_height=args.frame_height,
                )
            print(f"  -> {out_path}")
        except Exception as e:
            print(f"  [ERROR] Combined video: {e}")

    print(f"\nDone. {n} samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
