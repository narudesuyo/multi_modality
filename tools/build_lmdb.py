"""Build LMDB from annotation JSON + MP4 videos + NPZ motion data.

Extracts N frames (uniform middle sampling) from each video, JPEG-encodes them,
and packs {frames, motion_idx, caption} into LMDB via msgpack.

Usage:
    python tools/build_lmdb.py \
        --ann path/to/annotation.json \
        --data-root path/to/data_root \
        --output path/to/output.lmdb \
        --num-frames 8
"""

import argparse
import io
import json
import os
import sys

import cv2
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm


def get_frame_indices_middle(num_frames, vlen):
    """Uniform middle sampling — deterministic, same as eval."""
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    frame_indices = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(acc_samples)]
    if len(frame_indices) < num_frames:
        frame_indices += [frame_indices[-1]] * (num_frames - len(frame_indices))
    return frame_indices


def encode_jpeg(frame_bgr, quality=95):
    """Encode a BGR numpy frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def read_video_cv2(video_path, frame_indices):
    """Read specific frames from video using cv2.VideoCapture."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
        frames.append(frame_bgr)
    cap.release()
    return frames  # list of BGR numpy arrays


def process_sample(ann, data_root, motion_data_root, num_frames, jpeg_quality):
    """Process one sample: decode video frames + load motion indices."""
    video_path = os.path.join(data_root, ann["video"])

    # Get frame count
    cap = cv2.VideoCapture(video_path)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    indices = get_frame_indices_middle(num_frames, vlen)
    frames_bgr = read_video_cv2(video_path, indices)

    jpeg_frames = []
    for frame_bgr in frames_bgr:
        jpeg_frames.append(encode_jpeg(frame_bgr, jpeg_quality))

    # Motion indices
    motion_path = os.path.join(motion_data_root, ann["tok_pose"])
    data = np.load(motion_path)
    motion_idx = data["idx"].flatten().astype(np.int64).tolist()

    caption = ann["caption"]

    return msgpack.packb({
        "frames": jpeg_frames,
        "motion_idx": motion_idx,
        "caption": caption,
    }, use_bin_type=True)


def main():
    parser = argparse.ArgumentParser(description="Build LMDB from video+motion annotation")
    parser.add_argument("--ann", required=True, help="Path to annotation JSON")
    parser.add_argument("--data-root", required=True, help="Root dir for video and motion files")
    parser.add_argument("--motion-data-root", default=None, help="Root dir for motion files (default: same as data-root)")
    parser.add_argument("--output", required=True, help="Output LMDB path")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to extract")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG encode quality")
    parser.add_argument("--map-size-gb", type=float, default=50, help="LMDB map size in GB")
    args = parser.parse_args()

    motion_data_root = args.motion_data_root or args.data_root

    with open(args.ann, "r") as f:
        annos = json.load(f)
    print(f"Loaded {len(annos)} annotations from {args.ann}")

    map_size = int(args.map_size_gb * 1024 ** 3)
    env = lmdb.open(args.output, map_size=map_size, readonly=False)

    num_ok = 0
    num_fail = 0
    with env.begin(write=True) as txn:
        for i, ann in enumerate(tqdm(annos, desc="Building LMDB")):
            try:
                value = process_sample(ann, args.data_root, motion_data_root, args.num_frames, args.jpeg_quality)
                txn.put(f"{i:08d}".encode(), value)
                num_ok += 1
            except Exception as e:
                print(f"[WARN] Skip sample {i}: {e}", file=sys.stderr)
                num_fail += 1

        # Store metadata
        meta = msgpack.packb({
            "num_samples": num_ok,
            "num_frames": args.num_frames,
            "jpeg_quality": args.jpeg_quality,
        }, use_bin_type=True)
        txn.put(b"__meta__", meta)

    env.close()
    print(f"Done. {num_ok} samples written, {num_fail} failed. Output: {args.output}")


if __name__ == "__main__":
    main()
