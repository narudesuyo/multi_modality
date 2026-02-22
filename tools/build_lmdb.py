"""Build LMDB from annotation JSON + MP4 videos + NPZ motion data.

Extracts N frames (uniform middle sampling) from each video, JPEG-encodes them,
and packs {frames, motion_idx, caption} into LMDB via msgpack.

Usage:
    python tools/build_lmdb.py \
        --ann path/to/annotation.json \
        --data-root path/to/data_root \
        --output path/to/output.lmdb \
        --num-frames 8 \
        --workers 16
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

try:
    import decord
    decord.bridge.set_bridge("native")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


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


def read_and_sample_video(video_path, num_frames):
    """Read only the needed frames using decord (fast seek), cv2 fallback."""
    if HAS_DECORD:
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            vlen = len(vr)
            if vlen < 1:
                raise RuntimeError(f"No frames in {video_path}")
            indices = get_frame_indices_middle(num_frames, vlen)
            frames_rgb = vr.get_batch(indices).asnumpy()  # [N, H, W, 3] RGB
            # Convert RGB -> BGR for JPEG encoding (cv2 expects BGR)
            return [frame[:, :, ::-1] for frame in frames_rgb]
        except Exception:
            pass  # Fall through to cv2

    # cv2 fallback: seek to specific frames instead of reading all
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if vlen < 1:
        # Frame count unreliable — read all frames
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        vlen = len(all_frames)
        if vlen < 1:
            raise RuntimeError(f"No readable frames in {video_path}")
        indices = get_frame_indices_middle(num_frames, vlen)
        return [all_frames[i] for i in indices]

    indices = get_frame_indices_middle(num_frames, vlen)
    frames = []
    for idx in sorted(set(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
        frames.append((idx, frame))
    cap.release()

    idx_to_frame = {idx: frame for idx, frame in frames}
    return [idx_to_frame[i] for i in indices]


def process_sample(args_tuple):
    """Worker function: process one sample. Returns (index, packed_bytes) or (index, None)."""
    i, ann, data_root, motion_data_root, num_frames, jpeg_quality = args_tuple
    try:
        video_path = os.path.join(data_root, ann["video"])
        frames_bgr = read_and_sample_video(video_path, num_frames)

        jpeg_frames = [encode_jpeg(f, jpeg_quality) for f in frames_bgr]

        motion_path = os.path.join(motion_data_root, ann["tok_pose"])
        data = np.load(motion_path)
        motion_idx = data["idx"].flatten().astype(np.int64).tolist()

        value = msgpack.packb({
            "frames": jpeg_frames,
            "motion_idx": motion_idx,
            "caption": ann["caption"],
        }, use_bin_type=True)

        return (i, value)
    except Exception as e:
        print(f"[WARN] Skip sample {i}: {e}", file=sys.stderr)
        return (i, None)


def main():
    parser = argparse.ArgumentParser(description="Build LMDB from video+motion annotation")
    parser.add_argument("--ann", required=True, help="Path to annotation JSON")
    parser.add_argument("--data-root", required=True, help="Root dir for video and motion files")
    parser.add_argument("--motion-data-root", default=None, help="Root dir for motion files (default: same as data-root)")
    parser.add_argument("--output", required=True, help="Output LMDB path")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to extract")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG encode quality")
    parser.add_argument("--map-size-gb", type=float, default=50, help="LMDB map size in GB")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = single process)")
    args = parser.parse_args()

    motion_data_root = args.motion_data_root or args.data_root

    with open(args.ann, "r") as f:
        annos = json.load(f)
    print(f"Loaded {len(annos)} annotations from {args.ann}")
    print(f"Using {args.workers} workers (decord={'yes' if HAS_DECORD else 'no'})")

    work_items = [
        (i, ann, args.data_root, motion_data_root, args.num_frames, args.jpeg_quality)
        for i, ann in enumerate(annos)
    ]

    map_size = int(args.map_size_gb * 1024 ** 3)
    env = lmdb.open(args.output, map_size=map_size, readonly=False)

    num_ok = 0
    num_fail = 0

    if args.workers > 0:
        iterator = Pool(processes=args.workers).imap_unordered(process_sample, work_items)
    else:
        iterator = map(process_sample, work_items)

    with env.begin(write=True) as txn:
        for i, value in tqdm(iterator, total=len(work_items), desc="Building LMDB"):
            if value is not None:
                txn.put(f"{i:08d}".encode(), value)
                num_ok += 1
            else:
                num_fail += 1

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
