"""LMDB-backed dataset for Video + Motion + Text pre-training.

Reads pre-built LMDB (created by tools/build_lmdb.py) containing
JPEG-encoded frames, motion token indices, and captions.
"""

import logging
import io

import cv2
import lmdb
import msgpack
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.utils import pre_text

logger = logging.getLogger(__name__)


def _decode_jpeg(buf):
    """Decode JPEG bytes to uint8 numpy array (H, W, C) RGB."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


class VidMotionTxtLmdbPtTrainDataset(Dataset):
    """LMDB-backed training dataset for video + motion + text."""

    media_type = "video_motion"

    def __init__(self, ann_file, transform, num_frames=8, motion_T=21, **kwargs):
        super().__init__()
        self.transform = transform
        self.num_frames = num_frames
        self.motion_T = motion_T

        lmdb_path = ann_file.lmdb_path
        logger.info(f"Opening LMDB: {lmdb_path}")
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)

        # Read metadata
        with self.env.begin(write=False) as txn:
            meta_raw = txn.get(b"__meta__")
            if meta_raw is not None:
                meta = msgpack.unpackb(meta_raw, raw=False)
                self.num_examples = meta["num_samples"]
                self.stored_num_frames = meta["num_frames"]
                logger.info(f"LMDB meta: {meta}")
            else:
                raise RuntimeError(f"LMDB at {lmdb_path} has no __meta__ key")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        try:
            return self._get_item(index)
        except Exception as e:
            logger.warning(f"Error loading sample {index}: {e}")
            new_index = np.random.randint(0, len(self))
            return self.__getitem__(new_index)

    def _get_item(self, index):
        with self.env.begin(write=False) as txn:
            raw = txn.get(f"{index:08d}".encode())
        if raw is None:
            raise KeyError(f"Key {index:08d} not found in LMDB")

        sample = msgpack.unpackb(raw, raw=False)

        # Decode JPEG frames → (T, C, H, W) uint8 tensor
        frames_rgb = [_decode_jpeg(buf) for buf in sample["frames"]]
        frames = np.stack(frames_rgb, axis=0)  # (T, H, W, C)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)

        # Sub-sample if stored frames > requested frames
        if frames.shape[0] > self.num_frames:
            indices = np.linspace(0, frames.shape[0] - 1, self.num_frames).astype(int)
            frames = frames[indices]

        # Apply transforms (RandomResizedCrop, flip, normalize)
        frames = self.transform(frames)

        # Motion indices
        motion_indices = torch.LongTensor(sample["motion_idx"])

        caption = pre_text(sample["caption"])

        media = [frames, motion_indices]
        return media, caption, index


class VidMotionTxtLmdbRetEvalDataset(Dataset):
    """LMDB-backed eval dataset for video + motion + text retrieval."""

    media_type = "video_motion"

    def __init__(self, ann_file, transform, num_frames=8, **kwargs):
        super().__init__()
        self.transform = transform
        self.num_frames = num_frames
        self.max_txt_l = ann_file.get("max_txt_l", 32)

        lmdb_path = ann_file.lmdb_path
        logger.info(f"Opening LMDB (eval): {lmdb_path}")
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)

        with self.env.begin(write=False) as txn:
            meta_raw = txn.get(b"__meta__")
            meta = msgpack.unpackb(meta_raw, raw=False)
            self.num_examples = meta["num_samples"]

        # Build text/retrieval mappings by reading all captions
        self.text = []
        self.txt2img = {}
        self.img2txt = {}
        with self.env.begin(write=False) as txn:
            for i in range(self.num_examples):
                raw = txn.get(f"{i:08d}".encode())
                sample = msgpack.unpackb(raw, raw=False)
                self.text.append(pre_text(sample["caption"]))
                self.txt2img[i] = i
                self.img2txt[i] = [i]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        try:
            with self.env.begin(write=False) as txn:
                raw = txn.get(f"{index:08d}".encode())
            sample = msgpack.unpackb(raw, raw=False)

            frames_rgb = [_decode_jpeg(buf) for buf in sample["frames"]]
            frames = np.stack(frames_rgb, axis=0)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

            if frames.shape[0] > self.num_frames:
                indices = np.linspace(0, frames.shape[0] - 1, self.num_frames).astype(int)
                frames = frames[indices]

            frames = self.transform(frames)
            motion_indices = torch.LongTensor(sample["motion_idx"])

            return [frames, motion_indices], index
        except Exception as e:
            logger.warning(f"Error loading eval sample {index}: {e}")
            new_index = np.random.randint(0, len(self))
            return self.__getitem__(new_index)
