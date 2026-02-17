"""Dataset for Video + Body Motion + Text pre-training.

Annotation JSON format (per sample):
{
    "video": "relative/path/to/video.mp4",
    "body_motion": "relative/path/to/body_motion.npy",   # [T_m, 263]
    "hand_motion": "relative/path/to/hand_motion.npy",   # [T_m, 480]
    "caption": "text description"
}
"""

import logging
import os
import json
import random
import io

import numpy as np
import torch

from dataset.base_dataset import BaseDataset
from dataset.text_prompt import kinetics_templates, imagenet_templates
from dataset.utils import pre_text
from dataset.video_utils import VIDEO_READER_FUNCS
from dataset.serialize import get_local_rank, TorchShmSerializedList

logger = logging.getLogger(__name__)


class VidMotionTxtPtTrainDataset(BaseDataset):
    """Video + Body Motion + Text pre-training dataset."""

    media_type = "video_motion"

    def __init__(
        self,
        ann_file,
        transform,
        num_frames=4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=3,
        num_epochs=1,
        motion_T=21,
    ):
        super().__init__()

        logger.info(f"ann_file: {ann_file}")

        self.media_type = ann_file.media_type
        self.label_file = ann_file.anno_path
        self.data_root = ann_file.data_root
        self.data_root_prefix = ann_file.get("data_root_prefix", "")
        self.min_caption_length = ann_file.get("min_caption_length", 2)
        self.caption_augmentation = ann_file.get("caption_augmentation", None)
        self.transform = transform
        self.motion_T = motion_T

        # Motion data config
        self.motion_data_root = ann_file.get("motion_data_root", self.data_root)
        self.normalize_motion = ann_file.get("normalize_motion", False)
        self.motion_mean_path = ann_file.get("motion_mean_path", None)
        self.motion_std_path = ann_file.get("motion_std_path", None)

        if self.normalize_motion and self.motion_mean_path and self.motion_std_path:
            self.motion_mean = torch.from_numpy(np.load(self.motion_mean_path)).float()
            self.motion_std = torch.from_numpy(np.load(self.motion_std_path)).float()
        else:
            self.motion_mean = None
            self.motion_std = None

        # Video params
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries

        self.use_prompt = ann_file.get("prompt", "") != ""
        if self.use_prompt:
            if ann_file.prompt == "kinetics":
                self.prompt = kinetics_templates
            elif ann_file.prompt == "imagenet":
                self.prompt = imagenet_templates
            else:
                raise NotImplementedError(ann_file.prompt)

        # Load annotations
        if '.json' in self.label_file:
            logger.info(f"Loading json file {self.label_file}")

            if get_local_rank() == 0:
                try:
                    with io.BytesIO(self.client.get(self.label_file)) as f:
                        annos = json.load(f)
                except Exception:
                    with open(self.label_file, 'r') as f:
                        annos = json.load(f)

                if not ann_file.get("jump_filter", False):
                    captions = [pre_text(anno["caption"]) for anno in annos]
                    captions_len = [len(c.split()) for c in captions]
                    logger.info(f"Num samples: {len(captions)}")
                    logger.info(f"Num samples too short: {sum(l < self.min_caption_length for l in captions_len)}")
                    annos = [anno for anno, l in zip(annos, captions_len) if l >= self.min_caption_length]
            else:
                annos = []

            self.anno = TorchShmSerializedList(annos)
            self.num_examples = len(self.anno)
            logger.info(f"num_examples: {self.num_examples}")
        else:
            raise NotImplementedError("We need json file!")

    def __len__(self):
        return self.num_examples

    def get_anno(self, index):
        anno = {}
        anno["caption"] = self.anno[index]["caption"]
        anno["video"] = self.data_root_prefix + os.path.join(self.data_root, self.anno[index]["video"])
        anno["body_motion"] = os.path.join(self.motion_data_root, self.anno[index]["body_motion"])
        anno["hand_motion"] = os.path.join(self.motion_data_root, self.anno[index]["hand_motion"])

        if self.use_prompt:
            anno["caption"] = random.choice(self.prompt).format(anno["caption"])
        return anno

    def load_motion(self, path, target_T=None):
        """Load a motion numpy file and optionally pad/truncate to target_T."""
        motion = np.load(path).astype(np.float32)  # [T_m, D]
        motion = torch.from_numpy(motion)

        if target_T is not None:
            T_cur = motion.shape[0]
            if T_cur > target_T:
                # Random crop
                start = random.randint(0, T_cur - target_T)
                motion = motion[start:start + target_T]
            elif T_cur < target_T:
                # Pad with zeros
                pad = torch.zeros(target_T - T_cur, motion.shape[1], dtype=motion.dtype)
                motion = torch.cat([motion, pad], dim=0)

        return motion

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = pre_text(ann["caption"])

            # Load video
            video, index = self.load_and_transform_media_data(index, ann["video"])

            # Load motion
            body_motion = self.load_motion(ann["body_motion"], target_T=self.motion_T)
            hand_motion = self.load_motion(ann["hand_motion"], target_T=self.motion_T)

            # Normalize motion if needed
            if self.motion_mean is not None and self.motion_std is not None:
                body_dim = body_motion.shape[-1]
                body_motion = (body_motion - self.motion_mean[:body_dim]) / (self.motion_std[:body_dim] + 1e-8)
                hand_motion = (hand_motion - self.motion_mean[body_dim:body_dim + hand_motion.shape[-1]]) / (self.motion_std[body_dim:body_dim + hand_motion.shape[-1]] + 1e-8)

            # media = [video, body_motion, hand_motion]
            media = [video, body_motion, hand_motion]
            return media, caption, index

        except Exception as e:
            logger.warning(f"Caught exception {e} when loading {ann}")
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
