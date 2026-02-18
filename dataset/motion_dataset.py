"""Dataset for Video + Motion (tok_pose indices) + Text pre-training.

Annotation JSON format (per sample):
{
    "video": "relative/path/to/video.mp4",
    "tok_pose": "relative/path/to/tok_pose.npz",   # contains 'idx' array [N]
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
    """Video + Motion (token indices) + Text pre-training dataset."""

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
        anno["tok_pose"] = os.path.join(self.motion_data_root, self.anno[index]["tok_pose"])

        if self.use_prompt:
            anno["caption"] = random.choice(self.prompt).format(anno["caption"])
        return anno

    def load_tok_pose(self, path):
        """Load tokenized pose indices from .npz file.

        Returns: LongTensor [N] of token indices.
        """
        data = np.load(path)
        idx = data["idx"].flatten().astype(np.int64)
        return torch.from_numpy(idx)

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = pre_text(ann["caption"])

            # Load video
            video, index = self.load_and_transform_media_data(index, ann["video"])

            # Load tokenized pose indices
            motion_indices = self.load_tok_pose(ann["tok_pose"])

            media = [video, motion_indices]
            return media, caption, index

        except Exception as e:
            logger.warning(f"Caught exception {e} when loading {ann}")
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
