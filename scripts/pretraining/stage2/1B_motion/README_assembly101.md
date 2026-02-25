# Assembly101 + EgoExo4D Mixed Training

Assembly101 (3,405 train samples) を EgoExo4D (1,048 train samples) と混ぜて Stage2 motion pretraining を行うためのパイプライン。

## 出力ディレクトリ構造

```
$DATA_ROOT/Assembly101/processed/
├── train/
│   ├── frames_atomic/{session_id}/{sample_id}/
│   │   └── img00001.jpg ... img000XX.jpg    (symlink)
│   ├── motion_atomic/{session_id}/{sample_id}_kp3d.npy   [41, 154, 3]
│   └── tok_pose_atomic_40/{session_id}/{sample_id}_0000.npz  [1, 40, 8]
└── validation/
    └── (同構造)
```

## 前提条件

- `smplx` ライブラリ: `pip install smplx`
- SMPL-H body model (`.pkl`): `/home/narus/2026/EgoHand/BodyTokenize/models/smplx/smplh/SMPLH_MALE.pkl`

## 実行手順

全手順は [README.md](README.md) の Step 2 / Step 4 / Step 5 を参照。

### クイックリファレンス

```bash
export DATA_ROOT=/work/narus/data
cd /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality

# Step 1: フレーム + kp3d
python scripts/pretraining/stage2/1B_motion/prepare_assembly101_clips.py \
  --split train --device cuda

# Step 2: tok_pose (BodyTokenize)
cd /home/narus/2026/EgoHand/BodyTokenize
python inference_atomic.py \
  --data-root $DATA_ROOT/Assembly101/processed/train \
  --motion-dir $DATA_ROOT/Assembly101/processed/train/motion_atomic \
  --output-dir $DATA_ROOT/Assembly101/processed/train/tok_pose_atomic_40 \
  --annotation-json /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality/scripts/pretraining/stage2/1B_motion/annotation_assembly101_train_intermediate.json

# Step 3: annotation JSON
cd /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality
python scripts/pretraining/stage2/1B_motion/build_annotation_assembly101.py --split train
```

Validation split は `--split validation` に差し替えるだけ。

## config.py での設定

```python
train_file = [
    available_corpus["video_motion_train"],              # EgoExo4D (~1048)
    available_corpus["video_motion_assembly101_train"],   # Assembly101 (~3405)
]
```

`ConcatDataset` で自動結合される。
