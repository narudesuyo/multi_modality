# Stage 2: Video + Body Motion + Text Pre-training (1B)

EgoExo4D (exo view) + Assembly101 の混合データで Video-Motion-Text 対照学習を行う。

## ディレクトリ構成

全スクリプトは `DATA_ROOT=/work/narus/data` を基点とするデータセット中心のパス構成を使う。

```
$DATA_ROOT=/work/narus/data/
├── EgoExo4D/
│   ├── takes/                       # raw takes (1,585 takes)
│   ├── take_list.json
│   └── processed/
│       ├── train/
│       │   ├── frames_atomic/       # JPG フレーム (exo のみ, 直接抽出)
│       │   ├── motion_atomic/       # kp3d .npy [41, 154, 3]
│       │   ├── tok_pose_atomic_40/  # VQ-VAE tokens .npz [1, 40, 8]
│       │   └── ffmpeg_logs_atomic/
│       └── val/
│           └── (同構造)
├── Assembly101/
│   ├── video/                       # raw JPG frames
│   ├── motion/v1/                   # SMPL-H per-frame JSON
│   ├── text/v1/                     # CSV annotation
│   └── processed/
│       ├── train/
│       │   ├── frames_atomic/       # symlink された JPG frames
│       │   ├── motion_atomic/       # kp3d .npy [41, 154, 3]
│       │   └── tok_pose_atomic_40/  # VQ-VAE tokens .npz [1, 40, 8]
│       └── validation/
│           └── (同構造)
├── description/                     # atomic_descriptions_{split}.json
└── ee4d/                            # ee4d_motion_uniegomotion/
```

## データ準備の全手順

```bash
export DATA_ROOT=/work/narus/data
cd /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality
```

### Step 1: EgoExo4D — exo JPGフレーム + kp3d 抽出

raw takes から直接 JPG フレームを抽出 (MP4 中間ファイルなし)。
`--exo-only` がデフォルト。

```bash
# Train
python scripts/pretraining/stage2/1B_motion/prepare_atomic_clips.py \
  --split train --skip-existing

# Val
python scripts/pretraining/stage2/1B_motion/prepare_atomic_clips.py \
  --split val --skip-existing
```

出力:
- `EgoExo4D/processed/{split}/frames_atomic/` — exo JPG フレーム
- `EgoExo4D/processed/{split}/motion_atomic/` — kp3d `.npy` [41, 154, 3]
- `annotation_atomic_{split}_intermediate.json`

### Step 2: Assembly101 — フレーム symlink + SMPL-H → kp3d 変換

```bash
# Train (3,405 samples)
python scripts/pretraining/stage2/1B_motion/prepare_assembly101_clips.py \
  --split train --device cuda

# Validation (382 samples)
python scripts/pretraining/stage2/1B_motion/prepare_assembly101_clips.py \
  --split validation --device cuda
```

出力:
- `Assembly101/processed/{split}/frames_atomic/` — JPG フレーム (symlink)
- `Assembly101/processed/{split}/motion_atomic/` — kp3d `.npy` [41, 154, 3]
- `annotation_assembly101_{split}_intermediate.json`

主なオプション:
- `--device cuda` : GPU で SMPL-H FK を高速化
- `--skip-existing` : 既存ファイルをスキップ (再開時に便利)
- `--skip-frames` / `--skip-motion` : 片方だけ処理
- `--limit N` : テスト用に N サンプルだけ処理

### Step 3: BodyTokenize (tok_pose 生成)

```bash
cd /home/narus/2026/EgoHand/BodyTokenize

# EgoExo4D train
python inference_atomic.py \
  --data-root $DATA_ROOT/EgoExo4D/processed/train \
  --motion-dir $DATA_ROOT/EgoExo4D/processed/train/motion_atomic \
  --output-dir $DATA_ROOT/EgoExo4D/processed/train/tok_pose_atomic_40 \
  --annotation-json /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality/scripts/pretraining/stage2/1B_motion/annotation_atomic_train_intermediate.json

# EgoExo4D val
python inference_atomic.py \
  --data-root $DATA_ROOT/EgoExo4D/processed/val \
  --motion-dir $DATA_ROOT/EgoExo4D/processed/val/motion_atomic \
  --output-dir $DATA_ROOT/EgoExo4D/processed/val/tok_pose_atomic_40 \
  --annotation-json /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality/scripts/pretraining/stage2/1B_motion/annotation_atomic_val_intermediate.json

# Assembly101 train
python inference_atomic.py \
  --data-root $DATA_ROOT/Assembly101/processed/train \
  --motion-dir $DATA_ROOT/Assembly101/processed/train/motion_atomic \
  --output-dir $DATA_ROOT/Assembly101/processed/train/tok_pose_atomic_40 \
  --annotation-json /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality/scripts/pretraining/stage2/1B_motion/annotation_assembly101_train_intermediate.json

# Assembly101 validation
python inference_atomic.py \
  --data-root $DATA_ROOT/Assembly101/processed/validation \
  --motion-dir $DATA_ROOT/Assembly101/processed/validation/motion_atomic \
  --output-dir $DATA_ROOT/Assembly101/processed/validation/tok_pose_atomic_40 \
  --annotation-json /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality/scripts/pretraining/stage2/1B_motion/annotation_assembly101_validation_intermediate.json
```

### Step 4: 最終 annotation JSON 生成

`--exo-only` がデフォルト (exo view のみ annotation に含む)。

```bash
cd /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality

# EgoExo4D
python scripts/pretraining/stage2/1B_motion/build_annotation_atomic.py --split train
python scripts/pretraining/stage2/1B_motion/build_annotation_atomic.py --split val

# Assembly101
python scripts/pretraining/stage2/1B_motion/build_annotation_assembly101.py --split train
python scripts/pretraining/stage2/1B_motion/build_annotation_assembly101.py --split validation
```

### Step 5: 学習

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/pretraining/stage2/1B_motion/run_local.sh --nproc_per_node 1

# Multi-GPU
bash scripts/pretraining/stage2/1B_motion/run_local.sh
```

## 並列実行

Step 1 (EgoExo4D) と Step 2 (Assembly101) は独立。train / val も独立。
4ターミナルで並列実行可能:

```
Terminal 1: EgoExo4D train   → Step 1 → Step 3 → Step 4
Terminal 2: EgoExo4D val     → Step 1 → Step 3 → Step 4
Terminal 3: Assembly101 train      → Step 2 → Step 3 → Step 4
Terminal 4: Assembly101 validation → Step 2 → Step 3 → Step 4
```

Step 1 完了後、同一 split の Step 3 と Step 4 は同時に走らせられる。

## パス確認

```bash
export DATA_ROOT=/work/narus/data
python -c "
import os; os.environ['DATA_ROOT']='$DATA_ROOT'
exec(open('scripts/pretraining/stage2/1B_motion/config.py').read())
for k in ['video_motion_train','video_motion_val',
          'video_motion_assembly101_train','video_motion_assembly101_val']:
    print(f'{k}:')
    print(f'  data_root: {available_corpus[k][\"data_root\"]}')
    print(f'  anno_path: {available_corpus[k][\"anno_path\"]}')
"
```

## データ形式の違い

| | EgoExo4D | Assembly101 |
|---|---|---|
| Video | exo MP4 → JPG frames (`img00001.jpg`) | 既に JPG frames (`000000.jpg`) → symlink |
| Motion | kp3d `.npy` [41, 154, 3] | SMPL-H JSON → kp3d `.npy` [41, 154, 3] |
| Text | JSON (atomic descriptions) | CSV (`video,start_frame,end_frame,text`) |
| Tokenized | `.npz` idx [1, 40, 8] | `.npz` idx [1, 40, 8] |

## ファイル一覧

| ファイル | 説明 |
|------|-------------|
| `config.py` | 学習設定 (model, data, optimizer, etc.) |
| `run_local.sh` | 起動スクリプト (GPU 自動検出) |
| `prepare_atomic_clips.py` | EgoExo4D: exo JPG フレーム + kp3d 抽出 |
| `prepare_assembly101_clips.py` | Assembly101: フレーム symlink + SMPL-H → kp3d |
| `build_annotation_atomic.py` | EgoExo4D: 最終 annotation JSON 生成 (exo only) |
| `build_annotation_assembly101.py` | Assembly101: 最終 annotation JSON 生成 |
| `extract_frames.py` | (旧) MP4 → JPG 変換 (現在は prepare_atomic_clips.py に統合) |
| `build_annotation.py` | (旧) raw data からの annotation 生成 |
| `validate_videos.py` | (旧) MP4 の decord 互換性チェック |

## SMPL-H → kp3d 変換の詳細

1. per-frame JSON から SMPL-H パラメータ (axis-angle) を読み込み
2. `smplx` ライブラリで forward kinematics → 52 joints [T, 52, 3]
3. EgoExo4D の 154-joint format にマッピング:
   - `kp3d[0:22]` = body (22 joints)
   - `kp3d[22:25]` = zeros (jaw/eyes)
   - `kp3d[25:40]` = left hand (15 joints)
   - `kp3d[40:55]` = right hand (15 joints)
   - `kp3d[55:154]` = zeros
4. 41 フレームにリサンプル (線形補間)
