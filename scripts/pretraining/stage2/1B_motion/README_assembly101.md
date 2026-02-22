# Assembly101 + EgoExo4D Mixed Training

Assembly101 (3,405 train samples) を EgoExo4D (1,048 train samples) と混ぜて Stage2 motion pretraining を行うためのパイプライン。

## データ形式の違い

| | EgoExo4D | Assembly101 |
|---|---|---|
| Video | MP4 → JPG frames (`img00001.jpg`) | 既に JPG frames (`000000.jpg`) |
| Motion | kp3d `.npy` [41, 154, 3] | SMPL-H JSON per-frame (axis-angle) |
| Text | JSON annotations | CSV (`video,start_frame,end_frame,text`) |
| Tokenized | `.npz` idx [1, 40, 8] | 本パイプラインで生成 |

## 出力ディレクトリ構造

```
/work/narus/data/train/takes_clipped/assembly101/
├── frames_atomic/{session_id}/{sample_id}/
│   └── img00001.jpg ... img000XX.jpg    (symlink)
├── motion_atomic/{session_id}/{sample_id}_kp3d.npy   [41, 154, 3]
└── tok_pose_atomic_40/{session_id}/{sample_id}_0000.npz  [1, 40, 8]
```

## 前提条件

- `smplx` ライブラリ: `pip install smplx`
- SMPL-H body model (`.pkl`): `/home/narus/2026/EgoHand/BodyTokenize/models/smplx/smplh/SMPLH_MALE.pkl`
  - https://smpl-x.is.tue.mpg.de/ からダウンロード → `BodyTokenize/models/smplx/` に解凍済み

## 実行手順

### Step 1: フレーム抽出 + SMPL-H → kp3d 変換

```bash
cd /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality

# 全サンプル処理 (GPU推奨)
python scripts/pretraining/stage2/1B_motion/prepare_assembly101_clips.py --device cuda

# テスト (最初の10サンプルだけ)
python scripts/pretraining/stage2/1B_motion/prepare_assembly101_clips.py --limit 10 --device cpu
```

主なオプション:
- `--device cuda` : GPU で SMPL-H FK を高速化
- `--skip-existing` : 既存ファイルをスキップ (再開時に便利)
- `--skip-frames` : モーション変換のみ
- `--skip-motion` : フレーム抽出のみ
- `--copy-frames` : symlink ではなくファイルコピー

出力: `annotation_assembly101_train_intermediate.json`

### Step 2: BodyTokenize (tok_pose 生成)

```bash
cd /home/narus/2026/EgoHand/BodyTokenize

python inference_atomic.py \
  --data-root /work/narus/data/train/takes_clipped/assembly101 \
  --motion-dir /work/narus/data/train/takes_clipped/assembly101/motion_atomic \
  --output-dir /work/narus/data/train/takes_clipped/assembly101/tok_pose_atomic_40 \
  --annotation-json /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality/scripts/pretraining/stage2/1B_motion/annotation_assembly101_train_intermediate.json
```

### Step 3: 最終 annotation JSON 生成

```bash
cd /home/narus/2026/EgoHand/InternVideo/InternVideo2/multi_modality

python scripts/pretraining/stage2/1B_motion/build_annotation_assembly101.py
```

出力: `annotation_assembly101_train.json` (EgoExo4D と同一フォーマット)

### Step 4: Mixed Training

`config.py` は修正済み。`train_file` に EgoExo4D + Assembly101 の両方が含まれている。

```python
train_file = [
    available_corpus["video_motion_train"],              # EgoExo4D (~1048)
    available_corpus["video_motion_assembly101_train"],   # Assembly101 (~3405)
]
```

通常通り学習を実行すれば `ConcatDataset` で自動結合される。

## 変更ファイル一覧

| ファイル | 変更 |
|---|---|
| `scripts/.../prepare_assembly101_clips.py` | 新規 — SMPL-H→kp3d + frame symlink |
| `scripts/.../build_annotation_assembly101.py` | 新規 — annotation JSON 生成 |
| `scripts/.../config.py` | 修正 — Assembly101 corpus 追加, mixed train_file |
| `BodyTokenize/inference_atomic.py` | 修正 — 52/62/154 joint 数ハンドリング, `"video"` key 対応 |

## 技術的な詳細

### SMPL-H → kp3d 変換

1. per-frame JSON から SMPL-H パラメータ (axis-angle) を読み込み
2. `smplx` ライブラリで forward kinematics → 52 joints [T, 52, 3]
3. EgoExo4D の 154-joint format にマッピング:
   - `kp3d[0:22]` = body (22 joints)
   - `kp3d[22:25]` = zeros (jaw/eyes — SMPL-X 固有)
   - `kp3d[25:40]` = left hand (15 joints)
   - `kp3d[40:55]` = right hand (15 joints)
   - `kp3d[55:154]` = zeros
4. 41 フレームにリサンプル (線形補間)

### inference_atomic.py の joint 数ハンドリング

```python
if J == 154:   # EgoExo4D format
    kp52 = concat([kp[:, :22], kp[:, 25:55]])
elif J == 62:  # SMPL-H + fingertips
    kp52 = kp[:, :52]
elif J == 52:  # SMPL-H
    kp52 = kp
```

154 format で保存しているため、既存コードパスがそのまま使われる。
