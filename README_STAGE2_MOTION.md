# Stage2 Motion Training Guide

InternVideo2 に Motion (体の動きトークン) を組み込んだ Stage2 事前学習の手順書。

## 全体パイプライン

```
1. 環境構築
2. 元データの準備 (Ego4D / EgoExo4D)
3. Atomic Clips 切り出し + Motion kp3d 抽出
4. Motion Tokenization (VQ-VAE)
5. Annotation JSON 生成
6. (任意) LMDB 構築
7. 学習
```

---

## 1. 環境構築

### ローカル (GPU マシン)

```bash
cd multi_modality
bash install.sh
```

`install.sh` は以下を行う:
- `.conda/` に conda 環境を作成 (`environment.yml` ベース、Python 3.10)
- Flash Attention v2.6.3 + CUDA 拡張 (fused_dense_lib, dropout_layer_norm) をビルド

### Wisteria (2段階)

ログインノードで conda 環境を作成:
```bash
bash setup_login.sh
```

GPU インタラクティブノードで CUDA 拡張をビルド:
```bash
# インタラクティブジョブを取得してから
bash setup_gpu.sh
```

または `job_flashattn.sh` でジョブ投入:
```bash
pjsub job_flashattn.sh
```

### 主要な依存パッケージ

| パッケージ | バージョン | 用途 |
|-----------|----------|------|
| torch | >= 2.4.1 | 学習基盤 |
| flash_attn | >= 2.6.3 | 高速 Attention |
| deepspeed | >= 0.15.1 | 分散学習 (ZeRO Stage 1) |
| transformers | >= 4.45.1 | BERT テキストエンコーダ |
| decord / av | 動画デコード |
| lmdb, msgpack | LMDB 構築・読み込み |

---

## 2. 元データの準備

以下のデータを `${DATA_ROOT}` 配下に配置する (デフォルト: `EgoHand/data`)。

```
${DATA_ROOT}/
├── description/
│   ├── atomic_descriptions_train.json   # Atomic description アノテーション
│   └── atomic_descriptions_val.json
├── ee4d/
│   └── ee4d_motion_uniegomotion/
│       ├── takes.json                    # take_uid → take_name マッピング
│       └── uniegomotion/
│           ├── ee_train_joints_tips.pt   # 3D 関節キーポイント (train)
│           └── ee_val_joints_tips.pt     # 3D 関節キーポイント (val)
├── train/
│   ├── takes/                            # フル動画 (exo カメラ用)
│   │   └── {take_name}/frame_aligned_videos/{cam_id}.mp4
│   └── takes_clipped/egoexo/videos/      # 既存のクリップ済み ego 動画
│       └── {take_name}/{start}___{end}.mp4
└── val/
    └── (同じ構造)
```

環境変数 `DATA_ROOT` で場所を変更可能:
```bash
export DATA_ROOT=/path/to/your/data
```

---

## 3. Atomic Clips 切り出し + Motion kp3d 抽出

各 atomic description のタイムスタンプを中心に ±2秒 (4秒) のクリップを切り出す。

```bash
cd scripts/pretraining/stage2/1B_motion

# 初回: 動画 + motion kp3d 全部生成
python prepare_atomic_clips.py --split train --validate --skip-existing
python prepare_atomic_clips.py --split val --validate --skip-existing

# kp3d のみ再生成 (動画はスキップ)
python prepare_atomic_clips.py --split train --motion-only
python prepare_atomic_clips.py --split val --motion-only
```

**処理内容:**
- Ego 動画: 既存クリップから該当区間を ffmpeg で切り出し
- Exo 動画: フルテイク動画から best_exo カメラの該当区間を切り出し
- Motion: `ee_{split}_joints_tips.pt` から該当フレームの kp3d (3D 関節点) を抽出し、**常に 41 フレームにリサンプル** (線形補間)

**出力:**
```
${DATA_ROOT}/{split}/takes_clipped/egoexo/
├── videos_atomic/{take_name}/{sample_id}_{ego|exo}.mp4
├── motion_atomic/{take_name}/{sample_id}_kp3d.npy    # [41, 154, 3] (固定長)
└── ffmpeg_logs_atomic/                                # デバッグ用ログ
```

加えて、中間アノテーションが生成される:
```
scripts/pretraining/stage2/1B_motion/
├── annotation_atomic_{split}_intermediate.json   # 全エントリ (OK フラグ付き)
└── annotation_atomic_{split}.json                # フィルタ済み
```

**主なオプション:**
| オプション | 説明 |
|-----------|------|
| `--validate` | 出力 mp4 を decord で検証 (遅いが安全) |
| `--skip-existing` | 既存ファイルをスキップ (再実行時に便利) |
| `--require-both-videos` | ego と exo 両方ある場合のみ残す |
| `--require-motion` | motion kp3d がある場合のみ残す |
| `--motion-only` | kp3d のみ再生成 (動画クリッピングをスキップ) |

---

## 4. Motion Tokenization (VQ-VAE)

kp3d を VQ-VAE でトークン列に変換する。このステップは **BodyTokenize** リポジトリで行う。

```bash
cd /path/to/EgoHand/BodyTokenize

python inference_atomic.py \
    --split train \
    --data-root ${DATA_ROOT}/train/takes_clipped/egoexo
```

**処理内容:**
1. `motion_atomic/{take_name}/{sample_id}_kp3d.npy` (41 フレーム固定) を読み込み
2. kp3d → motion representation (RIC, rotation, velocity, feet contact) → 40 フレーム
3. VQ-VAE (body_down=1) でトークンインデックスに量子化 → 40 timesteps x 8 tokens

kp3d が 41 フレーム固定なので、clip_len=41 (デフォルト) で**チャンク分割なし** (1 サンプル = 1 ファイル)。

**出力:**
```
${DATA_ROOT}/{split}/takes_clipped/egoexo/
└── tok_pose_atomic_40/{take_name}/{sample_id}_0000.npz
    # idx: shape (1, 40, 8), dtype int64
```

NPZ の中身: `idx` キーに motion token indices (codebook サイズ K=1024, 4 body + 4 hand tokens/timestep)

**VQ-VAE チェックポイント:**
- モデル: `EgoHand/BodyTokenize/ckpt_vq/ckpt_best.pt`
- 設定: `EgoHand/BodyTokenize/ckpt_vq/config.yaml`

---

## 5. Annotation JSON 生成

Step 3 の中間アノテーションと Step 4 の tok_pose を紐づけて、最終的な学習用アノテーション JSON を作成する。

```bash
cd scripts/pretraining/stage2/1B_motion

# train
python build_annotation_atomic.py --split train

# val
python build_annotation_atomic.py --split val
```

**出力:**
```
scripts/pretraining/stage2/1B_motion/
├── annotation_atomic_train.json
└── annotation_atomic_val.json
```

**JSON のフォーマット:**
```json
[
  {
    "video": "videos_atomic/{take_name}/{sample_id}_ego.mp4",
    "tok_pose": "tok_pose_atomic_40/{take_name}/{sample_id}_0000.npz",
    "caption": "C adjusts the camera with both hands."
  },
  ...
]
```

- `video`: `data_root` からの相対パス (ego/exo は別エントリとして登録)
- `tok_pose`: `motion_data_root` からの相対パス
- `caption`: テキスト説明文

---

## 6. (任意) LMDB 構築

大量の動画 + NPZ を毎回開くと I/O がボトルネックになるため、事前に LMDB にまとめることで高速化できる。

```bash
cd multi_modality

# train
python tools/build_lmdb.py \
    --ann scripts/pretraining/stage2/1B_motion/annotation_atomic_train.json \
    --data-root ${DATA_ROOT}/train/takes_clipped/egoexo \
    --motion-data-root ${DATA_ROOT}/train/takes_clipped/egoexo \
    --output ${DATA_ROOT}/lmdb/train.lmdb \
    --num-frames 8 \
    --jpeg-quality 95 \
    --map-size-gb 50

# val
python tools/build_lmdb.py \
    --ann scripts/pretraining/stage2/1B_motion/annotation_atomic_val.json \
    --data-root ${DATA_ROOT}/val/takes_clipped/egoexo \
    --motion-data-root ${DATA_ROOT}/val/takes_clipped/egoexo \
    --output ${DATA_ROOT}/lmdb/val.lmdb \
    --num-frames 8 \
    --jpeg-quality 95 \
    --map-size-gb 10
```

動画フレームの読み込みは sequential decode で行うため、`cv2.CAP_PROP_FRAME_COUNT` の不正確さに影響されない。

**LMDB の中身 (1サンプルあたり、msgpack):**
```python
{
    "frames": [jpeg_bytes, ...],   # JPEG エンコード済みフレーム x 8
    "motion_idx": [int, ...],      # motion token indices (40*8 = 320 要素)
    "caption": "text"
}
```

`config.py` で raw / LMDB を切り替え:
```python
# raw ファイル版
train_file = [available_corpus["video_motion_train"]]
# LMDB 版 (デフォルト)
train_file = [available_corpus["video_motion_lmdb_train"]]
```

---

## 7. 学習

### 必要なチェックポイント

| ファイル | 配置先 | 説明 |
|---------|--------|------|
| Stage1 重み | `scripts/pretraining/stage1/1B_ft_k710_f8.pth` | Vision encoder 初期化 |
| VQ-VAE 重み | `EgoHand/BodyTokenize/ckpt_vq/ckpt_best.pt` | Motion encoder (freeze) |
| VQ-VAE 設定 | `EgoHand/BodyTokenize/ckpt_vq/config.yaml` | Motion encoder 設定 |

### ローカル実行

```bash
bash run_stage2.sh
```

GPU を指定する場合:
```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_stage2.sh
```

### Wisteria ジョブ投入

```bash
pjsub job_run_stage2.sh
```

`job_run_stage2.sh` のヘッダでリソースを調整:
```sh
#PJM -L node=1          # ノード数 (1ノード = 8 GPU)
#PJM -L elapse=1:00:00  # 制限時間
```

マルチノード対応済み (`mpirun` + `torchrun` の rendezvous)。

### wandb

学習のモニタリングは [wandb](https://wandb.ai/) で行う。`config.py` の設定:
```python
wandb = dict(
    enable=True,
    entity="narudesuyo-waseda-university",
    project="InternVideo2-Stage2-Motion",
)
```

無効にする場合:
```bash
WANDB_MODE=offline bash run_stage2.sh
```

### 主要な学習パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| num_frames | 8 | 動画フレーム数 |
| motion_T | 40 | Motion 系列長 (40 timesteps x 8 tokens) |
| batch_size | 2 | バッチサイズ/GPU |
| lr | 5e-5 | 学習率 |
| epochs | 10 | エポック数 |
| warmup_epochs | 1 | ウォームアップ |
| image_res | 224 | 入力解像度 |
| deepspeed stage | 1 | ZeRO Stage 1 |

---

## ディレクトリ構成まとめ

```
multi_modality/
├── run_stage2.sh                  # ローカル学習ラッパー
├── job_run_stage2.sh              # Wisteria ジョブスクリプト
├── job_flashattn.sh               # FlashAttn ビルド用ジョブ
├── install.sh                     # ローカル環境構築
├── setup_login.sh                 # Wisteria ログインノード用
├── setup_gpu.sh                   # Wisteria GPU ノード用
├── environment.yml                # conda 環境定義
├── requirements.txt               # pip 依存
├── tools/
│   ├── build_lmdb.py             # LMDB 構築スクリプト
│   └── visualize_dataset.py      # データセット可視化 (video.mp4 + motion.mp4)
├── tasks/
│   └── pretrain.py               # 学習エントリポイント
├── dataset/
│   ├── motion_dataset.py         # raw ファイル用 Dataset
│   └── motion_dataset_lmdb.py    # LMDB 用 Dataset
├── configs/
│   ├── data.py                   # データセット定義
│   └── model.py                  # モデル定義
└── scripts/pretraining/stage2/1B_motion/
    ├── config.py                  # 学習設定
    ├── prepare_atomic_clips.py    # Step 3: クリップ切り出し
    ├── build_annotation_atomic.py # Step 5: アノテーション生成
    ├── annotation_atomic_train.json
    └── annotation_atomic_val.json
```
