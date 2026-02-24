# ABCI Stage2 学習パイプライン

ABCI (H200x8, グループ gch51606) で Stage2 Video-Motion-Text 対照学習を実行する手順。

## ABCI 上のディレクトリ構成

```
/home/ach18478ho/code/tmp_code/
├── multi_modality/              ← メインコード (git clone 済み)
└── BodyTokenize/                ← git clone 予定

/groups/gch51606/dataset/egoexo4d/           ← 既存 (raw EgoExo4D takes)

/groups/gch51606/takehiko.ohkawa/tmp_data/   ← DATA_ROOT + ckpt
├── ee4d/                          # rsync (96GB)
├── description/                   # rsync (75MB)
├── EgoExo4D/
│   ├── takes/                     # copy_egoexo4d_takes.sh で ABCI 既存データから抽出
│   ├── take_list_{train,val}.json
│   └── processed/{split}/
├── Assembly101/                   # rsync (58GB)
│   └── processed/{split}/
├── 1B_ft_k710_f8.pth             # rsync (2GB): Stage1 pretrained
└── BodyTokenize/                  # rsync (146MB)
    ├── ckpt_best.pt
    ├── config.yaml
    └── statistics/{mean,std}.npy
```

## 環境変数

| 変数 | ローカル (フォールバック) | ABCI |
|------|-------------------------|------|
| `DATA_ROOT` | `/work/narus/data` | `/groups/gch51606/takehiko.ohkawa/tmp_data` |
| `CKPT_DIR` | (未設定→相対パス) | `/groups/gch51606/takehiko.ohkawa/tmp_data` |
| `BODYTOKENIZE_DIR` | (未設定→相対パス) | `/home/ach18478ho/code/tmp_code/BodyTokenize` |

## 実行フロー

### Step 0: データ転送 (ローカルで実行)

```bash
# ローカルマシンから ABCI へデータを rsync (~156 GB)
bash scripts/abci/rsync_to_abci.sh --dry-run   # まず確認
bash scripts/abci/rsync_to_abci.sh              # 実行
```

転送内容:
- `ee4d/` (~96 GB) — EgoExo4D motion data
- `description/` (~75 MB) — atomic descriptions
- `1B_ft_k710_f8.pth` (~2 GB) — Stage1 pretrained weights
- `BodyTokenize/` (~146 MB) — VQ-VAE checkpoint + config + statistics
- `Assembly101/` (~58 GB) — raw video + motion + text

### Step 1: 環境構築 (ABCI, 初回のみ)

```bash
# インタラクティブ GPU ノードを取得
qrsh -g gch51606 -l rt_HG=1 -l h_rt=1:00:00

# conda 環境 + 依存パッケージのインストール
bash scripts/abci/setup.sh
```

### Step 2: EgoExo4D takes コピー (ABCI)

ABCI 共有データセット → DATA_ROOT へ必要な takes のみコピー。

```bash
export DATA_ROOT=/groups/gch51606/takehiko.ohkawa/tmp_data

bash scripts/abci/copy_egoexo4d_takes.sh --dry-run   # 確認
bash scripts/abci/copy_egoexo4d_takes.sh              # 実行
```

### Step 3: データ準備 (ABCI GPU ノード)

全パイプライン一括実行:

```bash
qrsh -g gch51606 -l rt_HG=1 -l h_rt=6:00:00

export DATA_ROOT=/groups/gch51606/takehiko.ohkawa/tmp_data
export BODYTOKENIZE_DIR=/home/ach18478ho/code/tmp_code/BodyTokenize

bash scripts/abci/prepare_data.sh
```

ステップごとに実行する場合:

```bash
bash scripts/abci/prepare_data.sh --step 1    # EgoExo4D frames + kp3d
bash scripts/abci/prepare_data.sh --step 2    # Assembly101 frames + kp3d
bash scripts/abci/prepare_data.sh --step 3    # BodyTokenize (要 GPU)
bash scripts/abci/prepare_data.sh --step 4    # Annotation JSON 生成
```

### Step 4: 学習ジョブ投入 (ABCI)

```bash
# PBS スクリプト確認 (dry-run)
python scripts/abci/job.py --use-bf16 --dry-run

# ジョブ投入
python scripts/abci/job.py --use-bf16

# カスタムオプション例
python scripts/abci/job.py \
    --use-bf16 \
    --walltime 48:00:00 \
    --job-name stage2-motion-v1 \
    --conda-prefix /home/ach18478ho/code/tmp_code/multi_modality/.conda
```

## 検証

```bash
# 1. ローカルで config.py が CKPT_DIR なしで動くことを確認
python -c "
import os
exec(open('scripts/pretraining/stage2/1B_motion/config.py').read())
print('pretrained:', model['vision_encoder']['pretrained'])
print('ckpt_path:', model['motion_encoder']['ckpt_path'])
print('vqvae_config:', model['motion_encoder']['vqvae_config'])
"

# 2. ABCI 用パスの確認
CKPT_DIR=/groups/gch51606/takehiko.ohkawa/tmp_data python -c "
import os
exec(open('scripts/pretraining/stage2/1B_motion/config.py').read())
print('pretrained:', model['vision_encoder']['pretrained'])
print('ckpt_path:', model['motion_encoder']['ckpt_path'])
print('vqvae_config:', model['motion_encoder']['vqvae_config'])
"

# 3. job.py dry-run で PBS スクリプト確認
python scripts/abci/job.py --dry-run

# 4. rsync 転送リスト確認
bash scripts/abci/rsync_to_abci.sh --dry-run
```

## ファイル一覧

| ファイル | 説明 |
|------|-------------|
| `setup.sh` | conda 環境構築 (初回のみ) |
| `job.py` | PBS ジョブ投入スクリプト |
| `rsync_to_abci.sh` | ローカル → ABCI データ転送 |
| `copy_egoexo4d_takes.sh` | ABCI 共有データ → DATA_ROOT takes コピー |
| `prepare_data.sh` | 全データ準備パイプライン (Step 1-4) |
