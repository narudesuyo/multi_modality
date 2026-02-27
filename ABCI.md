# ABCI 実行手順

## Step 1: 環境構築 (初回のみ)

```bash
qrsh -g gch51606 -l rt_HG=1 -l h_rt=1:00:00
bash scripts/abci/setup.sh
```

## Step 2: EgoExo4D takes コピー

```bash
python scripts/abci/job_copy_egoexo4d.py --dry-run
python scripts/abci/job_copy_egoexo4d.py
```

## Step 3: EgoExo4D フレーム + kp3d 抽出

```bash
python scripts/abci/job_prepare_atomic.py --split train
python scripts/abci/job_prepare_atomic.py --split val
```

## Step 4: Assembly101 フレーム + kp3d 変換

```bash
python scripts/abci/job_prepare_assembly101.py --split train
python scripts/abci/job_prepare_assembly101.py --split validation
```

## Step 5: BodyTokenize VQ-VAE

```bash
python scripts/abci/job_bodytokenize.py
```

## Step 6: Annotation JSON 生成

```bash
python scripts/abci/job_build_annotation.py
```

## Step 7: 学習

```bash
python scripts/abci/job.py --use-bf16
```
