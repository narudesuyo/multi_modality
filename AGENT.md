# AGENT.md

## ルール

- 作業を行ったら、このファイルの「変更履歴」セクションに日付・内容を追記すること
- 何を変えたか、なぜ変えたかを簡潔に書く
- 会話中に出てきた重要な情報・判断・知見も記録すること（デバッグで分かったこと、設計判断、環境の注意点など）

---

## TODO: Stage2 データ準備 → 学習パイプライン

詳細は `scripts/pretraining/stage2/1B_motion/README.md` 参照。

### Step 1: EgoExo4D — exo JPGフレーム + kp3d 抽出
- [x] val (`prepare_atomic_clips.py --split val`) — 完了済み。ただし exo 動画 0 件マッチ（val takes 未ダウンロード）
- [x] train (`prepare_atomic_clips.py --split train`) — 92 takes 完了 (4K)。残り 1556 takes は 320px で再実行予定
- [ ] train 320px 全件 (`prepare_atomic_clips.py --split train --frames-dir-name frames_atomic_320`) — 未実行

### Step 2: Assembly101 — フレーム symlink + SMPL-H → kp3d
- [x] train (`prepare_assembly101_clips.py --split train`) — 3,405 samples 完了
- [x] validation (`prepare_assembly101_clips.py --split validation`) — 382 samples 完了

### Step 3: BodyTokenize (tok_pose 生成)
- [x] EgoExo4D train — 1,080 samples 完了
- [ ] EgoExo4D val — val takes 未ダウンロードのため skip
- [x] Assembly101 train — 3,403 samples 完了
- [x] Assembly101 validation — 382 samples 完了

### Step 4: 最終 annotation JSON 生成
- [x] EgoExo4D train (`build_annotation_atomic.py`) — 535 annotations
- [ ] EgoExo4D val — val takes 未ダウンロードのため skip
- [x] Assembly101 train (`build_annotation_assembly101.py`) — 3,403 annotations
- [x] Assembly101 validation (`build_annotation_assembly101.py`) — 382 annotations

### Step 5: 学習
- [x] Stage2 学習実行中（EgoExo4D 535 + Assembly101 3,403 train / Assembly101 382 val）

### 備考
- Step 1 と Step 2 は独立、並列実行可能
- Step 3 は Step 1/2 の完了が必要
- val の exo 動画 0 件問題は要調査（パスかデータ不足の可能性）
- 共有マシン (load avg ~36, 54 users) なので CPU 負荷に注意
- **重要: `DATA_ROOT` は `/work/narus/data` が正しい。** シェルで `/work/narus` にセットされてることがあるので、スクリプト実行時は必ず `DATA_ROOT=/work/narus/data` を明示すること
- **Assembly101 のローカルパス**: `/home/share/datasets/Assembly101/` 直下には `video/` 等はなく、`data_motionlang/` の中に `video/`, `motion/`, `text/` がある。`original/` には `videos/`, `fine-grained-annotations/` 等がある
- Step 1 の 4K JPG フレーム保存は 1 take あたり ~3.7GB。全 1648 take で ~5.9TB 必要で /work に収まらない。学習は 224px なので ffmpeg で縮小保存が必要

---

## 変更履歴

### 2026-02-26

- **BodyTokenize 内の Y-up / fingertip スクリプトを確認**
  - Y-up 変換: `BodyTokenize/motion_representation.py` と `BodyTokenize/preprocess/motion_representation.py` に `axis_stats()` / `pick_up_axis()` / `to_y_up()` / `unify_clip_to_y_up()` が実装済み
  - 指先生成: `BodyTokenize/preprocess/smplx2joints.py` に `append_fingertips_from_vertices()`（SMPL-X 頂点から 10 tips を追加）
  - 指先を含む kinematic chain: `BodyTokenize/paramUtil_add_tips.py`
- **Assembly101 + EgoExo4D 共通の motion preprocess を追加**
  - 追加: `scripts/pretraining/stage2/1B_motion/motion_preprocess.py`
  - 処理内容: Y-up 自動推定 + Z-forward への yaw 回転
  - 適用箇所: `prepare_atomic_clips.py`, `prepare_assembly101_clips.py` の kp3d 保存直前
  - フラグ: `--no-y-up`, `--no-z-forward` で無効化可能（デフォルト有効）
- **Assembly101 の SMPL-H に tips を付与できるように拡張**
  - `prepare_assembly101_clips.py` に `--add-tips` を追加
  - `smplx.vertex_ids` から `smplh`/`smplx` の指先頂点を取得して kp3d 末尾 10 点に追加
- **EgoExo4D の kp3d を BodyTokenize 相当で正規化**
  - 追加: `scripts/pretraining/stage2/1B_motion/normalize_egoexo_kp3d.py`
  - 処理: Z-up→Y-up、floor、root center、face Z+
  - 出力: `EgoExo4D/processed/train/motion_atomic` を上書き

### 2026-02-25

- **`rsync_to_abci.sh`: Assembly101 のローカルパスを修正**
  - `LOCAL_ASSEMBLY101` を `/home/share/datasets/Assembly101` → `/home/share/datasets/Assembly101/data_motionlang` に変更
  - 理由: `video/`, `motion/`, `text/` は直下ではなく `data_motionlang/` 配下にある
  - ローカルの rsync は 3.1.3（`--mkpath` 非対応）。ABCI側は 3.2.5。リモートにディレクトリがない場合は事前に `ssh abci mkdir -p ...` で作る

### 2026-02-24

- **`prepare_atomic_clips.py`: ffmpeg の `-ss` を `-i` の前に移動**
  - 対象: `clip_video_ffmpeg_safe()` と `extract_frames_ffmpeg()` の両方
  - 理由: `-ss` が `-i` の後だと全フレームデコードしてからシークするため、1クリップ数分かかっていた。前に移すことでキーフレームシークになり数秒で済む
  - 4秒クリップなので数フレームのズレは許容範囲
- **`prepare_atomic_clips.py`: `-threads 1` → `-threads 4` に変更 + load average 自動スロットル追加**
  - `FFMPEG_THREADS = 4` でffmpegを高速化
  - `_wait_for_low_load()`: ffmpeg 呼び出し前に `os.getloadavg()[0]` をチェック、28超なら10秒待機を繰り返す
  - マシンは36コア (2x Xeon Gold 6254)。`nproc` は cgroup 制限で4と返すので注意（`lscpu` で確認すること）
- **`prepare_atomic_clips.py`: フレーム抽出時に短辺 320px にリサイズ追加**
  - `FRAME_SHORT_SIDE = 320`、`-vf scale=-2:320`
  - 理由: 4K のまま保存すると 1648 takes で ~5.9TB 必要（/work 残り 1.2TB で足りない）。320px なら ~56GB
  - 学習の `RandomResizedCrop(224, scale=(0.5,1.0))` で拡大が発生しないギリギリのサイズ
  - 既存の 92 takes (4K, 327GB) はそのまま残し、`--skip-existing` でスキップされる
- **`prepare_atomic_clips.py`: `--frames-dir-name` オプション追加**
  - デフォルトは `frames_atomic`（既存動作と同じ）
  - `--frames-dir-name frames_atomic_320` で 320px フレームを `frames_atomic_320/` に保存可能
  - 既存の 4K `frames_atomic/` を壊さずに並行して 320px 版を生成できる
  - 中間 JSON の `video_exo` パスも自動的に `frames_atomic_320/...` になる
- **`build_annotation_atomic.py`: `_resolve_video_rel()` を `frames_atomic*` 全般対応に拡張**
  - `frames_atomic_320/` 等のカスタムディレクトリ名でも正常にパス解決される
- **学習パイプライン全体が通ることを確認**
  - EgoExo4D (535 samples) + Assembly101 (3,403 samples) で train
  - Assembly101 (382 samples) で val
  - Epoch 0 で損失が順調に減少中 (V_mlm: 8.67→2.45, VT_tok_mlm: 7.40→3.76)
