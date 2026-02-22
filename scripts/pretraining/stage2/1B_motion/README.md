# Stage 2: Video + Body Motion + Text Pre-training (1B)

## Data Preparation

### 1. Prepare atomic clips

Clip 4-second segments from EgoExo4D videos and extract motion data:

```bash
python prepare_atomic_clips.py --split train --validate --skip-existing
python prepare_atomic_clips.py --split val --validate --skip-existing
```

### 2. Extract JPG frames (recommended)

Convert MP4 clips to JPG frame directories to avoid decord hangs on
corrupted videos. This replaces MP4 decoding with simple JPG reads.

```bash
python extract_frames.py --split train --workers 8
python extract_frames.py --split val --workers 8
```

This produces:
- `annotation_atomic_{split}_frames.json` -- annotation with frame directory paths
- `frames_atomic/{take_name}/{sample_id}_{ego,exo}/img00001.jpg ...` -- extracted frames
- `bad_frames_{split}.txt` -- list of MP4s that failed extraction

Output size: ~54 GB for train (all ~120 frames per clip at quality 2).

### 3. Update config

In `config.py`, point `anno_path` to the frames annotation:

```python
available_corpus["video_motion_train"] = dict(
    anno_path=_os.path.join(_HERE, "annotation_atomic_train_frames.json"),
    ...
)
```

The config already sets `video_reader_type="img"` to read JPG frame
directories instead of MP4 files.

## Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 bash run_local.sh --nproc_per_node 1

# Multi-GPU (auto-detected)
bash run_local.sh
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Training config (model, data, optimizer, etc.) |
| `run_local.sh` | Launch script (auto-detects GPUs) |
| `prepare_atomic_clips.py` | Clip videos + extract motion from EgoExo4D |
| `extract_frames.py` | Convert MP4 clips to JPG frames |
| `validate_videos.py` | Check MP4 readability with decord |
| `build_annotation.py` | Build annotation JSON from raw data |
