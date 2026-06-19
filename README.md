# DOLO

**D**rosophila tracking with **YOLO** **P**ose — a pipeline for pose estimation, multi-fly tracking, and behavioral analysis in *Drosophila* videos.

DOLO combines [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose/) with custom annotation tools, per-video fine-tuning, and a multi-video foundation training workflow. Trained models track individual flies using three keypoints (head, mid, tail) and export trajectories for downstream analysis.

---

## Features

- **Manual pose annotation** — CSV-based labels with head / mid / tail keypoints per fly and frame
- **Per-video overfitting** — Train a dedicated YOLO Pose model for each recording (high accuracy on a single arena)
- **Foundation model training** — Merge all annotated videos, split by video ID (hold-out validation), augment from TIFF frames, and train one shared model
- **Video inference & tracking** — YOLO Pose + DeepSORT-style ID assignment; output GIF, MOV, and trajectory CSV
- **Behavior analysis** — Scripts to summarize contacts, interactions, and reinforcement-learning experiments

---

## Repository layout

```
DOLO/
├── scripts/                    # Main code (run from here)
│   ├── functions.py            # Annotation → YOLO format, augmentation, plotting
│   ├── functions_deepsort.py   # Tracking & GIF export
│   ├── overfitting_pipeline.py # Single-video train/val split & training
│   ├── foundation_pipeline.py  # Multi-video foundation pipeline
│   ├── foundation_train.sh     # Shell entry point for foundation training
│   ├── drosophila_predict.py   # Inference on new videos
│   └── README_foundation.md    # Detailed foundation pipeline docs (Japanese)
├── annotations/                # Not in git — labels, TIFFs, weights (local)
├── video/                      # Not in git — raw recordings (local)
└── LICENSE                     # AGPL-3.0
```

Large artifacts (videos, annotations, model weights, caches) are excluded via `.gitignore`. Clone the repo and place your data under `annotations/` and `video/` locally.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU(s) recommended for training
- Typical stack: `ultralytics`, `opencv-python`, `pandas`, `imgaug`, `torch`, and dependencies used in `functions.py` / `functions_deepsort.py`

Pretrained YOLO Pose weights (e.g. `yolo11x-pose.pt`) are **not** included; download or place them under `scripts/` before training.

---

## Quick start

All commands below assume the working directory is `scripts/`:

```bash
cd scripts
```

### 1. Per-video overfitting (one recording)

Prepare under `annotations/overfittings/`:

- `csvs/annotation_<video_id>_manual.csv`
- `tiffs/<video_id>/<frame_idx>.tif`

Then train (example):

```bash
python3 overfitting_pipeline.py \
  --unique_name whi-DM_6 \
  --load_model_path ./yolo11x-pose.pt \
  --gpu1 0 --gpu2 1
```

Or use the shell wrappers: `overfit_train.sh`, `overfit_single.sh`.

### 2. Predict & track on a new video

```bash
python3 drosophila_predict.py \
  --video_path ../video/whi-DM/whi-DM_6.avi \
  --output_gif_path ../annotations/overfittings/gifs/overfit_whi-DM_6.gif \
  --output_mov_path ../annotations/overfittings/movs/overfit_whi-DM_6.mov \
  --output_csv_path ./csvs/trajectory/whi-DM_6_overfit.csv \
  --model_path ../annotations/overfittings/overfits_weights/whi-DM_6/weights/best.pt \
  --start_frame 0 --end_frame 3000 --max_id 6
```

Shell examples: `predict_drosophila.sh`, `predict_drosophila_new.sh`.

---

## Annotation format

Manual CSVs must include at least:

| Column | Description |
|--------|-------------|
| `frame_idx` | Frame index in the video |
| `Head.x`, `Head.y` | Head keypoint |
| `mid.x`, `mid.y` | Mid-body keypoint |
| `Tail.x`, `Tail.y` | Tail keypoint |

Multiple flies per frame are supported (multiple rows with the same `frame_idx`).

---

## Training notes

- **Image caching** — On the first epoch, Ultralytics may log `Caching images…` while it builds a disk/RAM cache. This is one-time overhead; later epochs load faster.
- **Multi-GPU** — Passing two GPU IDs enables DDP. Long runs can hit NCCL timeouts; use `GPU_IDS="0"` or `--resume` to continue from `annotations/foundation/weights/<run_name>/weights/last.pt`.
- **`datasets_dir`** — Some YAML paths assume `/cellpose/scripts` as Ultralytics `datasets_dir`; adjust `--datasets_dir` or paths if your install differs.

---

## Branches

- **`dev`** — Active development (foundation pipeline, tracking updates)
- **`main`** — Stable snapshot

---

## License

This project is licensed under **AGPL-3.0** — see [LICENSE](LICENSE).

---

## Citation

If you use this code in published work, please cite the associated lab publication (to be added) and acknowledge use of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
