# DOLO

**D**rosophila tracking with **YOLO** **P**ose — a pipeline for pose estimation, multi-fly tracking, and behavioral analysis in *Drosophila* videos.

DOLO combines [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose/) with custom annotation tools, per-video fine-tuning, and a multi-video foundation training workflow. Trained models track individual flies using three keypoints (head, mid, tail) and export trajectories for downstream analysis.

---

## Features

- **Browser GUI** — Upload a video, run the default model, monitor progress, inspect per-ID metrics, and download CSV / JSONL / annotated video / GIF outputs
- **Manual pose annotation** — CSV-based labels with head / mid / tail keypoints per fly and frame
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

> **Where to push:** Active work (including this packaging / GUI phase) goes to **`dev`**, not `main`.
> Open a PR into `dev`. Promote `dev` → `main` only when the snapshot is stable enough to treat as release-ready.

---

## Model weights (not in Git)

`*.pt` / `*.onnx` / `*.engine` are gitignored. GitHub rejects files over **100 MB**, and the default
checkpoint (`best.pt`, ~120 MB) cannot be pushed as a normal Git blob.

**Download the default assets from Zenodo:**

- DOI: [10.5281/zenodo.21951363](https://doi.org/10.5281/zenodo.21951363)
- Files: `best.pt` (default model weights) and `test_movie.mov` (demo / test video)

After download, place `best.pt` at the repository root (or under `models/default.pt`, or set
`DOLO_MODEL_PATH`). Keep `test_movie.mov` anywhere convenient for local GUI / CLI trials — it is not
required for unit tests in CI.

Recommended workflow:

1. Keep weights **out of commits** (do not `git add -f best.pt`).
2. Place a local default at repo root `best.pt`, or `models/default.pt`, or set `DOLO_MODEL_PATH`.
3. Prefer the Zenodo deposit above as the citable source; optionally mirror the same files on a
   [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github) for convenience.

Git LFS can store large files in theory, but free quotas are easy to exhaust for ~100 MB+ models;
Zenodo (or Release assets) is a better fit for scientific checkpoints.

---

## GUI quick start

The GUI runs locally in a browser and uses the same Python tracking API as the CLI. It also works on a
remote GPU server through SSH port forwarding.

### 1. Install uv

On macOS or Linux, install `uv` with the official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell if the installer asks you to, then verify the command is available:

```bash
uv --version
```

On macOS, Homebrew is an alternative:

```bash
brew install uv
```

See the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for
Windows and other installation methods.

### 2. Create the Python environment

From the repository root, install Python 3.10 and create the project virtual environment:

```bash
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate
```

### 3. Install DOLO and the GUI

```bash
# Choose one platform lock file:
uv pip install -r requirements/lock-macos.txt       # macOS
# uv pip install -r requirements/lock-linux.txt     # NVIDIA Linux server

uv pip install -r requirements/gui.in
uv pip install --no-deps -e .
```

### 4. Check the environment and launch

```bash
dolo doctor
dolo gui
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080), choose a video in the popup, review the detected
default model, and press **推論を開始**. Every run is saved under `.dolo/runs/` with:

- trajectory CSV and frame-preserving JSON Lines;
- optional annotated MOV and GIF;
- per-ID distance, coverage, and confidence metrics in the GUI;
- `run.json`, containing the model path, tracking settings, summary, and output inventory.

#### If port 8080 is already in use

If startup fails with the following message, another process is already listening on port 8080:

```text
ERROR: [Errno 48] error while attempting to bind on address ('127.0.0.1', 8080): address already in use
```

The simplest solution is to launch DOLO on another port and open the corresponding URL:

```bash
dolo gui --port 8090
```

Then open [http://127.0.0.1:8090](http://127.0.0.1:8090). If an earlier DOLO GUI is still running,
return to its terminal and press `Ctrl+C`. On macOS or Linux, you can identify the process using port
8080 with:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

After confirming that the displayed PID belongs to the process you intend to stop, terminate it normally:

```bash
kill <PID>
```

Run `dolo gui` again after the port has been released. Do not stop an unfamiliar process; use another
port instead.

Default weights are resolved in this order: `--model`, `DOLO_MODEL_PATH`, repository `best.pt`,
`models/default.pt`, then the DOLO data/cache directories. Model weights are intentionally not committed
to Git. See [docs/GUI.md](docs/GUI.md) for remote-server, Docker, storage, and troubleshooting details.

> The GUI binds to `127.0.0.1` by default. Do not expose it directly to the public internet; use SSH
> forwarding or an authenticated reverse proxy.

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

| Branch | Role |
|--------|------|
| **`feature/*`** | Isolated implementation work (e.g. `feature/phase0-packaging`) |
| **`dev`** | Integration branch — **push / open PRs here** while the project is pre-release |
| **`main`** | Stable, release-ready snapshot only |

Flow: `feature/*` → PR into **`dev`** → after review and green CI, merge **`dev` → `main`**.
Direct pushes to `main` are discouraged. If you are unsure which branch to use, choose **`dev`**.

---

## Project provenance

DOLO (*Drosophila* tracking with YOLO) was developed as the behavioral tracking software associated
with the submitted manuscript:

> **Chemosensory input suppresses cannibalism by stabilizing social feeding boundaries in Drosophila
> larvae**<br>
> Nagisa Matsuda-Watanabe¹, Masato Tsutsumi²˒³, Misako Okumura¹˒⁴˒⁵, and Takahiro Chihara¹˒⁴\*

Software design and implementation: **Masato Tsutsumi**
([mtsutsumi@nagoya-u.jp](mailto:mtsutsumi@nagoya-u.jp)). DOLO is an original software work by Masato
Tsutsumi unless a file or bundled third-party component states otherwise.

For current publication status, see [Masato Tsutsumi's research page](https://masa10223.github.io/en/).

---

## Copyright and license

Copyright © 2024–2026 **Masato Tsutsumi**. All rights reserved except for the permissions granted under
the open-source license below.

The DOLO source code and original documentation are licensed under the **GNU Affero General Public
License v3.0 or later (AGPL-3.0-or-later)** — see [LICENSE](LICENSE). Third-party software, model weights,
datasets, publications, and trademarks remain subject to their respective licenses and rights. Citation
of the associated manuscript is academically requested but does not replace or modify the software
license.

---

## Citation

If you use DOLO in published work, please cite the associated manuscript above and acknowledge use of
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics). A DOI or journal-formatted citation will
be added when the manuscript record becomes publicly available.
