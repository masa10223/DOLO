[**English**](README.md) | [日本語](README.ja.md)

# DOLO

**D**rosophila tracking with **YOLO** **P**ose — software for pose estimation, multi-fly tracking, and behavioral analysis in *Drosophila* videos.

Use the browser GUI to pick a video, run a trained model, and export trajectory CSVs / annotated videos.  
This README is written so people who are **new to the terminal and Git** can get a first run working.

Associated manuscript (submitted):

> **Chemosensory input suppresses cannibalism by stabilizing social feeding boundaries in Drosophila larvae**  
> Nagisa Matsuda-Watanabe, Masato Tsutsumi, Misako Okumura, and Takahiro Chihara

Japanese lab members: see **[README.ja.md](README.ja.md)** for the same guide in Japanese.

---

## What you will do (overview)

1. Install **Git** and **uv** (a convenient Python installer) on your computer  
2. **`git clone`** the DOLO program from GitHub  
3. Download **`best.pt` (model weights)** and **`test_movie.mov` (demo video)** from Zenodo  
4. Place those files in the correct location inside the cloned folder  
5. Install dependencies and open the browser UI with **`dolo gui`**

Time estimate: about 15–40 minutes the first time (network and machine dependent).  
`best.pt` is about **120 MB**.

Supported OS: **macOS** or **Linux** (Windows is not supported yet; WSL2 is at your own risk).

---

## 0. Tiny glossary

| Term | Meaning |
|------|---------|
| **Terminal** | App where you type commands (on macOS: Terminal) |
| **Repository** | The project on GitHub; cloning copies it to your PC |
| **clone** | Download the project from the internet to your computer |
| **Weights (`best.pt`)** | Trained model file — required for tracking |
| **Virtual environment (`.venv`)** | A project-local Python install that will not mix with other projects |

Copy commands **one line at a time** and press Enter. If something fails, save the full error text.

---

## 1. Install Git (only if needed)

### macOS

Open Terminal and run:

```bash
git --version
```

If you see `git version ...`, you are done.  
If macOS asks to install Command Line Developer Tools, click **Install**, then run the same command again.

With Homebrew:

```bash
brew install git
```

### Linux (Ubuntu example)

```bash
sudo apt update
sudo apt install -y git
git --version
```

---

## 2. `git clone` DOLO

Move to a folder you like (home directory example), then:

```bash
cd ~
git clone https://github.com/masa10223/DOLO.git
cd DOLO
```

This creates a `DOLO` folder with the full project.  
Run later commands **from inside this `DOLO` folder**.

Check where you are:

```bash
pwd
ls
```

`pwd` should end with `.../DOLO`, and `ls` should show `README.md` and `dolo`.

> **Updating later**  
> In the same folder, run `git pull` to fetch newer changes from GitHub.

---

## 3. Download weights and the demo video from Zenodo

Model weights are **not** stored on GitHub (the file is too large).  
Download them from this Zenodo record:

- DOI: **[10.5281/zenodo.21951363](https://doi.org/10.5281/zenodo.21951363)**  
- Page: https://doi.org/10.5281/zenodo.21951363  
- Files included:
  - **`best.pt`** — default trained weights (**required**)
  - **`test_movie.mov`** — short demo video (**recommended**)

### Browser steps (recommended)

1. Open the DOI link above  
2. Find **Files** on the page  
3. Download `best.pt` (~120 MB)  
4. Download `test_movie.mov` (~1.2 MB)  
5. Files usually land in your Downloads folder

### Where to put the files (most important step)

Open the cloned `DOLO` folder in Finder / your file manager.

| Downloaded file | Put it here | Final path |
|-----------------|-------------|------------|
| `best.pt` | **Top level of `DOLO`** (same level as `README.md`) | `DOLO/best.pt` |
| `test_movie.mov` | Same top level is fine | `DOLO/test_movie.mov` |

Terminal example (macOS, moving from Downloads):

```bash
cd ~/DOLO
mv ~/Downloads/best.pt .
mv ~/Downloads/test_movie.mov .
ls -lh best.pt test_movie.mov
```

If `best.pt` shows roughly **120M**, placement succeeded.

> **If you skip this**  
> The GUI may start, but tracking will not run until a model is found.  
> Check with `dolo doctor`.

Advanced options:

- place as `models/default.pt`, or  
- set `DOLO_MODEL_PATH=/absolute/path/to/best.pt`

---

## 4. Install uv

DOLO uses [uv](https://docs.astral.sh/uv/) to install Python and libraries.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal (or run the `source ...` line the installer prints), then:

```bash
uv --version
```

On macOS with Homebrew:

```bash
brew install uv
```

Other platforms: [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

---

## 5. Install DOLO

From the `DOLO` folder:

```bash
cd ~/DOLO
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate
```

A `(.venv)` prefix in the prompt means the virtual environment is active.

Then run **one** of the following blocks.

**macOS (Apple Silicon / Intel):**

```bash
uv pip install -r requirements/lock-macos.txt
uv pip install -r requirements/gui.in
uv pip install --no-deps -e .
```

**NVIDIA Linux server:**

```bash
uv pip install -r requirements/lock-linux.txt
uv pip install -r requirements/gui.in
uv pip install --no-deps -e .
```

The first install can take several minutes.

### Sanity check

```bash
dolo doctor
```

`READY` means you are set.  
If you see `SETUP REQUIRED`, fix the listed items (especially missing `best.pt`).

> After closing the terminal, activate again before using the GUI:  
> `cd ~/DOLO` → `source .venv/bin/activate`

---

## 6. Launch the GUI and try a run

```bash
cd ~/DOLO
source .venv/bin/activate
dolo gui
```

If the browser does not open, go to:  
http://127.0.0.1:8080

1. Choose a video (`test_movie.mov` for a first try)  
2. Confirm the default model shows `best.pt`  
3. Press **Start inference** / **推論を開始**  

Outputs are saved under `.dolo/runs/` (trajectory CSV, optional video/GIF, run metadata).

### Troubleshooting

**Port 8080 already in use**

```text
address already in use
```

Use another port:

```bash
dolo gui --port 8090
```

Then open http://127.0.0.1:8090

**Model not found**

- Confirm `DOLO/best.pt` exists, or  
- pass an explicit path: `dolo gui --model /full/path/best.pt`

**Stop the GUI**

Press `Ctrl+C` in the terminal where it is running.

More detail: [docs/GUI.md](docs/GUI.md), [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

> By default the GUI listens only on your machine (`127.0.0.1`). Do not expose it to the public internet.

---

## Features (short)

- **Browser GUI** — upload, track, monitor, download results  
- **Manual annotation hooks** — head / mid / tail CSV labels  
- **Training pipelines** — per-video fine-tune and multi-video foundation training (advanced)  
- **Downstream analysis scripts** — contacts / interactions, etc.

Training and large batches belong on a GPU server. Start with the GUI + `test_movie.mov`.

---

## FAQ

**Q. Why not put `best.pt` on GitHub?**  
A. GitHub rejects ordinary pushes of very large files (especially over 100 MB). We host them on Zenodo.

**Q. I cloned the repo but there is no `best.pt`**  
A. Expected. Download it from Zenodo and place it at the top of `DOLO`.

**Q. Can I use my own experimental videos?**  
A. Yes, select them in the GUI. Very different imaging conditions may need fine-tuning.

**Q. Which branch should developers target?**  
A. Open PRs into **`dev`** while actively developing; promote to **`main`** when stable. If unsure, use `dev`.

---

## For developers (short)

```
feature/*  →  PR →  dev  →  (when stable)  main
```

Do not commit large artifacts (videos, annotations, `*.pt`); they are gitignored.

Legacy training/inference scripts live under `scripts/`. The packaged API / GUI live under `dolo/`.

---

## Citation

Please cite the associated manuscript when published, and acknowledge [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). A DOI / journal citation will be added when available.

Model weights and demo video archive:

> Tsutsumi, M. (2026). Test movie and pre-trained model weight for DOLO. Zenodo.  
> https://doi.org/10.5281/zenodo.21951363

---

## Copyright and license

Copyright © 2024–2026 **Masato Tsutsumi**.  
Source code and original documentation are **AGPL-3.0-or-later** ([LICENSE](LICENSE)).  
Third-party software, model weights, datasets, publications, and trademarks remain under their own terms.

Software design and implementation: **Masato Tsutsumi** ([mtsutsumi@nagoya-u.jp](mailto:mtsutsumi@nagoya-u.jp))  
Updates: [research page](https://masa10223.github.io/en/)
