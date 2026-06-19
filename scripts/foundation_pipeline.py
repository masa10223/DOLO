import argparse
import random
import re
import shutil
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from functions import create_yolo_annotations_from_images
from ultralytics import YOLO, settings
from ultralytics.nn.tasks import torch_safe_load


REQUIRED_COLUMNS = [
    "frame_idx",
    "Head.x",
    "Head.y",
    "mid.x",
    "mid.y",
    "Tail.x",
    "Tail.y",
]


@dataclass(frozen=True)
class FoundationPaths:
    foundation_root: Path
    run_name: str

    @property
    def manifests_dir(self) -> Path:
        return self.foundation_root / "manifests"

    @property
    def tiffs_dir(self) -> Path:
        return self.foundation_root / "tiffs"

    @property
    def csvs_dir(self) -> Path:
        return self.foundation_root / "csvs"

    @property
    def train_annotations_root(self) -> Path:
        return self.foundation_root / "train_annotations"

    @property
    def val_annotations_root(self) -> Path:
        return self.foundation_root / "val_annotations"

    @property
    def train_annotations_dir(self) -> Path:
        return self.train_annotations_root / self.run_name

    @property
    def val_annotations_dir(self) -> Path:
        return self.val_annotations_root / self.run_name

    @property
    def yamls_dir(self) -> Path:
        return self.foundation_root / "yamls"

    @property
    def yaml_path(self) -> Path:
        return self.yamls_dir / f"{self.run_name}.yaml"

    @property
    def weights_root(self) -> Path:
        return self.foundation_root / "weights"

    @property
    def weights_dir(self) -> Path:
        return self.weights_root / self.run_name

    def last_pt_path(self) -> Path:
        """Ultralytics saves under <run_dir>/weights/last.pt, not <run_dir>/last.pt."""
        nested = self.weights_dir / "weights" / "last.pt"
        flat = self.weights_dir / "last.pt"
        if nested.exists():
            return nested
        if flat.exists():
            return flat
        return nested

    @property
    def logs_dir(self) -> Path:
        return self.foundation_root / "logs"

    @property
    def all_samples_csv(self) -> Path:
        return self.manifests_dir / "all_samples.csv"

    @property
    def all_videos_txt(self) -> Path:
        return self.manifests_dir / "all_videos.txt"

    @property
    def train_videos_txt(self) -> Path:
        return self.manifests_dir / "train_videos.txt"

    @property
    def val_videos_txt(self) -> Path:
        return self.manifests_dir / "val_videos.txt"


def ensure_layout(paths: FoundationPaths) -> None:
    dirs = [
        paths.foundation_root,
        paths.manifests_dir,
        paths.tiffs_dir,
        paths.csvs_dir,
        paths.train_annotations_root,
        paths.val_annotations_root,
        paths.yamls_dir,
        paths.weights_root,
        paths.logs_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def extract_video_id(csv_path: Path) -> str:
    matched = re.match(r"annotation_(.+)_manual\.csv$", csv_path.name)
    if matched is None:
        raise ValueError(f"Unexpected CSV filename: {csv_path}")
    return matched.group(1)


def discover_csvs(annotations_root: Path) -> List[Path]:
    csvs = sorted(annotations_root.glob("**/csvs/annotation_*_manual.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No annotation CSVs found under: {annotations_root.as_posix()}"
        )
    return csvs


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")


def build_manifest(paths: FoundationPaths, annotations_root: Path) -> pd.DataFrame:
    csv_paths = discover_csvs(annotations_root)
    rows = []

    for csv_path in csv_paths:
        video_id = extract_video_id(csv_path)
        df = pd.read_csv(csv_path)
        validate_columns(df, csv_path)
        df = df.dropna(subset=REQUIRED_COLUMNS).copy()
        df["frame_idx"] = df["frame_idx"].astype(int)
        source_csv = csv_path.resolve().as_posix()

        for _, row in df.iterrows():
            frame_idx = int(row["frame_idx"])
            tif_path = (paths.tiffs_dir / video_id / f"{frame_idx}.tif").resolve()
            rows.append(
                {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "source_csv": source_csv,
                    "source_tif": tif_path.as_posix(),
                    "Head.x": float(row["Head.x"]),
                    "Head.y": float(row["Head.y"]),
                    "mid.x": float(row["mid.x"]),
                    "mid.y": float(row["mid.y"]),
                    "Tail.x": float(row["Tail.x"]),
                    "Tail.y": float(row["Tail.y"]),
                }
            )

    if not rows:
        raise ValueError("No valid rows found after validating required columns.")

    manifest_df = pd.DataFrame(rows)
    manifest_df = manifest_df.sort_values(
        by=["video_id", "frame_idx"], kind="stable"
    ).reset_index(drop=True)
    manifest_df.to_csv(paths.all_samples_csv, index=False)

    video_ids = sorted(manifest_df["video_id"].unique().tolist())
    paths.all_videos_txt.write_text("\n".join(video_ids) + "\n", encoding="utf-8")

    print(f"Wrote manifest: {paths.all_samples_csv}")
    print(f"Wrote video list: {paths.all_videos_txt}")
    print(f"Rows={len(manifest_df)}, videos={len(video_ids)}")
    return manifest_df


def split_videos(
    paths: FoundationPaths, manifest_df: pd.DataFrame, val_ratio: float, seed: int
) -> None:
    video_ids = sorted(manifest_df["video_id"].unique().tolist())
    rng = random.Random(seed)
    shuffled = video_ids[:]
    rng.shuffle(shuffled)

    n_videos = len(shuffled)
    if n_videos == 1:
        val_count = 0
    else:
        val_count = int(round(n_videos * val_ratio))
        val_count = max(1, min(val_count, n_videos - 1))

    val_videos = sorted(shuffled[:val_count])
    train_videos = sorted(shuffled[val_count:])

    paths.train_videos_txt.write_text(
        "\n".join(train_videos) + ("\n" if train_videos else ""), encoding="utf-8"
    )
    paths.val_videos_txt.write_text(
        "\n".join(val_videos) + ("\n" if val_videos else ""), encoding="utf-8"
    )

    print(f"Wrote train videos: {paths.train_videos_txt} ({len(train_videos)})")
    print(f"Wrote val videos: {paths.val_videos_txt} ({len(val_videos)})")


def _merge_temp_into_final(
    temp_dir: Path, final_dir: Path, video_id: str
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for f in temp_dir.iterdir():
        if f.suffix not in (".jpg", ".txt"):
            continue
        new_name = f"{video_id}_{f.name}"
        shutil.copy2(f, final_dir / new_name)


def build_yolo_data(
    paths: FoundationPaths,
    manifest_df: pd.DataFrame,
    tiffs_root: Path,
    target_size: int,
) -> None:
    train_list = [
        s.strip()
        for s in paths.train_videos_txt.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    val_list = [
        s.strip()
        for s in paths.val_videos_txt.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    video_to_csv = (
        manifest_df.drop_duplicates("video_id")[["video_id", "source_csv"]]
        .set_index("video_id")["source_csv"]
        .to_dict()
    )
    tiffs_root = tiffs_root.resolve()

    temp_base = paths.foundation_root / "_temp_yolo"
    if temp_base.exists():
        shutil.rmtree(temp_base)
    for final_dir in (paths.train_annotations_dir, paths.val_annotations_dir):
        if final_dir.exists():
            shutil.rmtree(final_dir)

    def process_split(
        video_ids: List[str], out_dir: Path, augment: bool, label: str
    ) -> None:
        for video_id in video_ids:
            csv_path = Path(video_to_csv[video_id])
            image_dir = tiffs_root / video_id
            if not image_dir.exists():
                print(f"Warning: TIFF dir not found {image_dir}, skipping {video_id}")
                continue
            df = pd.read_csv(csv_path)
            validate_columns(df, csv_path)
            df = df.dropna(subset=REQUIRED_COLUMNS).copy()
            df["frame_idx"] = df["frame_idx"].astype(int)
            temp_out = temp_base / label / video_id
            temp_out.mkdir(parents=True, exist_ok=True)
            create_yolo_annotations_from_images(
                df,
                str(image_dir),
                str(temp_out),
                augment=augment,
                target_size=target_size,
            )
            _merge_temp_into_final(temp_out, out_dir, video_id)
        print(f"Built {label}: {out_dir} ({len(list(out_dir.glob('*.jpg')))} images)")

    process_split(train_list, paths.train_annotations_dir, augment=True, label="train")
    process_split(val_list, paths.val_annotations_dir, augment=False, label="val")
    if temp_base.exists():
        shutil.rmtree(temp_base)


def _scalar_ckpt_epoch(ckpt: dict) -> int:
    """Match ultralytics/engine/trainer.py: ckpt.get('epoch', -1) semantics."""
    e: Any = ckpt.get("epoch")
    if e is None:
        return -1
    if hasattr(e, "item"):
        return int(e.item())
    return int(e)


def _train_args_epochs(ckpt: dict, fallback: int) -> int:
    ta: Any = ckpt.get("train_args")
    if ta is None:
        return fallback
    if isinstance(ta, dict):
        return int(ta.get("epochs", fallback))
    v = getattr(ta, "epochs", None)
    return int(v) if v is not None else fallback


def _last_epoch_from_results_csv(weights_dir: Path) -> Optional[int]:
    """Fallback when ckpt['epoch'] is missing (Ultralytics results.csv, 1-based epoch column)."""
    p = weights_dir / "results.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty or "epoch" not in df.columns:
            return None
        return int(df["epoch"].iloc[-1])
    except (OSError, ValueError, KeyError):
        return None


def _cache_arg(cache_mode: str):
    if cache_mode.lower() == "false":
        return False
    return cache_mode


def _resolved_micro_batch(arguments: argparse.Namespace) -> Optional[int]:
    """Effective micro_batch for Ultralytics (None = forward batch == --batch, no nbs override)."""
    if getattr(arguments, "no_micro_batch", False):
        return None
    if arguments.micro_batch is not None:
        return int(arguments.micro_batch)
    eff = int(arguments.batch)
    # YOLO26x-pose + DDP: full global batch often OOMs; halve forward batch when user did not opt out.
    if len(arguments.gpu) >= 2 and eff >= 64 and eff % 2 == 0:
        return eff // 2
    return None


def _ultralytics_batch_nbs(
    arguments: argparse.Namespace,
) -> tuple[int, Optional[int]]:
    """Return (forward_batch, nbs_or_none).

    Ultralytics sets accumulate ≈ round(nbs / batch). Using forward batch B_f < B_eff with nbs=B_eff
    lowers per-step VRAM while keeping loss scaling consistent with training at global batch B_eff.
    """
    eff = int(arguments.batch)
    mb = _resolved_micro_batch(arguments)
    if mb is None or mb == eff:
        return eff, None
    if mb < 1 or mb > eff:
        raise ValueError(
            f"--micro_batch must satisfy 1 <= micro_batch <= --batch; got micro_batch={mb}, batch={eff}"
        )
    return mb, eff


def _yolo_train_common_kwargs(
    arguments: argparse.Namespace, paths: FoundationPaths
) -> dict:
    fb, nbs = _ultralytics_batch_nbs(arguments)
    if nbs is not None:
        acc = max(round(nbs / fb), 1)
        auto = arguments.micro_batch is None and not getattr(
            arguments, "no_micro_batch", False
        )
        tag = " (auto split for multi-GPU + batch>=64)" if auto else ""
        print(
            f"Ultralytics train: forward batch={fb}, nbs={nbs}{tag}, "
            f"accumulate≈{acc} (effective batch scale {arguments.batch})."
        )
    out: dict = {
        "batch": fb,
        "device": list(arguments.gpu),
        "project": str(paths.weights_root),
        "exist_ok": True,
        "cache": _cache_arg(arguments.cache_mode),
        "deterministic": False,
        "workers": arguments.workers,
        "save_period": arguments.save_period,
        "fraction": arguments.fraction,
    }
    if nbs is not None:
        out["nbs"] = nbs
    return out


def write_foundation_yaml(paths: FoundationPaths) -> None:
    data = {
        "train": str(paths.train_annotations_dir.resolve()),
        "val": str(paths.val_annotations_dir.resolve()),
        "nc": 1,
        "names": ["pose"],
        "kpt_shape": [3, 3],
        "kpts": 3,
    }
    paths.yamls_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Wrote YAML: {paths.yaml_path}")


def train_foundation(arguments: argparse.Namespace, paths: FoundationPaths) -> None:
    yaml_path = Path(arguments.data_yaml) if arguments.data_yaml else paths.yaml_path
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"YAML not found: {yaml_path}. Build YOLO dataset/yaml before training."
        )

    settings.update({"datasets_dir": arguments.datasets_dir})
    last_pt = paths.last_pt_path()
    do_resume = arguments.resume and last_pt.exists()
    run_name_for_train = arguments.run_name

    if do_resume:
        ckpt, _ = torch_safe_load(str(last_pt))
        raw_epoch = _scalar_ckpt_epoch(ckpt)
        ckpt_missing_epoch = raw_epoch < 0
        if ckpt_missing_epoch:
            csv_ep = _last_epoch_from_results_csv(paths.weights_dir)
            if csv_ep is not None:
                raw_epoch = csv_ep - 1
                print(
                    f"Note: checkpoint missing 'epoch'; inferred raw epoch {raw_epoch} from results.csv (last epoch {csv_ep})."
                )
        start_epoch = raw_epoch + 1
        ckpt_total = _train_args_epochs(ckpt, arguments.epochs)
        del ckpt

        if ckpt_missing_epoch:
            if not arguments.resume_allow_stripped:
                raise RuntimeError(
                    f"Checkpoint {last_pt} has no usable epoch (raw={raw_epoch}). "
                    f"The file may be weights-only or corrupt. Ultralytics needs a full training checkpoint to resume.\n"
                    f"  → To continue from these weights: RESUME=0 LOAD_MODEL_PATH={last_pt}\n"
                    f"  → To emulate resume from results.csv: add --resume_allow_stripped"
                )
            csv_ep = _last_epoch_from_results_csv(paths.weights_dir)
            if csv_ep is None:
                raise RuntimeError(
                    f"Checkpoint {last_pt} has no usable epoch and {paths.weights_dir / 'results.csv'} not found. "
                    f"Cannot emulate resume. Use RESUME=0 with LOAD_MODEL_PATH={last_pt}."
                )
            start_epoch = csv_ep
            if arguments.epochs <= start_epoch:
                raise RuntimeError(
                    f"Requested --epochs={arguments.epochs} but results.csv indicates epoch {start_epoch} already done. "
                    f"Set EPOCHS higher than {start_epoch}."
                )
            # Ultralytics strips optimizer/epoch at final_eval. For this case we emulate resume:
            # load last.pt as weights, disable true resume, and train for remaining epochs only.
            remaining_epochs = arguments.epochs - start_epoch
            do_resume = False
            load_path = str(last_pt)
            run_name_for_train = f"{arguments.run_name}{arguments.resume_stripped_run_suffix}"
            print(
                f"Resume emulation from stripped checkpoint: base_epoch={start_epoch}, "
                f"target_total={arguments.epochs}, running extra_epochs={remaining_epochs}, "
                f"save_dir name={run_name_for_train}"
            )
            model = YOLO(load_path)
            common = _yolo_train_common_kwargs(arguments, paths)
            model.train(
                data=str(yaml_path),
                epochs=remaining_epochs,
                name=run_name_for_train,
                resume=False,
                **common,
            )
            return
        if start_epoch <= 0:
            raise RuntimeError(
                f"Checkpoint {last_pt} has invalid start_epoch={start_epoch}. "
                f"Use RESUME=0 LOAD_MODEL_PATH={last_pt}."
            )
        if arguments.epochs <= start_epoch:
            raise RuntimeError(
                f"Requested --epochs={arguments.epochs} but checkpoint is already at start_epoch={start_epoch} "
                f"(Ultralytics convention: epoch field + 1). Set EPOCHS higher than {start_epoch} "
                f"(e.g. 150 to continue toward 150 total)."
            )
        load_path = str(last_pt)
        print(
            f"Resuming from checkpoint: {load_path} (Ultralytics start_epoch={start_epoch}, "
            f"checkpoint had total_epochs={ckpt_total}, requesting total_epochs={arguments.epochs})"
        )
    else:
        load_path = arguments.load_model_path
        if arguments.resume and not last_pt.exists():
            print(
                f"Warning: --resume set but no checkpoint at {paths.weights_dir / 'weights' / 'last.pt'} "
                f"(or legacy {paths.weights_dir / 'last.pt'}); starting from {load_path} (resume=False)"
            )

    model = YOLO(load_path)
    common = _yolo_train_common_kwargs(arguments, paths)
    model.train(
        data=str(yaml_path),
        epochs=arguments.epochs,
        name=run_name_for_train,
        resume=do_resume,
        **common,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Foundation model pipeline: layout/manifests/training"
    )
    parser.add_argument(
        "--mode",
        choices=["manifest", "yolo", "train", "all"],
        default="all",
        help="manifest: build manifests only; yolo: build YOLO data+yaml (manifests must exist); train: train only; all: manifest then yolo then train",
    )
    parser.add_argument("--run_name", type=str, default="foundation_v1")
    parser.add_argument(
        "--foundation_root",
        type=Path,
        default=Path("../annotations/foundation"),
        help="Base directory for foundation artifacts.",
    )
    parser.add_argument(
        "--annotations_root",
        type=Path,
        default=Path("../annotations"),
        help="Root directory to search annotation_*_manual.csv recursively.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_model_path", type=str, default="./yolo11x-pose.pt")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from weights/<run_name>/weights/last.pt (Ultralytics layout) if it exists.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch",
        type=int,
        default=20,
        help="Global batch for loss scaling. If --micro_batch is set and smaller, it becomes nbs (effective scale).",
    )
    parser.add_argument(
        "--micro_batch",
        type=int,
        default=None,
        help="Per-forward global batch (split across GPUs). With --batch 64 and --micro_batch 32, passes nbs=64. "
        "If omitted and --batch>=64 with 2+ GPUs, defaults to half of --batch (OOM guard).",
    )
    parser.add_argument(
        "--no_micro_batch",
        action="store_true",
        help="Disable automatic halving and use full --batch every forward (may OOM on YOLO26x + DDP).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Dataloader workers per process. Lower if CPU/I/O contention is high.",
    )
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="disk",
        choices=["disk", "ram", "False", "false"],
        help="Ultralytics cache mode. 'disk' speeds subsequent epochs but may bottleneck I/O on first epoch.",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=-1,
        help="Checkpoint save interval in epochs. Set 1 to keep epoch*.pt (useful for resume fallback).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Ultralytics fraction of dataset to use for training/val in [0,1].",
    )
    parser.add_argument(
        "--resume_allow_stripped",
        action="store_true",
        help="If checkpoint has stripped epoch/optimizer, emulate resume using results.csv and run remaining epochs.",
    )
    parser.add_argument(
        "--resume_stripped_run_suffix",
        type=str,
        default="_resume_from_stripped",
        help="Output run-name suffix used when resume emulation is activated.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0, 1],
        help="GPU ids for YOLO training. Example: --gpu 0 1",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="/cellpose/scripts",
        help="ultralytics settings.datasets_dir",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default=None,
        help="Optional explicit YAML path for training.",
    )
    parser.add_argument(
        "--tiffs_root",
        type=Path,
        default=Path("/inthdd/tsutsumi/drosophila/DOLO/annotations/overfittings/tiffs"),
        help="Root directory of TIFF images: <tiffs_root>/<video_id>/<frame_idx>.tif",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=100,
        help="Target number of augmented images per video for training (used only when augment=True).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.val_ratio < 0.0 or arguments.val_ratio > 1.0:
        raise ValueError("--val_ratio must be in [0, 1].")

    paths = FoundationPaths(
        foundation_root=arguments.foundation_root.resolve(),
        run_name=arguments.run_name,
    )
    ensure_layout(paths)

    manifest_df: Optional[pd.DataFrame] = None
    if arguments.mode in {"manifest", "all"}:
        manifest_df = build_manifest(paths, arguments.annotations_root.resolve())
        split_videos(paths, manifest_df, arguments.val_ratio, arguments.seed)

    if arguments.mode in {"yolo", "all"}:
        tiffs_root = arguments.tiffs_root.resolve()
        if manifest_df is None:
            if not paths.all_samples_csv.exists():
                raise FileNotFoundError(
                    f"Manifests not found: {paths.all_samples_csv}. Run with --mode manifest first."
                )
            manifest_df = pd.read_csv(paths.all_samples_csv)
        if not paths.train_videos_txt.exists() or not paths.val_videos_txt.exists():
            raise FileNotFoundError(
                "train_videos.txt or val_videos.txt not found. Run with --mode manifest first."
            )
        build_yolo_data(paths, manifest_df, tiffs_root, arguments.target_size)
        write_foundation_yaml(paths)

    if arguments.mode in {"train", "all"}:
        train_foundation(arguments, paths)


if __name__ == "__main__":
    main()
