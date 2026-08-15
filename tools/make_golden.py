#!/usr/bin/env python3
"""ゴールデン回帰テスト用の基準CSVを、**現行（リファクタ前）のコード**で生成する。

これが「リファクタしても壊れていない」ことの基準になるので、必ず
`scripts/functions_deepsort.py` をいじる前に一度実行しておくこと。

使い方（リポジトリのルートで）::

    python3 tools/make_golden.py

    # デバイスや個体数を指定する場合
    python3 tools/make_golden.py --device mps --max-id 6

macOS では CUDA が無いので `--device` を省略すると mps → cpu の順に自動選択する。
（現行コードは device="cuda:3" 決め打ちだが、ここでは引数で上書きしている。）
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DATA / "sample.mov")
    parser.add_argument("--model", type=Path, default=DATA / "model.pt")
    parser.add_argument("--out", type=Path, default=DATA / "golden_trajectory.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-id", type=int, default=6)
    parser.add_argument("--conf", type=float, default=1e-3)
    parser.add_argument("--iou-thr", type=float, default=0.45)
    parser.add_argument("--max-missing-frames", type=int, default=15)
    parser.add_argument("--dist-thresh", type=float, default=30.0)
    parser.add_argument("--head-tail-jump-thresh", type=float, default=50.0)
    parser.add_argument("--overlap-thresh", type=float, default=5.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--keep-visuals",
        action="store_true",
        help="GIF/MOV も tests/data/ に残す（既定では捨てる）",
    )
    args = parser.parse_args()

    for path, label in ((args.video, "動画"), (args.model, "重み")):
        if not path.exists():
            print(f"エラー: {label}が見つかりません: {path}", file=sys.stderr)
            print("tests/data/README.md を参照してください。", file=sys.stderr)
            return 1

    from dolo.device import describe_devices, resolve_device

    print(describe_devices())
    device = resolve_device(args.device)
    print(f"\n使用デバイス: {device}")

    # 現行の実装をそのまま呼ぶ（ここを変えると基準にならない）
    from functions_deepsort import process_video_to_gif_with_angles

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        visual_dir = DATA if args.keep_visuals else Path(tmp)
        gif_path = visual_dir / "golden_preview.gif"
        mov_path = visual_dir / "golden_preview.mov"

        print(f"追跡中: {args.video.name} → {args.out}")
        process_video_to_gif_with_angles(
            video_path=str(args.video),
            output_gif_path=str(gif_path),
            output_mov_path=str(mov_path),
            output_csv_path=str(args.out),
            model_path=str(args.model),
            conf_thres=args.conf,
            iou_thres=args.iou_thr,
            frame_skip=1,
            device=device,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_age=args.max_missing_frames,
            max_ids=args.max_id,
            n_init=2,
            dist_thresh=args.dist_thresh,
            head_tail_jump_thresh=args.head_tail_jump_thresh,
            overlap_thresh=args.overlap_thresh,
        )

    meta = {
        "max_id": args.max_id,
        "conf": args.conf,
        "iou_thr": args.iou_thr,
        "max_missing_frames": args.max_missing_frames,
        "dist_thresh": args.dist_thresh,
        "head_tail_jump_thresh": args.head_tail_jump_thresh,
        "overlap_thresh": args.overlap_thresh,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "device": device,
        "video": args.video.name,
        "model": args.model.name,
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    import pandas as pd

    df = pd.read_csv(args.out)
    print(f"\n完了: {len(df)} 行, ID={sorted(df['ID'].unique().tolist())}")
    print(f"  {args.out}")
    print(f"  {DATA / 'meta.json'}")
    print("\n次に `python3 -m pytest tests/ -q` を実行して全て通ることを確認してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
