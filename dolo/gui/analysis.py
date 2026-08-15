"""GUI の結果サマリー用の軽量集計。pandas 不要。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class IdMetrics:
    track_id: int
    visible_frames: int
    coverage: float
    total_distance_px: float
    mean_confidence: float
    mean_abs_angle: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunMetrics:
    rows: int
    unique_frames: int
    first_frame: int | None
    last_frame: int | None
    ids: tuple[IdMetrics, ...]

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "unique_frames": self.unique_frames,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "ids": [item.to_dict() for item in self.ids],
        }


def summarize_csv(path: str | Path, expected_frames: int | None = None) -> RunMetrics:
    """軌跡 CSV を1パスで集計する。

    CSV はフレーム順・ID順に出るため、全行をメモリに載せずに大きな結果も扱える。
    """
    counters: dict[int, dict[str, float | int | None]] = {}
    rows = 0
    unique_frames = 0
    first_frame = None
    last_frame = None
    previous_global_frame = None

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Frame", "ID", "DistMoved", "Confidence", "Angle"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"結果 CSV に必要な列がありません: {sorted(missing)}")

        for row in reader:
            frame = int(float(row["Frame"]))
            track_id = int(float(row["ID"]))
            rows += 1
            if frame != previous_global_frame:
                unique_frames += 1
                previous_global_frame = frame
            first_frame = frame if first_frame is None else min(first_frame, frame)
            last_frame = frame if last_frame is None else max(last_frame, frame)

            item = counters.setdefault(
                track_id,
                {"frames": 0, "last_frame": None, "distance": 0.0, "confidence": 0.0, "angle": 0.0},
            )
            if item["last_frame"] != frame:
                item["frames"] = int(item["frames"]) + 1
                item["last_frame"] = frame
            item["distance"] = float(item["distance"]) + float(row["DistMoved"] or 0.0)
            item["confidence"] = float(item["confidence"]) + float(row["Confidence"] or 0.0)
            item["angle"] = float(item["angle"]) + abs(float(row["Angle"] or 0.0))

    denominator = expected_frames or unique_frames or 1
    metrics = []
    for track_id, item in sorted(counters.items()):
        count = int(item["frames"])
        metrics.append(
            IdMetrics(
                track_id=track_id,
                visible_frames=count,
                coverage=min(1.0, count / denominator),
                total_distance_px=float(item["distance"]),
                mean_confidence=float(item["confidence"]) / count if count else 0.0,
                mean_abs_angle=float(item["angle"]) / count if count else 0.0,
            )
        )

    return RunMetrics(
        rows=rows,
        unique_frames=unique_frames,
        first_frame=first_frame,
        last_frame=last_frame,
        ids=tuple(metrics),
    )
