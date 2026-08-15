from __future__ import annotations

import csv

import pytest

from dolo.gui.analysis import summarize_csv
from dolo.results import CSV_COLUMNS


def _row(frame, track_id, distance, confidence, angle=10):
    return [frame, track_id, 1, 2, 3, 4, 5, 6, angle, distance, confidence, 0]


def test_summarize_csv_builds_per_id_metrics(tmp_path):
    path = tmp_path / "tracks.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerow(_row(1, 1, 2.5, 0.8, -10))
        writer.writerow(_row(1, 2, 1.0, 0.7, 20))
        writer.writerow(_row(2, 1, 3.5, 1.0, -30))

    metrics = summarize_csv(path, expected_frames=4)
    assert metrics.rows == 3
    assert metrics.unique_frames == 2
    assert (metrics.first_frame, metrics.last_frame) == (1, 2)
    assert len(metrics.ids) == 2
    first = metrics.ids[0]
    assert first.track_id == 1
    assert first.visible_frames == 2
    assert first.coverage == pytest.approx(0.5)
    assert first.total_distance_px == pytest.approx(6.0)
    assert first.mean_confidence == pytest.approx(0.9)
    assert first.mean_abs_angle == pytest.approx(20.0)


def test_summarize_empty_csv_is_safe(tmp_path):
    path = tmp_path / "tracks.csv"
    path.write_text(",".join(CSV_COLUMNS) + "\n")
    metrics = summarize_csv(path)
    assert metrics.rows == 0
    assert metrics.ids == ()


def test_summarize_rejects_wrong_schema(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("Frame,ID\n1,1\n")
    with pytest.raises(ValueError, match="必要な列"):
        summarize_csv(path)
