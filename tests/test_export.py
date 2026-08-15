"""Sink（出力層）のテスト。torch も動画も不要。

GUI の「出力選択ダイアログ」がこの層に依存するので、選択の組み合わせが
正しく Sink に落ちることを固定する。
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from dolo.export import (
    AVAILABLE_FORMATS,
    CsvSink,
    GifSink,
    JsonSink,
    MemorySink,
    MovSink,
    Mp4Sink,
    build_sinks,
)
from dolo.results import CSV_COLUMNS, FrameResult, TrackRecord, VideoMeta


@pytest.fixture
def meta():
    return VideoMeta(
        path="sample.mov",
        width=1280,
        height=960,
        fps=10.0,
        total_frames=194,
        start_frame=0,
        end_frame=150,
    )


def rec(track_id=1, angle=-12.5):
    return TrackRecord(
        track_id=track_id,
        head=(100.0, 200.0),
        middle=(110.0, 210.0),
        tail=(120.0, 220.0),
        angle=angle,
        dist_moved=1.5,
        confidence=1.0,
        time_since_update=0,
    )


# --------------------------------------------------------------------------
# データ構造
# --------------------------------------------------------------------------
def test_csv_row_matches_column_order():
    row = rec(track_id=3).as_csv_row(frame_idx=42)
    assert len(row) == len(CSV_COLUMNS)
    assert row[CSV_COLUMNS.index("Frame")] == 42
    assert row[CSV_COLUMNS.index("ID")] == 3
    assert row[CSV_COLUMNS.index("Head_X")] == 100.0
    assert row[CSV_COLUMNS.index("Angle")] == -12.5


def test_frame_result_reports_ids_and_length():
    result = FrameResult(frame_idx=1, tracks=[rec(1), rec(2)])
    assert len(result) == 2
    assert result.track_ids == [1, 2]


def test_video_meta_counts_frames_to_process(meta):
    assert meta.n_frames_to_process == 150


# --------------------------------------------------------------------------
# CsvSink
# --------------------------------------------------------------------------
def test_csv_sink_writes_header_and_rows(tmp_path, meta):
    path = tmp_path / "out.csv"
    sink = CsvSink(path)
    sink.open(meta)
    sink.write(FrameResult(1, [rec(1), rec(2)]), None)
    sink.write(FrameResult(2, [rec(1)]), None)
    sink.close()

    rows = list(csv.reader(path.open()))
    assert rows[0] == CSV_COLUMNS
    assert len(rows) == 4  # ヘッダ + 3行
    assert sink.rows_written == 3


def test_csv_sink_creates_missing_directories(tmp_path, meta):
    path = tmp_path / "deep" / "nested" / "out.csv"
    with CsvSink(path) as sink:
        sink.open(meta)
        sink.write(FrameResult(1, [rec()]), None)
    assert path.exists()


def test_csv_sink_writes_nothing_for_empty_frames(tmp_path, meta):
    """検出ゼロのフレームは CSV に行を作らない（リファクタ前と同じ）。"""
    path = tmp_path / "out.csv"
    sink = CsvSink(path)
    sink.open(meta)
    sink.write(FrameResult(1, []), None)
    sink.close()
    assert len(list(csv.reader(path.open()))) == 1  # ヘッダのみ


def test_csv_sink_does_not_need_images():
    assert CsvSink("x.csv").needs_image is False


# --------------------------------------------------------------------------
# JsonSink
# --------------------------------------------------------------------------
def test_json_sink_records_meta_then_frames(tmp_path, meta):
    path = tmp_path / "out.jsonl"
    sink = JsonSink(path)
    sink.open(meta)
    sink.write(FrameResult(1, [rec(1)]), None)
    sink.write(FrameResult(2, []), None)
    sink.close()

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["type"] == "meta"
    assert lines[0]["width"] == 1280
    assert lines[1]["type"] == "frame"
    assert lines[1]["tracks"][0]["id"] == 1


def test_json_sink_keeps_empty_frames(tmp_path, meta):
    """CSV と違い、検出ゼロのフレームも記録される。手動修正UIで効いてくる差。"""
    path = tmp_path / "out.jsonl"
    with JsonSink(path) as sink:
        sink.open(meta)
        sink.write(FrameResult(7, []), None)

    frames = [json.loads(x) for x in path.read_text().splitlines() if '"frame"' in x]
    assert frames[0]["frame"] == 7
    assert frames[0]["tracks"] == []


# --------------------------------------------------------------------------
# MemorySink
# --------------------------------------------------------------------------
def test_memory_sink_collects_frames(meta):
    sink = MemorySink()
    sink.open(meta)
    sink.write(FrameResult(1, [rec(1), rec(2)]), None)
    sink.write(FrameResult(2, [rec(1)]), None)
    assert len(sink.frames) == 2


def test_memory_sink_to_dataframe_matches_csv_schema(meta):
    pytest.importorskip("pandas")
    sink = MemorySink()
    sink.open(meta)
    sink.write(FrameResult(1, [rec(1), rec(2)]), None)
    df = sink.to_dataframe()
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == 2


def test_memory_sink_resets_on_reopen(meta):
    sink = MemorySink()
    sink.open(meta)
    sink.write(FrameResult(1, [rec()]), None)
    sink.open(meta)
    assert sink.frames == []


# --------------------------------------------------------------------------
# 可視 Sink は画像を要求する
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls,arg", [(GifSink, "x.gif"), (MovSink, "x.mov"), (Mp4Sink, "x.mp4")])
def test_visual_sinks_declare_needs_image(cls, arg):
    assert cls(arg).needs_image is True


def test_mov_sink_defaults_to_source_fps():
    """リファクタ前は fps=10 決め打ちだった。既定では入力に追従する。"""
    assert MovSink("x.mov").fps is None
    assert MovSink("x.mov", fps=10).fps == 10


def test_mp4_sink_writes_browser_video(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "preview.mp4"
    small_meta = VideoMeta(
        path="sample.mov",
        width=64,
        height=48,
        fps=10.0,
        total_frames=2,
        start_frame=0,
        end_frame=2,
    )
    sink = Mp4Sink(path)
    sink.open(small_meta)
    sink.write(FrameResult(0, []), np.zeros((48, 64, 3), dtype=np.uint8))
    sink.write(FrameResult(1, []), np.full((48, 64, 3), 100, dtype=np.uint8))
    sink.close()

    assert path.read_bytes()[4:8] == b"ftyp"
    assert path.stat().st_size > 500


# --------------------------------------------------------------------------
# build_sinks — GUI の出力選択ダイアログが使う入口
# --------------------------------------------------------------------------
def test_build_sinks_csv_only(tmp_path):
    sinks = build_sinks(tmp_path, "video", {"csv"})
    assert len(sinks) == 1
    assert isinstance(sinks[0], CsvSink)
    assert sinks[0].path.name == "video.csv"


def test_build_sinks_all_formats(tmp_path):
    sinks = build_sinks(tmp_path, "video", set(AVAILABLE_FORMATS))
    assert len(sinks) == len(AVAILABLE_FORMATS)
    assert {s.path.name for s in sinks} == {
        "video.csv",
        "video.jsonl",
        "video_pose.mp4",
        "video_angle.mp4",
        "video_center_track.mp4",
        "video.gif",
    }


def test_center_video_receives_trail_length(tmp_path):
    sinks = build_sinks(tmp_path, "video", {"center_mov"}, trail_frames=75)
    assert len(sinks) == 1
    assert isinstance(sinks[0], Mp4Sink)
    assert sinks[0].render_mode == "center"
    assert sinks[0].trail_frames == 75


def test_build_sinks_with_nothing_selected(tmp_path):
    assert build_sinks(tmp_path, "video", set()) == []


def test_build_sinks_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="未知の出力形式"):
        build_sinks(tmp_path, "video", {"csv", "hdf5"})


def test_build_sinks_error_lists_valid_formats(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        build_sinks(tmp_path, "video", {"xlsx"})
    assert "csv" in str(excinfo.value)


def test_selecting_only_csv_means_no_rendering_needed(tmp_path):
    """描画スキップの判定ロジックが成立していること。"""
    sinks = build_sinks(tmp_path, "video", {"csv", "json"})
    assert not any(s.needs_image for s in sinks)

    sinks = build_sinks(tmp_path, "video", {"csv", "mov"})
    assert any(s.needs_image for s in sinks)


def test_available_formats_have_labels_and_extensions():
    for name, (label, description, ext) in AVAILABLE_FORMATS.items():
        assert label and description
        assert ext.startswith("."), name
