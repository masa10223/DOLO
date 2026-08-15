"""追跡ループ本体のテスト。

ultralytics をスタブし、その場で生成した小さな動画を流すことで、
GPU もモデルも無い環境でループの振る舞い（進捗・中断・描画スキップ・
Sink への配線）を検証する。
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dolo.export import CsvSink, MemorySink  # noqa: E402
from dolo.results import FrameResult, VideoMeta  # noqa: E402

WIDTH, HEIGHT, N_FRAMES = 160, 120, 12
N_FLIES = 3


# --------------------------------------------------------------------------
# ultralytics のスタブ
# --------------------------------------------------------------------------
class _Tensor:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Boxes:
    def __init__(self, confs):
        self.conf = _Tensor(confs)

    def __len__(self):
        return len(self.conf.numpy())


class _Keypoints:
    def __init__(self, arr):
        self.data = _Tensor(arr)


class _Result:
    def __init__(self, keypoints, confs):
        self.keypoints = _Keypoints(keypoints) if keypoints is not None else None
        self.boxes = _Boxes(confs) if confs is not None else None


class FakeYOLO:
    """フレームごとに少しずつ動く個体を返す、決定的なダミー検出器。"""

    detect = True  # False にすると検出ゼロを返す

    def __init__(self, path):
        self.path = path
        self.device = None
        self.calls = 0

    def to(self, device):
        self.device = device
        return self

    def __call__(self, frame, conf=None, iou=None, device=None):
        self.calls += 1
        if not FakeYOLO.detect:
            return [_Result(None, None)]

        step = self.calls
        keypoints, confs = [], []
        for i in range(N_FLIES):
            cx = 20 + i * 40 + step
            cy = 30 + i * 20
            # head, middle, tail の順
            keypoints.append([[cx + 8, cy, 1.0], [cx, cy, 1.0], [cx - 8, cy, 1.0]])
            confs.append(0.9 - i * 0.1)
        return [_Result(np.array(keypoints, dtype=np.float32), np.array(confs, dtype=np.float32))]


@pytest.fixture(autouse=True)
def stub_ultralytics(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    FakeYOLO.detect = True
    yield
    FakeYOLO.detect = True


@pytest.fixture
def video(tmp_path):
    """短いダミー動画。画の内容は使われない（検出はスタブが返す）。"""
    path = tmp_path / "clip.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (WIDTH, HEIGHT))
    rng = np.random.default_rng(0)
    for _ in range(N_FRAMES):
        writer.write(rng.integers(0, 60, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()
    assert path.exists()
    return path


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"not-a-real-model")
    return path


# --------------------------------------------------------------------------
# 基本動作
# --------------------------------------------------------------------------
def test_tracks_all_frames_and_reports_summary(video, model):
    from dolo.tracking import TrackParams, track_video

    sink = MemorySink()
    summary = track_video(video, model, [sink], params=TrackParams(max_ids=N_FLIES), device="cpu")

    assert summary.frames_processed == N_FRAMES
    assert not summary.cancelled
    assert summary.ids_seen == {1, 2, 3}
    assert summary.elapsed_sec >= 0


def test_sink_receives_one_result_per_frame(video, model):
    from dolo.tracking import TrackParams, track_video

    sink = MemorySink()
    track_video(video, model, [sink], params=TrackParams(max_ids=N_FLIES), device="cpu")

    assert len(sink.frames) == N_FRAMES
    assert [f.frame_idx for f in sink.frames] == list(range(N_FRAMES))
    assert all(isinstance(f, FrameResult) for f in sink.frames)


def test_sink_receives_video_metadata(video, model):
    from dolo.tracking import track_video

    sink = MemorySink()
    track_video(video, model, [sink], device="cpu")

    assert isinstance(sink.meta, VideoMeta)
    assert (sink.meta.width, sink.meta.height) == (WIDTH, HEIGHT)
    assert sink.meta.total_frames == N_FRAMES


def test_first_frame_has_no_output_because_tracks_are_tentative(video, model):
    """n_init=2 なので最初のフレームには確定トラックが無い。

    実データのゴールデンCSVが Frame=1 から始まるのはこれが理由。
    """
    from dolo.tracking import TrackParams, track_video

    sink = MemorySink()
    track_video(video, model, [sink], params=TrackParams(max_ids=N_FLIES, n_init=2), device="cpu")

    assert len(sink.frames[0]) == 0
    assert len(sink.frames[1]) == N_FLIES


def test_csv_written_to_disk(video, model, tmp_path):
    from dolo.tracking import track_to_csv

    out = tmp_path / "traj.csv"
    summary = track_to_csv(video, model, out, device="cpu", max_ids=N_FLIES)

    assert out.exists()
    assert summary.rows_written > 0
    assert out.read_text().splitlines()[0].startswith("Frame,ID,Head_X")


def test_max_missing_frames_is_accepted_as_alias(video, model, tmp_path):
    """既存 CLI との後方互換。"""
    from dolo.tracking import track_to_csv

    track_to_csv(video, model, tmp_path / "a.csv", device="cpu", max_missing_frames=5)


def test_unknown_parameter_is_rejected_clearly(video, model, tmp_path):
    from dolo.tracking import track_to_csv

    with pytest.raises(TypeError, match="未知のパラメータ"):
        track_to_csv(video, model, tmp_path / "a.csv", device="cpu", nonsense=1)


# --------------------------------------------------------------------------
# フレーム範囲
# --------------------------------------------------------------------------
def test_start_and_end_frame_limit_the_range(video, model):
    from dolo.tracking import track_video

    sink = MemorySink()
    track_video(video, model, [sink], device="cpu", start_frame=3, end_frame=8)

    assert [f.frame_idx for f in sink.frames] == [3, 4, 5, 6, 7]


def test_end_frame_beyond_video_is_clamped(video, model):
    from dolo.tracking import track_video

    sink = MemorySink()
    track_video(video, model, [sink], device="cpu", end_frame=9999)
    assert len(sink.frames) == N_FRAMES


# --------------------------------------------------------------------------
# 進捗と中断（GUI に直結する部分）
# --------------------------------------------------------------------------
def test_progress_callback_is_called_for_every_frame(video, model):
    from dolo.tracking import track_video

    seen = []
    track_video(video, model, [], device="cpu", progress=lambda d, t: seen.append((d, t)))

    assert len(seen) == N_FRAMES
    assert seen[0] == (1, N_FRAMES)
    assert seen[-1] == (N_FRAMES, N_FRAMES)


def test_cancel_stops_the_loop_early(video, model):
    from dolo.tracking import track_video

    class CancelAfter:
        def __init__(self, n):
            self.n = n
            self.count = 0

        def is_set(self):
            self.count += 1
            return self.count > self.n

    sink = MemorySink()
    summary = track_video(video, model, [sink], device="cpu", cancel=CancelAfter(4))

    assert summary.cancelled
    assert summary.frames_processed == 4
    assert len(sink.frames) == 4


def test_sinks_are_closed_even_when_cancelled(video, model, tmp_path):
    """中断しても CSV は壊れず、読める状態で閉じられる。"""
    from dolo.tracking import track_video

    class AlwaysCancel:
        def is_set(self):
            return True

    out = tmp_path / "cancelled.csv"
    sink = CsvSink(out)
    summary = track_video(video, model, [sink], device="cpu", cancel=AlwaysCancel())

    assert summary.cancelled
    assert out.exists()
    assert out.read_text().splitlines()[0].startswith("Frame,")


def test_sinks_are_closed_when_an_error_occurs(video, model, tmp_path):
    from dolo.tracking import track_video

    class Exploding(CsvSink):
        def write(self, result, image):
            raise RuntimeError("boom")

    sink = Exploding(tmp_path / "broken.csv")
    with pytest.raises(RuntimeError, match="boom"):
        track_video(video, model, [sink], device="cpu")

    assert sink._file is None, "例外時もファイルは閉じられるべき"


def test_log_callback_receives_messages(video, model):
    from dolo.tracking import track_video

    lines = []
    track_video(video, model, [], device="cpu", log=lines.append)
    assert any("デバイス" in x for x in lines)


# --------------------------------------------------------------------------
# 描画のスキップ（CSV のみ選択時の高速化）
# --------------------------------------------------------------------------
def test_rendering_is_skipped_when_no_visual_sink(video, model, monkeypatch):
    from dolo import tracking

    called = []
    monkeypatch.setattr(tracking, "get_renderer", lambda name: lambda *a: called.append(1))

    tracking.track_video(video, model, [MemorySink()], device="cpu")
    assert called == [], "可視出力が無いのに描画が走っている"


def test_rendering_runs_when_a_visual_sink_is_present(video, model, tmp_path, monkeypatch):
    from dolo import tracking
    from dolo.export import MovSink

    calls = []

    def fake_renderer(frame, result, id_to_color):
        calls.append(result.frame_idx)
        return frame

    monkeypatch.setattr(tracking, "get_renderer", lambda name: fake_renderer)
    tracking.track_video(video, model, [MovSink(tmp_path / "out.mov", fps=10)], device="cpu")

    assert len(calls) > 0


# --------------------------------------------------------------------------
# 異常系
# --------------------------------------------------------------------------
def test_missing_video_raises_clear_error(model, tmp_path):
    from dolo.tracking import track_video

    with pytest.raises(FileNotFoundError, match="動画が見つかりません"):
        track_video(tmp_path / "nope.mov", model, [], device="cpu")


def test_no_detections_produces_empty_frames(video, model):
    from dolo.tracking import track_video

    FakeYOLO.detect = False
    sink = MemorySink()
    summary = track_video(video, model, [sink], device="cpu")

    assert summary.frames_processed == N_FRAMES
    assert summary.rows_written == 0
    assert all(len(f) == 0 for f in sink.frames)


def test_running_with_no_sinks_is_allowed(video, model):
    """統計だけ欲しい場合（GUI のドライラン等）。"""
    from dolo.tracking import track_video

    summary = track_video(video, model, [], device="cpu")
    assert summary.frames_processed == N_FRAMES
