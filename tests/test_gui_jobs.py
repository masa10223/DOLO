from __future__ import annotations

import json
import threading

import pytest

from dolo.gui.jobs import InferenceConfig, JobManager, JobState
from dolo.results import FrameResult, TrackingSummary, TrackRecord, VideoMeta
from dolo.tracking import TrackParams


def _record(track_id=1):
    return TrackRecord(
        track_id=track_id,
        head=(11.0, 10.0),
        middle=(10.0, 10.0),
        tail=(9.0, 10.0),
        angle=0.0,
        dist_moved=2.0,
        confidence=0.9,
        time_since_update=0,
    )


def _fake_tracker(**kwargs):
    meta = VideoMeta(
        path=str(kwargs["video_path"]),
        width=100,
        height=80,
        fps=10.0,
        total_frames=3,
        start_frame=0,
        end_frame=3,
    )
    sinks = kwargs["sinks"]
    for sink in sinks:
        sink.open(meta)
    try:
        for frame in range(3):
            result = FrameResult(frame, [_record(1)])
            for sink in sinks:
                sink.write(result, None)
            kwargs["progress"](frame + 1, 3)
    finally:
        for sink in sinks:
            sink.close()
    kwargs["log"]("fake inference complete")
    return TrackingSummary(
        frames_processed=3,
        rows_written=3,
        ids_seen={1},
        elapsed_sec=0.01,
    )


def _config(tmp_path):
    video = tmp_path / "input.mov"
    model = tmp_path / "best.pt"
    video.write_bytes(b"video")
    model.write_bytes(b"weights")
    return InferenceConfig(
        video_path=video,
        model_path=model,
        output_root=tmp_path / "runs",
        formats=frozenset({"csv", "json"}),
        params=TrackParams(max_ids=2),
        device="cpu",
    )


def test_job_runs_to_completion_and_writes_manifest(tmp_path):
    manager = JobManager(tracker=_fake_tracker)
    try:
        submitted = manager.submit(_config(tmp_path))
        done = manager.wait(submitted.id, timeout=5)
    finally:
        manager.shutdown()

    assert done.state == JobState.COMPLETE
    assert done.progress == 1.0
    assert done.metrics is not None
    assert done.metrics.ids[0].total_distance_px == 6.0
    manifest = done.run_dir / "run.json"
    data = json.loads(manifest.read_text())
    assert data["state"] == "complete"
    assert data["input"]["model"].endswith("best.pt")
    assert set(data["outputs"]) == {"input.csv", "input.jsonl"}


def test_failed_job_is_reported_instead_of_crashing_manager(tmp_path):
    def explode(**_kwargs):
        raise RuntimeError("GPU unavailable")

    manager = JobManager(tracker=explode)
    try:
        submitted = manager.submit(_config(tmp_path))
        done = manager.wait(submitted.id, timeout=5)
    finally:
        manager.shutdown()

    assert done.state == JobState.FAILED
    assert "GPU unavailable" in (done.error or "")
    assert json.loads((done.run_dir / "run.json").read_text())["state"] == "failed"


def test_negative_center_trail_is_rejected(tmp_path):
    config = _config(tmp_path)
    invalid = InferenceConfig(
        video_path=config.video_path,
        model_path=config.model_path,
        output_root=config.output_root,
        formats=config.formats,
        trail_frames=-1,
    )
    with pytest.raises(ValueError, match="残像 frame"):
        invalid.validated()


def test_running_job_can_be_cancelled(tmp_path):
    started = threading.Event()

    def cancellable(**kwargs):
        started.set()
        kwargs["cancel"].wait(timeout=2)
        return TrackingSummary(cancelled=True)

    manager = JobManager(tracker=cancellable)
    try:
        submitted = manager.submit(_config(tmp_path))
        assert started.wait(timeout=2)
        assert manager.cancel(submitted.id)
        done = manager.wait(submitted.id, timeout=5)
    finally:
        manager.shutdown()

    assert done.state == JobState.CANCELLED
