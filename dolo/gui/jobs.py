"""推論をGUIスレッドから分離するジョブ実行層。

NiceGUI のイベントループを塞がず、CUDA を同時に複数ジョブから奪い合わないよう
既定では1ワーカーで順番に実行する。UI は :meth:`snapshot` をポーリングするだけで、
ワーカースレッドからUI要素へ触れない。
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
import warnings
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from dolo import __version__
from dolo.export import AVAILABLE_FORMATS, build_sinks
from dolo.results import TrackingSummary
from dolo.tracking import TrackParams

from .analysis import RunMetrics, summarize_csv
from .config import safe_filename, validate_model_path, validate_video_path


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETE, self.FAILED, self.CANCELLED}


@dataclass(frozen=True)
class InferenceConfig:
    video_path: Path
    model_path: Path
    output_root: Path
    formats: frozenset[str] = frozenset({"csv", "mov"})
    params: TrackParams = field(default_factory=TrackParams)
    device: str = "auto"
    start_frame: int = 0
    end_frame: int | None = None
    renderer: str = "fast"
    trail_frames: int = 30

    def validated(self) -> InferenceConfig:
        video = validate_video_path(self.video_path)
        model = validate_model_path(self.model_path)
        unknown = set(self.formats) - set(AVAILABLE_FORMATS)
        if unknown:
            raise ValueError(f"未知の出力形式: {sorted(unknown)}")
        if not self.formats:
            raise ValueError("出力形式を1つ以上選んでください")
        if self.start_frame < 0:
            raise ValueError("開始フレームは0以上にしてください")
        if self.end_frame is not None and self.end_frame <= self.start_frame:
            raise ValueError("終了フレームは開始フレームより後にしてください")
        if self.params.max_ids < 1:
            raise ValueError("個体数は1以上にしてください")
        if self.params.frame_skip < 1:
            raise ValueError("frame skip は1以上にしてください")
        if self.trail_frames < 0:
            raise ValueError("中心軌跡の残像 frame は0以上にしてください")
        if self.renderer not in {"fast", "matplotlib"}:
            raise ValueError("描画方式は fast または matplotlib を指定してください")
        output = Path(self.output_root).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        return InferenceConfig(
            video_path=video,
            model_path=model,
            output_root=output,
            formats=frozenset(self.formats),
            params=self.params,
            device=self.device,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            renderer=self.renderer,
            trail_frames=int(self.trail_frames),
        )


@dataclass(frozen=True)
class JobSnapshot:
    id: str
    state: JobState
    created_at: str
    started_at: str | None
    finished_at: str | None
    progress: float
    done: int
    total: int
    logs: tuple[str, ...]
    run_dir: Path
    outputs: tuple[Path, ...]
    summary: TrackingSummary | None
    metrics: RunMetrics | None
    error: str | None


@dataclass
class _Job:
    id: str
    config: InferenceConfig
    run_dir: Path
    state: JobState = JobState.QUEUED
    created_at: str = field(default_factory=lambda: _now())
    started_at: str | None = None
    finished_at: str | None = None
    done: int = 0
    total: int = 0
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=400))
    outputs: list[Path] = field(default_factory=list)
    summary: TrackingSummary | None = None
    metrics: RunMetrics | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    future: Future | None = None

    def log(self, message: str) -> None:
        line = str(message).strip()
        if not line:
            return
        with self.lock:
            self.logs.append(line)

    def set_progress(self, done: int, total: int) -> None:
        with self.lock:
            self.done = max(0, int(done))
            self.total = max(0, int(total))

    def snapshot(self) -> JobSnapshot:
        with self.lock:
            progress = min(1.0, self.done / self.total) if self.total else 0.0
            return JobSnapshot(
                id=self.id,
                state=self.state,
                created_at=self.created_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
                progress=progress,
                done=self.done,
                total=self.total,
                logs=tuple(self.logs),
                run_dir=self.run_dir,
                outputs=tuple(self.outputs),
                summary=self.summary,
                metrics=self.metrics,
                error=self.error,
            )


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


Tracker = Callable[..., TrackingSummary]


class JobManager:
    """推論ジョブの投入・進捗取得・中断を管理する。"""

    def __init__(self, max_workers: int = 1, tracker: Tracker | None = None) -> None:
        if max_workers < 1:
            raise ValueError("max_workers は1以上にしてください")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="dolo-infer"
        )
        self._tracker = tracker
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()

    def submit(self, config: InferenceConfig) -> JobSnapshot:
        config = config.validated()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = safe_filename(config.video_path.stem, "video")
        job_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        run_dir = config.output_root / f"{timestamp}-{stem}-{job_id[-8:]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        job = _Job(id=job_id, config=config, run_dir=run_dir)
        job.log("ジョブを受け付けました")
        with self._lock:
            self._jobs[job_id] = job
        job.future = self._executor.submit(self._execute, job)
        return job.snapshot()

    def snapshot(self, job_id: str) -> JobSnapshot:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError:
                raise KeyError(f"ジョブが見つかりません: {job_id}") from None
        return job.snapshot()

    def latest(self) -> JobSnapshot | None:
        with self._lock:
            if not self._jobs:
                return None
            job = next(reversed(self._jobs.values()))
        return job.snapshot()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError:
                return False
        with job.lock:
            if job.state.terminal:
                return False
            job.cancel_event.set()
            if job.future is not None and job.future.cancel():
                job.state = JobState.CANCELLED
                job.finished_at = _now()
                job.log("待機中のジョブを中断しました")
                self._write_manifest(job)
            else:
                job.log("中断を要求しました。現在のフレーム処理後に停止します")
            return True

    def wait(self, job_id: str, timeout: float | None = None) -> JobSnapshot:
        with self._lock:
            job = self._jobs[job_id]
        if job.future is not None:
            job.future.result(timeout=timeout)
        return job.snapshot()

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(self, job: _Job) -> None:
        with job.lock:
            if job.cancel_event.is_set():
                job.state = JobState.CANCELLED
                job.finished_at = _now()
                self._write_manifest(job)
                return
            job.state = JobState.RUNNING
            job.started_at = _now()
        job.log("default 重みを読み込んで推論を開始します")

        stem = safe_filename(job.config.video_path.stem, "video")
        sinks = build_sinks(
            job.run_dir,
            stem,
            job.config.formats,
            trail_frames=job.config.trail_frames,
        )
        job.outputs = [sink.path for sink in sinks if hasattr(sink, "path")]

        try:
            tracker = self._tracker
            if tracker is None:
                from dolo.tracking import track_video

                tracker = track_video

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                summary = tracker(
                    video_path=job.config.video_path,
                    model_path=job.config.model_path,
                    sinks=sinks,
                    params=job.config.params,
                    device=job.config.device,
                    start_frame=job.config.start_frame,
                    end_frame=job.config.end_frame,
                    renderer=job.config.renderer,
                    progress=job.set_progress,
                    cancel=job.cancel_event,
                    log=job.log,
                )
                for warning in caught:
                    job.log(f"警告: {warning.message}")

            expected_frames = job.total or summary.frames_processed
            csv_path = next((p for p in job.outputs if p.suffix.lower() == ".csv"), None)
            metrics = (
                summarize_csv(csv_path, expected_frames=expected_frames)
                if csv_path is not None and csv_path.exists()
                else None
            )

            with job.lock:
                job.summary = summary
                job.metrics = metrics
                job.state = JobState.CANCELLED if summary.cancelled else JobState.COMPLETE
                job.finished_at = _now()
            if job.state == JobState.COMPLETE:
                job.log("すべての出力が完成しました")
            else:
                job.log("中断時点までの出力を保存しました")
        except Exception as exc:  # noqa: BLE001 - ジョブ境界で全例外を状態へ変換する
            with job.lock:
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = JobState.FAILED
                job.finished_at = _now()
            job.log(job.error)
            job.log(traceback.format_exc(limit=8))
        finally:
            manifest = job.run_dir / "run.json"
            if manifest not in job.outputs:
                job.outputs.append(manifest)
            self._write_manifest(job)

    def _write_manifest(self, job: _Job) -> None:
        with job.lock:
            summary = asdict(job.summary) if job.summary is not None else None
            if summary is not None:
                summary["ids_seen"] = sorted(summary["ids_seen"])
            payload = {
                "schema_version": 1,
                "dolo_version": __version__,
                "job_id": job.id,
                "state": job.state.value,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "input": {
                    "video": str(job.config.video_path),
                    "model": str(job.config.model_path),
                    "model_size_bytes": job.config.model_path.stat().st_size,
                },
                "settings": {
                    "device": job.config.device,
                    "start_frame": job.config.start_frame,
                    "end_frame": job.config.end_frame,
                    "renderer": job.config.renderer,
                    "trail_frames": job.config.trail_frames,
                    "formats": sorted(job.config.formats),
                    "tracking": asdict(job.config.params),
                },
                "progress": {"done": job.done, "total": job.total},
                "summary": summary,
                "metrics": job.metrics.to_dict() if job.metrics else None,
                "outputs": [p.name for p in job.outputs if p.name != "run.json" and p.exists()],
                "error": job.error,
            }
        target = job.run_dir / "run.json"
        temporary = job.run_dir / ".run.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
