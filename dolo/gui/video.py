"""動画の軽量な事前検査とサムネイル生成。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import validate_video_path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frames: int
    duration_sec: float
    size_bytes: int
    codec: str

    @property
    def resolution(self) -> str:
        return f"{self.width} × {self.height}"

    @property
    def duration_label(self) -> str:
        minutes, seconds = divmod(max(0, round(self.duration_sec)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def to_manifest(self) -> dict:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def probe_video(path: str | Path) -> VideoInfo:
    """OpenCV で動画を開き、推論前に壊れた入力を検出する。"""
    import cv2

    video = validate_video_path(path)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise OSError(f"動画を開けません。コーデックまたはファイルを確認してください: {video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
        if width <= 0 or height <= 0:
            raise OSError(f"動画サイズを取得できません: {video}")
        if fps <= 0 or fps != fps:
            fps = 10.0
        duration = frames / fps if frames > 0 else 0.0
        return VideoInfo(
            path=video,
            width=width,
            height=height,
            fps=fps,
            frames=frames,
            duration_sec=duration,
            size_bytes=video.stat().st_size,
            codec=codec or "unknown",
        )
    finally:
        cap.release()


def create_thumbnail(video: VideoInfo, output_path: str | Path) -> Path:
    """動画の中央付近から JPEG サムネイルを1枚作る。"""
    import cv2

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video.path))
    try:
        target = max(0, min(video.frames - 1, video.frames // 2)) if video.frames else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise OSError(f"サムネイル用フレームを読み込めません: {video.path}")
        max_width = 1280
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(output), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
            raise OSError(f"サムネイルを書き出せません: {output}")
        return output
    finally:
        cap.release()
