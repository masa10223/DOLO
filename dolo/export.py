"""追跡結果の書き出し（Sink）。

**GUI の「出力選択ダイアログ」の受け皿がここ。** チェックされた項目に対応する
Sink を組み立てて :func:`dolo.tracking.track_video` に渡すだけでよい。

10万フレームの動画では描画済みフレームを全てメモリに保持できないため、
「結果をまとめて返してから書く」設計は採れない。Sink は1フレームずつ
受け取って逐次書き出す。

新しい出力形式（HDF5、SLEAP互換、DeepLabCut互換など）を足したいときは、
:class:`Sink` を実装したクラスを1つ追加するだけでよい。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .results import CSV_COLUMNS, FrameResult, VideoMeta

__all__ = [
    "Sink",
    "CsvSink",
    "GifSink",
    "Mp4Sink",
    "MovSink",
    "JsonSink",
    "MemorySink",
    "build_sinks",
    "AVAILABLE_FORMATS",
]


@runtime_checkable
class Sink(Protocol):
    """追跡結果の書き出し先。

    ``needs_image`` が False の Sink しか使われない場合、追跡ループは
    **描画処理を丸ごとスキップする**。CSV だけ欲しいときに大幅に速くなる。
    """

    needs_image: bool

    def open(self, meta: VideoMeta) -> None: ...

    def write(self, result: FrameResult, image: np.ndarray | None) -> None: ...

    def close(self) -> None: ...


class _BaseSink:
    """`with` でも使えるようにする共通部分。"""

    needs_image = False

    def open(self, meta: VideoMeta) -> None:  # pragma: no cover - 既定は何もしない
        pass

    def close(self) -> None:  # pragma: no cover
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class CsvSink(_BaseSink):
    """軌跡を CSV に書き出す。列は :data:`dolo.results.CSV_COLUMNS`。

    最も重要な出力。解析パイプラインはこれを入力にする。
    """

    needs_image = False

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = None
        self._writer = None
        self.rows_written = 0

    def open(self, meta: VideoMeta) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)

    def write(self, result: FrameResult, image: np.ndarray | None) -> None:
        for track in result.tracks:
            self._writer.writerow(track.as_csv_row(result.frame_idx))
            self.rows_written += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class JsonSink(_BaseSink):
    """軌跡を JSON Lines で書き出す。1行1フレーム。

    CSV と違い、あるフレームで検出ゼロだったことも記録される。
    手動修正UIや外部ツール連携で扱いやすい。
    """

    needs_image = False

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = None

    def open(self, meta: VideoMeta) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        header = {
            "type": "meta",
            "video": meta.path,
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "start_frame": meta.start_frame,
            "end_frame": meta.end_frame,
        }
        self._file.write(json.dumps(header, ensure_ascii=False) + "\n")

    def write(self, result: FrameResult, image: np.ndarray | None) -> None:
        payload = {
            "type": "frame",
            "frame": result.frame_idx,
            "tracks": [
                {
                    "id": t.track_id,
                    "head": list(t.head),
                    "middle": list(t.middle),
                    "tail": list(t.tail),
                    "angle": t.angle,
                    "dist_moved": t.dist_moved,
                    "confidence": t.confidence,
                    "time_since_update": t.time_since_update,
                }
                for t in result.tracks
            ],
        }
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class GifSink(_BaseSink):
    """注釈付きフレームを GIF にする。プレビュー用。

    GIF は非可逆かつ巨大になりやすいので、長い動画では :class:`MovSink` を推奨。
    """

    needs_image = True
    render_mode = "pose"

    def __init__(self, path: str | Path, fps: int = 10) -> None:
        self.path = Path(path)
        self.fps = fps
        self._writer = None

    def open(self, meta: VideoMeta) -> None:
        import imageio  # 遅延 import（CSV だけ欲しい人に imageio を要求しない）

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.path, mode="I", fps=self.fps)

    def write(self, result: FrameResult, image: np.ndarray | None) -> None:
        if image is None:
            return
        import cv2

        self._writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class MovSink(_BaseSink):
    """注釈付きフレームを動画ファイルにする。

    Note
    ----
    リファクタ前の実装は fps を 10 に決め打ちしていた。既定値を ``None`` に
    すると入力動画の fps を引き継ぐ。既存の出力を再現したい場合は
    ``fps=10`` を明示すること。
    """

    needs_image = True
    render_mode = None

    def __init__(self, path: str | Path, fps: float | None = None, fourcc: str = "mp4v") -> None:
        self.path = Path(path)
        self.fps = fps
        self.fourcc = fourcc
        self._writer = None

    def open(self, meta: VideoMeta) -> None:
        import cv2

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fps = self.fps if self.fps is not None else meta.fps
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*self.fourcc),
            fps,
            (meta.width, meta.height),
        )
        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise OSError(
                f"注釈動画のwriterを開けません: {self.path} "
                f"(codec={self.fourcc}, size={meta.width}x{meta.height}, fps={fps})"
            )

    def write(self, result: FrameResult, image: np.ndarray | None) -> None:
        if image is None:
            return
        self._writer.write(image)  # BGR のまま

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class Mp4Sink(_BaseSink):
    """ブラウザ再生向けのH.264 MP4を書き出す。

    ``imageio-ffmpeg`` が同梱するFFmpegを使うため、OS側のFFmpegやOpenCVの
    codec構成に依存しない。``yuv420p`` と ``faststart`` により主要ブラウザで
    そのまま再生できる。
    """

    needs_image = True

    def __init__(
        self,
        path: str | Path,
        *,
        render_mode: str = "pose",
        trail_frames: int = 30,
        fps: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.render_mode = render_mode
        self.trail_frames = int(trail_frames)
        self.fps = fps
        self._writer = None

    def open(self, meta: VideoMeta) -> None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:  # pragma: no cover - 環境診断で先に検出する
            raise ImportError(
                "MP4動画の生成には imageio-ffmpeg が必要です。"
                "`pip install imageio-ffmpeg` を実行してください。"
            ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fps = self.fps if self.fps is not None else meta.fps
        self._writer = imageio_ffmpeg.write_frames(
            str(self.path),
            (meta.width, meta.height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=fps,
            quality=7,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="warning",
            output_params=["-movflags", "+faststart"],
        )
        self._writer.send(None)

    def write(self, result: FrameResult, image: np.ndarray | None) -> None:
        if image is None:
            return
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._writer.send(rgb)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class MemorySink(_BaseSink):
    """結果をメモリに溜める。テストと、短い動画のプレビュー用。

    長い動画に使うとメモリを食い潰すので注意。
    """

    needs_image = False

    def __init__(self) -> None:
        self.frames: list[FrameResult] = []
        self.meta: VideoMeta | None = None

    def open(self, meta: VideoMeta) -> None:
        self.meta = meta
        self.frames = []

    def write(self, result: FrameResult, image: np.ndarray | None) -> None:
        self.frames.append(result)

    def to_dataframe(self):
        """pandas.DataFrame へ変換する（列は CSV と同じ）。"""
        import pandas as pd

        rows = [t.as_csv_row(f.frame_idx) for f in self.frames for t in f.tracks]
        return pd.DataFrame(rows, columns=CSV_COLUMNS)


# GUI の出力選択ダイアログが列挙する形式の一覧。
AVAILABLE_FORMATS = {
    "csv": ("軌跡 CSV", "解析パイプラインの入力になる主要な出力", ".csv"),
    "json": ("軌跡 JSON Lines", "検出ゼロのフレームも記録される", ".jsonl"),
    "mov": ("姿勢キーポイント動画", "Head ◯  Middle ×  Triangle △ と個体IDを表示", ".mp4"),
    "angle_mov": ("角度動画", "姿勢と曲がり角度を表示する別動画", ".mp4"),
    "center_mov": ("中心軌跡動画", "中部の移動軌跡。残像フレーム数を指定可能", ".mp4"),
    "gif": ("注釈付き GIF", "共有しやすいが大きくなりやすい", ".gif"),
}


def build_sinks(
    output_dir: str | Path,
    stem: str,
    formats,
    mov_fps: float | None = None,
    trail_frames: int = 30,
):
    """選択された形式名から Sink のリストを組み立てる。

    GUI のチェックボックスの状態をそのまま渡せる。

    Parameters
    ----------
    output_dir
        出力先ディレクトリ。
    stem
        拡張子を除いたファイル名（通常は入力動画名）。
    formats
        ``{"csv", "mov"}`` のような形式名の集合。
    mov_fps
        MOV の fps。None なら入力動画に合わせる。

    Returns
    -------
    list[Sink]

    Raises
    ------
    ValueError
        未知の形式名が含まれていた場合。
    """
    output_dir = Path(output_dir)
    unknown = set(formats) - set(AVAILABLE_FORMATS)
    if unknown:
        raise ValueError(
            f"未知の出力形式: {sorted(unknown)}。使えるのは {sorted(AVAILABLE_FORMATS)}"
        )

    sinks: list[Sink] = []
    if "csv" in formats:
        sinks.append(CsvSink(output_dir / f"{stem}.csv"))
    if "json" in formats:
        sinks.append(JsonSink(output_dir / f"{stem}.jsonl"))
    if "mov" in formats:
        sinks.append(Mp4Sink(output_dir / f"{stem}_pose.mp4", render_mode="pose", fps=mov_fps))
    if "angle_mov" in formats:
        sinks.append(Mp4Sink(output_dir / f"{stem}_angle.mp4", render_mode="angle", fps=mov_fps))
    if "center_mov" in formats:
        sinks.append(
            Mp4Sink(
                output_dir / f"{stem}_center_track.mp4",
                render_mode="center",
                trail_frames=trail_frames,
                fps=mov_fps,
            )
        )
    if "gif" in formats:
        sinks.append(GifSink(output_dir / f"{stem}.gif"))
    return sinks
