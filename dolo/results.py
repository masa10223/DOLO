"""追跡結果を表すデータ構造。

追跡ループと出力（CSV/GIF/MOV/…）を分離するための境界。numpy にしか依存
しないので、torch も opencv も無い環境で import・テストできる。

リファクタ前は追跡と書き出しが1つの関数に融合しており、出力の取捨選択が
できなかった。この層を挟むことで「どの形式で出すか」を呼び出し側が決められる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TrackRecord", "FrameResult", "VideoMeta", "TrackingSummary", "CSV_COLUMNS"]

# 追跡が出力する CSV の列。リファクタ前から変更していない（既存の解析コードが依存）。
CSV_COLUMNS = [
    "Frame",
    "ID",
    "Head_X",
    "Head_Y",
    "Middle_X",
    "Middle_Y",
    "Tail_X",
    "Tail_Y",
    "Angle",
    "DistMoved",
    "Confidence",
    "TimeSinceUpdate",
]


@dataclass(frozen=True)
class TrackRecord:
    """1フレーム・1個体分の追跡結果。"""

    track_id: int
    head: tuple[float, float]
    middle: tuple[float, float]
    tail: tuple[float, float]
    angle: float
    """tail→middle→head の角度。符号あり [-180, 180] 度。"""
    dist_moved: float
    """直前の出力フレームからの移動距離（ピクセル）。"""
    confidence: float
    """トラックの信頼度。検出の信頼度ではないことに注意。"""
    time_since_update: int
    """最後に観測で更新されてからのフレーム数。0 なら今フレームで実測。"""

    def as_csv_row(self, frame_idx: int) -> list:
        """CSV の1行へ変換する。列順は :data:`CSV_COLUMNS` に一致する。"""
        return [
            frame_idx,
            self.track_id,
            self.head[0],
            self.head[1],
            self.middle[0],
            self.middle[1],
            self.tail[0],
            self.tail[1],
            self.angle,
            self.dist_moved,
            self.confidence,
            self.time_since_update,
        ]


@dataclass
class FrameResult:
    """1フレーム分の追跡結果。"""

    frame_idx: int
    tracks: list[TrackRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracks)

    @property
    def track_ids(self) -> list[int]:
        return [t.track_id for t in self.tracks]


@dataclass(frozen=True)
class VideoMeta:
    """入力動画のメタ情報。Sink が出力を準備するのに使う。"""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    start_frame: int
    end_frame: int

    @property
    def n_frames_to_process(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class TrackingSummary:
    """追跡が終わったときの要約。GUI の完了通知に使う。"""

    frames_processed: int = 0
    rows_written: int = 0
    ids_seen: set[int] = field(default_factory=set)
    cancelled: bool = False
    elapsed_sec: float = 0.0

    def __str__(self) -> str:
        state = "中断" if self.cancelled else "完了"
        ids = ", ".join(str(i) for i in sorted(self.ids_seen)) or "なし"
        return (
            f"{state}: {self.frames_processed} フレーム処理, "
            f"{self.rows_written} 行出力, ID=[{ids}], {self.elapsed_sec:.1f} 秒"
        )
