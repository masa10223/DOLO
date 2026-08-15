"""追跡結果をフレームに描画する。

2つの実装がある。

``draw_pose`` / ``draw_angle`` / ``CenterTrailRenderer``
    cv2 のみで描く。GUI の用途別動画を高速に生成する。
``draw_matplotlib``
    凡例付きで matplotlib が描く。リファクタ前の GIF/MOV と同じ見た目。
    **1フレームごとに Figure を生成・破棄するため極端に遅い。**

リファクタ前は速い実装（`annotate_frame_with_keypoints_and_angle`）も
定義されていたが呼ばれておらず、常に遅い方が使われていた。
"""

from __future__ import annotations

import numpy as np

from .results import FrameResult

__all__ = [
    "CenterTrailRenderer",
    "draw_angle",
    "draw_fast",
    "draw_matplotlib",
    "draw_pose",
    "color_for_id",
    "get_renderer",
    "DEFAULT_RENDERER",
]

DEFAULT_RENDERER = "matplotlib"
"""既定はリファクタ前と同じ見た目を保つため matplotlib。速度が要るなら "fast"。"""


# seaborn.color_palette("colorblind", 10) を固定値にしたもの。seaborn を
# 推論の必須依存にせず、旧スクリプトの見やすい配色を再現する。
_SEABORN_COLORBLIND_RGB = (
    (1, 115, 178),
    (222, 143, 5),
    (2, 158, 115),
    (213, 94, 0),
    (204, 120, 188),
    (202, 145, 97),
    (251, 175, 228),
    (148, 148, 148),
    (236, 225, 51),
    (86, 180, 233),
)


def color_for_id(track_id: int) -> tuple[float, float, float]:
    """ID から安定した色を決める（seaborn colorblind の巡回）。RGB 各 0..1。

    高速rendererでも matplotlib を読み込んでいた旧実装を定数化し、初回のfont cache
    作成と書き込み可能なhomeディレクトリへの依存をなくしている。
    """
    rgb = _SEABORN_COLORBLIND_RGB[(track_id - 1) % len(_SEABORN_COLORBLIND_RGB)]
    return tuple(channel / 255 for channel in rgb)


def _bgr_color(track_id: int, id_to_color: dict) -> tuple[int, int, int]:
    rgb = id_to_color.get(track_id, color_for_id(track_id))
    return tuple(int(c * 255) for c in reversed(rgb))


def _put_text_with_outline(image, text, origin, scale, color, thickness=1) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (15, 24, 24),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_pose_markers(image, track, color) -> None:
    """Head=丸、Middle=×、Tail=三角で姿勢を描く。"""
    import cv2

    head = np.asarray(track.head, dtype=int)
    middle = np.asarray(track.middle, dtype=int)
    tail = np.asarray(track.tail, dtype=int)

    cv2.line(image, tuple(tail), tuple(middle), color, 2, cv2.LINE_AA)
    cv2.line(image, tuple(middle), tuple(head), color, 2, cv2.LINE_AA)

    # Head: filled circle
    cv2.circle(image, tuple(head), 7, (248, 248, 248), 2, cv2.LINE_AA)
    cv2.circle(image, tuple(head), 5, color, -1, cv2.LINE_AA)

    # Middle: tilted cross (X)
    cv2.drawMarker(
        image,
        tuple(middle),
        (248, 248, 248),
        cv2.MARKER_TILTED_CROSS,
        17,
        5,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        image,
        tuple(middle),
        color,
        cv2.MARKER_TILTED_CROSS,
        17,
        2,
        cv2.LINE_AA,
    )

    # Tail: filled triangle
    x, y = int(tail[0]), int(tail[1])
    triangle = np.array(((x, y - 8), (x - 7, y + 6), (x + 7, y + 6)), dtype=np.int32)
    cv2.fillConvexPoly(image, triangle, color, cv2.LINE_AA)
    cv2.polylines(image, [triangle], True, (248, 248, 248), 2, cv2.LINE_AA)


def _draw_frame_label(image, frame_idx: int, note: str | None = None) -> None:
    import cv2

    label = f"Frame {frame_idx}"
    if note:
        label += f"  |  {note}"
    (width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(image, (12, 12), (width + 34, 48), (15, 24, 24), -1)
    cv2.putText(
        image,
        label,
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def _draw_pose_legend(image, frame_idx: int) -> None:
    """動画上部に ``Head ◯  Middle ×  Triangle △`` の凡例を描く。

    OpenCV の標準フォントは Unicode の記号を描けないため、文字は ASCII で
    描画し、丸・バツ・三角は図形として描く。これによりOSのフォント環境に
    依存せず、出力動画へ同じ凡例を焼き込める。
    """
    import cv2

    height, width = image.shape[:2]
    scale = min(0.62, max(0.34, width / 1050))
    thickness = 2 if scale >= 0.5 else 1
    text_color = (245, 245, 245)
    marker_color = (154, 218, 174)
    font = cv2.FONT_HERSHEY_SIMPLEX
    baseline_y = min(38, max(22, height - 8))
    x = 22

    def put(label: str) -> None:
        nonlocal x
        cv2.putText(
            image,
            label,
            (x, baseline_y),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        (label_width, _), _ = cv2.getTextSize(label, font, scale, thickness)
        x += label_width + max(7, int(10 * scale))

    legend_width = min(width - 12, max(1, int(535 * scale + 150)))
    cv2.rectangle(image, (12, 12), (legend_width, min(50, height - 1)), (15, 24, 24), -1)

    put(f"Frame {frame_idx}  |  Head")
    radius = max(4, int(7 * scale / 0.62))
    cv2.circle(image, (x + radius, baseline_y - radius), radius, marker_color, 2, cv2.LINE_AA)
    x += radius * 2 + max(14, int(20 * scale))

    put("Middle")
    marker_size = max(10, int(16 * scale / 0.62))
    cv2.drawMarker(
        image,
        (x + marker_size // 2, baseline_y - marker_size // 2),
        marker_color,
        cv2.MARKER_TILTED_CROSS,
        marker_size,
        2,
        cv2.LINE_AA,
    )
    x += marker_size + max(14, int(20 * scale))

    put("Triangle")
    half = max(5, int(7 * scale / 0.62))
    center_x = x + half
    center_y = baseline_y - half
    triangle = np.array(
        (
            (center_x, center_y - half),
            (center_x - half, center_y + half),
            (center_x + half, center_y + half),
        ),
        dtype=np.int32,
    )
    cv2.polylines(image, [triangle], True, marker_color, 2, cv2.LINE_AA)


def draw_pose(frame: np.ndarray, result: FrameResult, id_to_color: dict) -> np.ndarray:
    """部位別形状とIDだけを描く、通常の姿勢確認動画。入力・出力ともBGR。"""
    annotated = frame.copy()

    for track in result.tracks:
        middle = np.asarray(track.middle).astype(int)
        color = _bgr_color(track.track_id, id_to_color)
        _draw_pose_markers(annotated, track, color)
        _put_text_with_outline(
            annotated, f"ID {track.track_id}", tuple(middle + (-14, -14)), 0.52, color, 2
        )

    _draw_pose_legend(annotated, result.frame_idx)
    return annotated


def draw_angle(frame: np.ndarray, result: FrameResult, id_to_color: dict) -> np.ndarray:
    """姿勢に角度値を加えた、角度確認専用動画。"""
    annotated = draw_pose(frame, result, id_to_color)
    for track in result.tracks:
        middle = np.asarray(track.middle, dtype=int)
        color = _bgr_color(track.track_id, id_to_color)
        _put_text_with_outline(
            annotated,
            f"Angle {track.angle:+.1f} deg",
            tuple(middle + (12, 22)),
            0.48,
            color,
            2,
        )
    return annotated


# 既存API名。今後の通常注釈は角度なしの姿勢動画を指す。
draw_fast = draw_pose


class CenterTrailRenderer:
    """Middle点の移動履歴を、ID色の軌跡として描くstateful renderer。"""

    def __init__(self, trail_frames: int = 30) -> None:
        if trail_frames < 0:
            raise ValueError("trail_frames は0以上にしてください")
        self.trail_frames = int(trail_frames)
        self.history: dict[int, list[tuple[int, tuple[int, int]]]] = {}

    def __call__(self, frame: np.ndarray, result: FrameResult, id_to_color: dict) -> np.ndarray:
        import cv2

        annotated = frame.copy()
        visible_ids = set()
        for track in result.tracks:
            visible_ids.add(track.track_id)
            points = self.history.setdefault(track.track_id, [])
            point = tuple(np.asarray(track.middle, dtype=int))
            if track.time_since_update == 0 and (not points or points[-1][1] != point):
                points.append((result.frame_idx, point))

        if self.trail_frames:
            cutoff = result.frame_idx - self.trail_frames + 1
            for track_id in list(self.history):
                self.history[track_id] = [
                    item for item in self.history[track_id] if item[0] >= cutoff
                ]
                if not self.history[track_id]:
                    del self.history[track_id]

        for track_id, timed_points in self.history.items():
            points = [point for _, point in timed_points]
            base = _bgr_color(track_id, id_to_color)
            segment_count = max(1, len(points) - 1)
            for index, (start, end) in enumerate(zip(points, points[1:], strict=False)):
                brightness = 0.3 + 0.7 * ((index + 1) / segment_count)
                color = tuple(int(channel * brightness) for channel in base)
                cv2.line(annotated, start, end, color, 3, cv2.LINE_AA)
            if points and track_id in visible_ids:
                current = points[-1]
                cv2.circle(annotated, current, 8, (248, 248, 248), 2, cv2.LINE_AA)
                cv2.circle(annotated, current, 5, base, -1, cv2.LINE_AA)
                _put_text_with_outline(
                    annotated,
                    f"ID {track_id}",
                    (current[0] + 10, current[1] - 10),
                    0.52,
                    base,
                    2,
                )

        note = (
            "Center trail: all" if self.trail_frames == 0 else f"Center trail: {self.trail_frames}f"
        )
        _draw_frame_label(annotated, result.frame_idx, note)
        return annotated


def _canvas_to_array(fig) -> np.ndarray:
    """Figure を RGB 配列にする。

    ``tostring_rgb()`` は matplotlib 3.8 で非推奨、**3.10 で削除**された。
    新旧どちらでも動くようにここで吸収する。
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    if hasattr(fig.canvas, "tostring_rgb"):  # matplotlib < 3.10
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(height, width, 3)

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(height, width, 4)[..., :3]


def draw_matplotlib(frame: np.ndarray, result: FrameResult, id_to_color: dict) -> np.ndarray:
    """凡例付きで描く。リファクタ前と同じ見た目。入力・出力とも BGR。

    Warning
    -------
    1フレームごとに Figure を作るため非常に遅い。長い動画では
    :func:`draw_fast` を使うこと。
    """
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    height, width = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    try:
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for track in result.tracks:
            color = id_to_color.get(track.track_id, (1.0, 0.0, 0.0))
            ax.scatter(track.head[0], track.head[1], color=color, s=60, marker="o")
            ax.scatter(track.middle[0], track.middle[1], color=color, s=60, marker="x")
            ax.scatter(track.tail[0], track.tail[1], color=color, s=60, marker="^")

        ax.axis("off")

        id_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"ID {cid}",
                markerfacecolor=id_to_color[cid],
                markersize=10,
            )
            for cid in sorted(id_to_color)
        ]
        marker_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                label="Head ◯",
                markerfacecolor="gray",
                markersize=10,
            ),
            Line2D([0], [0], marker="x", color="gray", label="Middle ×", markersize=10),
            Line2D(
                [0],
                [0],
                marker="^",
                color="gray",
                label="Triangle △",
                markerfacecolor="gray",
                markersize=10,
            ),
        ]

        first = ax.legend(handles=id_legend, loc="upper right", title="IDs")
        ax.add_artist(first)
        ax.legend(handles=marker_legend, loc="upper left", title="Keypoints")

        ax.text(
            50,
            height - 50,
            f"Frame: {result.frame_idx}",
            color="red",
            fontsize=25,
            fontweight="bold",
        )

        plt.tight_layout(pad=0)
        plt.subplots_adjust(wspace=0, hspace=0)

        rgb = _canvas_to_array(fig)
    finally:
        plt.close(fig)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


_RENDERERS = {
    "fast": draw_pose,
    "pose": draw_pose,
    "angle": draw_angle,
    "matplotlib": draw_matplotlib,
}


def get_renderer(name: str):
    """名前から描画関数を引く。GUI の設定値をそのまま渡せる。"""
    try:
        return _RENDERERS[name]
    except KeyError:
        raise ValueError(f"未知の描画方式: {name!r}。使えるのは {sorted(_RENDERERS)}") from None
