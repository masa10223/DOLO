"""動画の推論＋追跡のメインループ。

リファクタ前の ``process_video_to_gif_with_angles()`` は、推論・追跡・描画・
GIF書き・MOV書き・CSV書きを1つの関数の中で同時に行っていた。ここでは

* 追跡ループ … この関数
* 書き出し   … :mod:`dolo.export` の Sink
* 描画       … :mod:`dolo.render`

に分離してある。GUI が「どの出力を作るか」を選べるのはこの分離のおかげ。
進捗通知と中断にも対応する。

CSV の中身はリファクタ前と同一になるよう作ってある（`tests/data/` の
ゴールデンで検証）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .device import resolve_device
from .export import CsvSink, Sink
from .geometry import angle_signed
from .render import DEFAULT_RENDERER, CenterTrailRenderer, color_for_id, get_renderer
from .results import FrameResult, TrackingSummary, TrackRecord, VideoMeta
from .tracker import FixedIDTracker

__all__ = ["TrackParams", "track_video", "track_to_csv"]


@dataclass
class TrackParams:
    """追跡のパラメータ。GUI の設定パネルはこれを1対1で編集すればよい。"""

    max_ids: int = 5
    """同時に存在しうる個体数の上限。分かっているなら実際の個体数を入れる。"""
    conf_thres: float = 1e-3
    """YOLO の検出信頼度の下限。低くすると拾いすぎ、高くすると見逃す。"""
    iou_thres: float = 0.45
    """YOLO の NMS の IoU 閾値。"""
    max_age: int = 15
    """この数だけ連続で見失ったトラックを削除する。"""
    n_init: int = 2
    """トラックが確定するのに必要な連続検出数。"""
    dist_thresh: float = 30.0
    """フレーム間の対応付けを許す基本距離（ピクセル）。"""
    head_tail_jump_thresh: float = 50.0
    """head/tail が1フレームでこれ以上動いた対応付けを棄却する。"""
    overlap_thresh: float = 5.0
    """キーポイントがこれより近い検出は潰れているとみなして捨てる。"""
    adaptive_thresh_factor: float = 2.0
    min_confidence: float = 0.3
    frame_skip: int = 1
    """N フレームに1枚だけ処理する。1 なら全フレーム。"""


def _open_video(video_path, start_frame, end_frame):
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"動画を開けません: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise OSError(
            f"動画のメタ情報を取得できません（コーデックを確認してください）: {video_path}"
        )

    start_frame = max(0, start_frame or 0)
    if end_frame is None or end_frame > total:
        end_frame = total
    if end_frame <= start_frame:
        cap.release()
        raise ValueError(
            f"処理範囲が空です: start_frame={start_frame}, end_frame={end_frame}, total={total}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    meta = VideoMeta(
        path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        total_frames=total,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return cap, meta


def _detections_from_yolo(dets) -> list:
    """YOLO の結果を トラッカーが受け取る形式へ変換する。

    キーポイントの並びは head, middle, tail の順（学習時の定義）。
    信頼度の高い検出から順に並べる（リファクタ前と同じ）。
    """
    keypoints_data = None
    confs: np.ndarray = np.empty(0)

    if dets is not None and dets.boxes is not None and len(dets.boxes) > 0:
        confs = dets.boxes.conf.cpu().numpy()
    if dets is not None and dets.keypoints is not None:
        keypoints_data = dets.keypoints.data.cpu().numpy()

    detection_list = []
    if keypoints_data is None:
        return detection_list

    for det_idx in np.argsort(-confs):
        kp = keypoints_data[det_idx]
        if kp.shape[0] < 3:
            continue
        head_xy, middle_xy, tail_xy = kp[0, :2], kp[1, :2], kp[2, :2]
        detection_list.append(
            (
                middle_xy[0],
                middle_xy[1],
                {"head": head_xy, "middle": middle_xy, "tail": tail_xy},
                det_idx,
            )
        )
    return detection_list


def _predict_one(model, frame, params: TrackParams, device: str):
    """Ultralytics の通常APIと、テスト用の軽量callableの両方を扱う。"""
    if hasattr(model, "predict"):
        return model.predict(
            frame,
            conf=params.conf_thres,
            iou=params.iou_thres,
            device=device,
            verbose=False,
        )[0]
    return model(frame, conf=params.conf_thres, iou=params.iou_thres, device=device)[0]


def _collect_frame_result(tracker: FixedIDTracker, frame_idx: int) -> FrameResult:
    """確定済みトラックから、このフレームの出力行を作る。"""
    result = FrameResult(frame_idx=frame_idx)

    for track_id, trk in tracker.tracks.items():
        if not (trk.is_confirmed() and not trk.is_deleted() and trk.last_keypoints is not None):
            continue

        head = trk.last_keypoints["head"]
        middle = trk.last_keypoints["middle"]
        tail = trk.last_keypoints["tail"]

        # 前回出力時からの移動量。トラックに前回値を憶えさせている。
        dist_moved = trk.cumulative_distance - getattr(trk, "_prev_cumdist", 0.0)
        trk._prev_cumdist = trk.cumulative_distance

        result.tracks.append(
            TrackRecord(
                track_id=track_id,
                head=(float(head[0]), float(head[1])),
                middle=(float(middle[0]), float(middle[1])),
                tail=(float(tail[0]), float(tail[1])),
                angle=angle_signed(tail, middle, head),
                dist_moved=float(dist_moved),
                confidence=float(trk.confidence),
                time_since_update=int(trk.time_since_update),
            )
        )
    return result


def track_video(
    video_path: str | Path,
    model_path: str | Path,
    sinks: list[Sink],
    params: TrackParams | None = None,
    device: str | int | None = "auto",
    start_frame: int = 0,
    end_frame: int | None = None,
    renderer: str = DEFAULT_RENDERER,
    progress: Callable[[int, int], None] | None = None,
    cancel=None,
    log: Callable[[str], None] | None = None,
) -> TrackingSummary:
    """動画を推論・追跡し、結果を ``sinks`` へ流す。

    Parameters
    ----------
    video_path, model_path
        入力動画と学習済み重み。
    sinks
        書き出し先のリスト。空でも動く（何も出力せず統計だけ返る）。
        :func:`dolo.export.build_sinks` で GUI の選択から組み立てられる。
    params
        追跡パラメータ。None なら既定値。
    device
        ``"auto"`` / ``"cuda:0"`` / ``"mps"`` / ``"cpu"`` など。
        使えないものを指定すると警告して代替へフォールバックする。
    progress
        ``progress(done, total)`` が毎フレーム呼ばれる。GUI の進捗バー用。
    cancel
        ``is_set()`` を持つオブジェクト（``threading.Event`` 等）。
        True になった時点でループを抜け、Sink を正しく閉じて戻る。
    log
        ログ1行を受け取る関数。None なら何も出さない。

    Returns
    -------
    TrackingSummary

    Notes
    -----
    可視出力（GIF/MOV）を1つも指定しない場合、描画処理は**一切実行されない**。
    CSV だけ欲しいときの速度差は大きい。
    """
    params = params or TrackParams()
    say = log or (lambda _msg: None)
    started = time.time()

    model_path = Path(model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"モデル重みが見つかりません: {model_path}")

    device = resolve_device(device)
    say(f"デバイス: {device}")

    cap, meta = _open_video(video_path, start_frame, end_frame)
    say(
        f"動画: {meta.width}x{meta.height}, {meta.fps:.2f} fps, "
        f"フレーム {meta.start_frame}-{meta.end_frame} / {meta.total_frames}"
    )

    summary = TrackingSummary()
    opened_sinks: list[Sink] = []

    try:
        from ultralytics import YOLO

        say(f"モデル: {model_path}")
        model = YOLO(str(model_path))
        model.to(device)

        tracker = FixedIDTracker(
            max_ids=params.max_ids,
            max_age=params.max_age,
            n_init=params.n_init,
            dist_thresh=params.dist_thresh,
            head_tail_jump_thresh=params.head_tail_jump_thresh,
            overlap_thresh=params.overlap_thresh,
            adaptive_thresh_factor=params.adaptive_thresh_factor,
            min_confidence=params.min_confidence,
        )

        needs_image = any(getattr(s, "needs_image", False) for s in sinks)
        sink_renderers: dict[int, tuple[tuple[str, int | None], Callable]] = {}
        renderers: dict[tuple[str, int | None], Callable] = {}
        for sink in sinks:
            if not getattr(sink, "needs_image", False):
                continue
            mode = getattr(sink, "render_mode", None) or renderer
            trail_frames = int(getattr(sink, "trail_frames", 30)) if mode == "center" else None
            key = (mode, trail_frames)
            if key not in renderers:
                renderers[key] = (
                    CenterTrailRenderer(trail_frames or 0)
                    if mode == "center"
                    else get_renderer(mode)
                )
            sink_renderers[id(sink)] = (key, renderers[key])
        if not needs_image:
            say("可視出力なし → 描画をスキップします")
        else:
            say("動画描画: " + ", ".join(sorted({key[0] for key in renderers})))

        for sink in sinks:
            sink.open(meta)
            opened_sinks.append(sink)

        id_to_color: dict[int, tuple] = {}
        total_to_do = meta.n_frames_to_process
        frame_idx = meta.start_frame

        while frame_idx < meta.end_frame:
            if cancel is not None and cancel.is_set():
                summary.cancelled = True
                say("中断要求を受け付けました")
                break

            ok, frame = cap.read()
            if not ok:
                break

            if params.frame_skip > 1 and frame_idx % params.frame_skip != 0:
                frame_idx += 1
                continue

            detections = _detections_from_yolo(
                _predict_one(model, frame, params=params, device=device)
            )
            tracker.update(detections)

            result = _collect_frame_result(tracker, frame_idx)

            if needs_image:
                for track in result.tracks:
                    if track.track_id not in id_to_color:
                        id_to_color[track.track_id] = color_for_id(track.track_id)

            rendered: dict[tuple[str, int | None], np.ndarray] = {}
            for sink in sinks:
                image = None
                if getattr(sink, "needs_image", False):
                    key, draw = sink_renderers[id(sink)]
                    if key not in rendered:
                        rendered[key] = draw(frame, result, id_to_color)
                    image = rendered[key]
                sink.write(result, image)

            summary.frames_processed += 1
            summary.rows_written += len(result.tracks)
            summary.ids_seen.update(result.track_ids)

            if progress is not None:
                progress(frame_idx - meta.start_frame + 1, total_to_do)

            frame_idx += 1
    finally:
        cap.release()
        close_errors = []
        for sink in reversed(opened_sinks):
            try:
                sink.close()
            except Exception as exc:  # pragma: no cover - コーデック依存
                close_errors.append(f"{type(sink).__name__}: {exc}")
        if close_errors:
            say("出力を閉じる際の警告: " + "; ".join(close_errors))

    summary.elapsed_sec = time.time() - started
    say(str(summary))
    return summary


def track_to_csv(
    video_path: str | Path,
    model_path: str | Path,
    output_csv_path: str | Path,
    *,
    device: str | int | None = "auto",
    start_frame: int = 0,
    end_frame: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel=None,
    log: Callable[[str], None] | None = None,
    **param_overrides,
) -> TrackingSummary:
    """CSV だけ出力する簡易版。描画を行わないので速い。

    ``param_overrides`` は :class:`TrackParams` のフィールド名で渡す。
    後方互換のため ``max_missing_frames`` は ``max_age`` の別名として受け付ける。
    """
    if "max_missing_frames" in param_overrides:
        param_overrides["max_age"] = param_overrides.pop("max_missing_frames")

    known = set(TrackParams.__dataclass_fields__)
    unknown = set(param_overrides) - known
    if unknown:
        raise TypeError(f"未知のパラメータ: {sorted(unknown)}。使えるのは {sorted(known)}")

    return track_video(
        video_path=video_path,
        model_path=model_path,
        sinks=[CsvSink(output_csv_path)],
        params=TrackParams(**param_overrides),
        device=device,
        start_frame=start_frame,
        end_frame=end_frame,
        progress=progress,
        cancel=cancel,
        log=log,
    )
