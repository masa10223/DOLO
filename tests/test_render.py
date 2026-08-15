from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from dolo.render import CenterTrailRenderer, color_for_id, draw_angle, draw_pose, get_renderer
from dolo.results import FrameResult, TrackRecord


def _track(frame: int, x: float = 40.0, track_id: int = 1) -> FrameResult:
    return FrameResult(
        frame_idx=frame,
        tracks=[
            TrackRecord(
                track_id=track_id,
                head=(x + 12, 40.0),
                middle=(x, 40.0),
                tail=(x - 12, 40.0),
                angle=32.5,
                dist_moved=1.0,
                confidence=0.9,
                time_since_update=0,
            )
        ],
    )


def test_pose_and_angle_are_separate_renderers():
    frame = np.zeros((100, 180, 3), dtype=np.uint8)
    colors = {1: color_for_id(1)}

    pose = draw_pose(frame, _track(1), colors)
    angle = draw_angle(frame, _track(1), colors)

    assert pose.shape == frame.shape
    assert np.any(pose != frame)
    assert np.any(angle != pose), "角度動画には姿勢動画とは別の注釈が必要"
    assert get_renderer("pose") is draw_pose
    assert get_renderer("angle") is draw_angle


def test_pose_video_contains_keypoint_legend(monkeypatch):
    import cv2

    labels = []
    original_put_text = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    frame = np.zeros((120, 720, 3), dtype=np.uint8)

    draw_pose(frame, _track(1), {1: color_for_id(1)})

    assert any(label.endswith("Head") for label in labels)
    assert "Middle" in labels
    assert "Triangle" in labels


def test_seaborn_colorblind_palette_is_stable_and_cycles():
    assert color_for_id(1) == pytest.approx((1 / 255, 115 / 255, 178 / 255))
    assert color_for_id(11) == color_for_id(1)


def test_center_trail_prunes_by_frame_age():
    renderer = CenterTrailRenderer(trail_frames=3)
    frame = np.zeros((90, 140, 3), dtype=np.uint8)
    colors = {1: color_for_id(1)}

    renderer(frame, _track(0, 20), colors)
    renderer(frame, _track(1, 30), colors)
    rendered = renderer(frame, _track(4, 60), colors)

    assert renderer.history[1] == [(4, (60, 40))]
    assert np.any(rendered != frame)


def test_zero_center_trail_keeps_full_history():
    renderer = CenterTrailRenderer(trail_frames=0)
    frame = np.zeros((90, 140, 3), dtype=np.uint8)
    colors = {1: color_for_id(1)}

    renderer(frame, _track(0, 20), colors)
    renderer(frame, _track(100, 60), colors)

    assert [item[0] for item in renderer.history[1]] == [0, 100]


def test_negative_center_trail_is_rejected():
    with pytest.raises(ValueError, match="0以上"):
        CenterTrailRenderer(-1)
