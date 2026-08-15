"""移行期の差分テスト: 旧 `scripts/functions_deepsort.py` と `dolo` の等価性を保証する。

リファクタが挙動を変えていないことを機械的に検証するための一時的なテスト。
旧ファイルを削除したらこのテストも役目を終える（存在しなければ自動 skip）。

旧ファイルはモジュール先頭で ultralytics を import するため、ここではスタブを
差し込んで読み込む。トラッカー本体は ultralytics に依存していないので問題ない。
"""

from __future__ import annotations

import importlib
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dolo.geometry import angle_signed, check_head_tail_jump, is_suspicious_detection
from dolo.tracker import FixedIDTracker

LEGACY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "functions_deepsort.py"

pytestmark = pytest.mark.skipif(
    not LEGACY_PATH.exists(),
    reason="旧 scripts/functions_deepsort.py が無い（移行完了後は正常）",
)


@pytest.fixture(scope="module")
def legacy():
    """重い依存をスタブして旧モジュールを読み込む。"""
    originals = {name: sys.modules.get(name) for name in ("ultralytics", "imageio")}
    for name in ("ultralytics", "imageio"):
        stub = types.ModuleType(name)
        stub.YOLO = object
        stub.get_writer = lambda *a, **k: None
        sys.modules[name] = stub

    sys.path.insert(0, str(LEGACY_PATH.parent))
    try:
        yield importlib.import_module("functions_deepsort")
    finally:
        sys.path.pop(0)
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


PARAMS = dict(
    max_ids=6,
    max_age=15,
    n_init=2,
    dist_thresh=30.0,
    head_tail_jump_thresh=50.0,
    overlap_thresh=5.0,
    adaptive_thresh_factor=2.0,
    min_confidence=0.3,
)


def _make_stream(seed, n_frames=120, n_obj=4, width=640, height=480):
    """検出欠損や向きの揺らぎを含む、現実に近い検出列を合成する。"""
    rng = random.Random(seed)
    pos = [[rng.uniform(50, width - 50), rng.uniform(50, height - 50)] for _ in range(n_obj)]
    vel = [[rng.uniform(-6, 6), rng.uniform(-6, 6)] for _ in range(n_obj)]

    frames = []
    for _ in range(n_frames):
        dets = []
        for i in range(n_obj):
            if rng.random() < 0.12:  # 検出欠損
                continue
            pos[i][0] += vel[i][0]
            pos[i][1] += vel[i][1]
            if not 0 < pos[i][0] < width:
                vel[i][0] *= -1
            if not 0 < pos[i][1] < height:
                vel[i][1] *= -1

            cx, cy = pos[i]
            ang = rng.uniform(0, 2 * np.pi)
            offset = np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(6, 18)
            center = np.array([cx, cy])
            dets.append(
                (cx, cy, {"head": center + offset, "middle": center, "tail": center - offset}, i)
            )
        rng.shuffle(dets)
        frames.append(dets)
    return frames


def _snapshot(tracker):
    """比較に使うトラッカーの完全な状態。"""
    tracks = sorted(
        (
            tid,
            trk.state,
            round(float(trk.mean[0]), 6),
            round(float(trk.mean[1]), 6),
            round(float(trk.mean[2]), 6),
            round(float(trk.mean[3]), 6),
            trk.hits,
            trk.age,
            trk.time_since_update,
            round(float(trk.confidence), 9),
            round(float(trk.cumulative_distance), 6),
        )
        for tid, trk in tracker.tracks.items()
    )
    return tracks, list(tracker.available_ids)


def _copy(dets):
    return [(cx, cy, {k: v.copy() for k, v in kd.items()}, i) for cx, cy, kd, i in dets]


@pytest.mark.parametrize("seed", range(10))
def test_tracker_state_matches_legacy_frame_by_frame(legacy, seed):
    """全フレームでトラッカーの内部状態が旧実装と完全に一致する。"""
    old = legacy.FixedIDTracker(**PARAMS)
    new = FixedIDTracker(**PARAMS)

    for frame_idx, dets in enumerate(_make_stream(seed)):
        old.update(_copy(dets))
        new.update(_copy(dets))
        assert _snapshot(old) == _snapshot(new), f"seed={seed} frame={frame_idx} で乖離"


def test_angle_matches_legacy(legacy):
    """旧実装で実際に使われていたのは符号あり（arctan2）版である。"""
    rng = np.random.default_rng(7)
    for _ in range(2000):
        tail, middle, head = rng.uniform(-200, 200, size=(3, 2))
        assert legacy.calculate_angle_between_vectors(tail, middle, head) == pytest.approx(
            angle_signed(tail, middle, head), abs=1e-9
        )


def test_suspicious_detection_matches_legacy(legacy):
    rng = np.random.default_rng(11)
    for _ in range(2000):
        head, middle, tail = rng.uniform(-30, 30, size=(3, 2))
        assert legacy.is_suspicious_detection(head, middle, tail, 5.0) == is_suspicious_detection(
            head, middle, tail, 5.0
        )


def test_head_tail_jump_matches_legacy(legacy):
    rng = np.random.default_rng(13)
    for _ in range(2000):
        head, middle, tail = rng.uniform(-200, 200, size=(3, 2))
        old_kd = {"head": head, "middle": middle, "tail": tail}
        new_kd = {
            "head": head + rng.uniform(-80, 80, 2),
            "middle": middle,
            "tail": tail + rng.uniform(-80, 80, 2),
        }
        assert legacy.check_head_tail_jump(old_kd, new_kd, 50.0) == check_head_tail_jump(
            old_kd, new_kd, 50.0
        )
