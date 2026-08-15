"""FixedIDTracker と Kalman フィルタのテスト。モデルも動画も不要。

合成した検出列をトラッカーに流し込み、ID の維持・解放・再利用といった
GUI から見える振る舞いを固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from dolo.kalman import KalmanFilter, Track
from dolo.tracker import FixedIDTracker


def det(cx, cy, idx=0, spread=10.0):
    """中心 (cx, cy) の周りに head/middle/tail を並べた検出を作る。"""
    kdict = {
        "head": np.array([cx + spread, cy], dtype=float),
        "middle": np.array([cx, cy], dtype=float),
        "tail": np.array([cx - spread, cy], dtype=float),
    }
    return (float(cx), float(cy), kdict, idx)


def confirmed_ids(tracker):
    return sorted(tid for tid, t in tracker.tracks.items() if t.is_confirmed())


# --------------------------------------------------------------------------
# Kalman フィルタ
# --------------------------------------------------------------------------
def test_kalman_initiate_has_zero_velocity_and_large_velocity_covariance():
    kf = KalmanFilter()
    mean, cov = kf.initiate(100.0, 50.0)
    assert mean.tolist() == [100.0, 50.0, 0.0, 0.0]
    assert cov[2, 2] > cov[0, 0], "速度の初期不確実性は位置より大きいはず"


def test_kalman_predict_advances_position_by_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    mean[2], mean[3] = 5.0, -3.0  # 速度を差し込む
    pred, _ = kf.predict(mean, cov, dt=1.0)
    assert pred[0] == pytest.approx(5.0)
    assert pred[1] == pytest.approx(-3.0)


def test_kalman_predict_damps_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    mean[2] = 10.0
    pred, _ = kf.predict(mean, cov)
    assert pred[2] == pytest.approx(10.0 * kf.velocity_damping)


def test_kalman_predict_grows_uncertainty_and_update_shrinks_it():
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_cov[0, 0] > cov[0, 0]

    _, upd_cov = kf.update(pred_mean, pred_cov, np.array([1.0, 1.0], dtype=np.float32))
    assert upd_cov[0, 0] < pred_cov[0, 0]


def test_kalman_converges_to_constant_velocity_motion():
    """等速直線運動を与えると、速度推定が真値に近づく。"""
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    vx = 4.0
    for step in range(1, 40):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([vx * step, 0.0], dtype=np.float32))
    assert mean[2] == pytest.approx(vx, rel=0.2)


# --------------------------------------------------------------------------
# Track の状態遷移
# --------------------------------------------------------------------------
def test_track_promotes_to_confirmed_after_n_init_hits():
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    trk = Track(1, mean, cov, max_age=15, n_init=2)
    assert trk.is_tentative()
    trk.update_track(kf, 1.0, 1.0, None)
    assert trk.is_confirmed()


def test_tentative_track_is_deleted_quickly():
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    trk = Track(1, mean, cov, max_age=15, n_init=2)
    for _ in range(3):
        trk.mark_missed()
    assert trk.is_deleted(), "Tentative は max_age を待たずに 3 フレームで消える"


def test_confirmed_track_survives_until_max_age():
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    trk = Track(1, mean, cov, max_age=15, n_init=2)
    trk.update_track(kf, 1.0, 1.0, None)
    trk.confidence = 1.0
    for _ in range(15):
        trk.mark_missed()
    assert not trk.is_deleted()
    trk.mark_missed()
    assert trk.is_deleted()


def test_confidence_decays_when_missed_and_recovers_when_updated():
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    trk = Track(1, mean, cov)
    trk.mark_missed()
    assert trk.confidence < 1.0
    trk.update_track(kf, 0.0, 0.0, None)
    assert trk.confidence == pytest.approx(1.0)


def test_cumulative_distance_accumulates():
    kf = KalmanFilter()
    mean, cov = kf.initiate(0.0, 0.0)
    trk = Track(1, mean, cov)
    trk.update_track(kf, 3.0, 4.0, None)  # 原点から 5
    trk.update_track(kf, 3.0, 8.0, None)  # さらに 4
    assert trk.cumulative_distance == pytest.approx(9.0)


# --------------------------------------------------------------------------
# FixedIDTracker
# --------------------------------------------------------------------------
def test_assigns_ids_starting_from_one():
    tracker = FixedIDTracker(max_ids=3)
    tracker.update([det(100, 100, 0), det(200, 200, 1)])
    assert sorted(tracker.tracks.keys()) == [1, 2]


def test_never_exceeds_max_ids():
    tracker = FixedIDTracker(max_ids=2)
    tracker.update([det(100, 100, i) for i in range(10)])
    assert len(tracker.tracks) <= 2
    assert tracker.available_ids == []


def test_keeps_id_stable_for_smoothly_moving_target():
    """滑らかに動く1個体は、同じ ID を保持し続ける。"""
    tracker = FixedIDTracker(max_ids=3, dist_thresh=30.0)
    for step in range(30):
        tracker.update([det(100 + step * 3, 100, 0)])
    assert confirmed_ids(tracker) == [1]
    assert tracker.tracks[1].hits == 30


def test_two_targets_keep_separate_ids_while_passing_far_apart():
    tracker = FixedIDTracker(max_ids=4, dist_thresh=30.0)
    for step in range(20):
        tracker.update([det(100 + step * 2, 100, 0), det(400 - step * 2, 400, 1)])
    assert confirmed_ids(tracker) == [1, 2]


def test_id_is_released_and_reusable_after_track_deletion():
    tracker = FixedIDTracker(max_ids=2, max_age=3)
    tracker.update([det(100, 100, 0)])
    tracker.update([det(103, 100, 0)])
    assert 1 in tracker.tracks

    for _ in range(30):  # 検出が途絶える
        tracker.update([])
    assert tracker.tracks == {}
    assert 1 in tracker.available_ids

    tracker.update([det(800, 800, 0)])
    assert len(tracker.tracks) == 1


def test_collapsed_detection_is_discarded():
    """head/middle/tail が重なった検出はトラックを作らない。"""
    tracker = FixedIDTracker(max_ids=3, overlap_thresh=5.0)
    collapsed = (
        100.0,
        100.0,
        {
            "head": np.array([100.0, 100.0]),
            "middle": np.array([101.0, 100.0]),
            "tail": np.array([102.0, 100.0]),
        },
        0,
    )
    tracker.update([collapsed])
    assert tracker.tracks == {}


def test_detection_beyond_gate_creates_new_track_instead_of_hijacking():
    """遠くに現れた検出は既存トラックを奪わず、新しい ID になる。"""
    tracker = FixedIDTracker(max_ids=4, dist_thresh=20.0, adaptive_thresh_factor=0.0)
    tracker.update([det(100, 100, 0)])
    tracker.update([det(100, 100, 0)])
    tracker.update([det(100, 100, 0), det(2000, 2000, 1)])
    assert len(tracker.tracks) == 2


def test_gate_widens_while_track_is_lost():
    """未更新が続くほど対応付け閾値が広がる（再捕捉を助ける）。"""
    tracker = FixedIDTracker(max_ids=2, dist_thresh=30.0)
    tracker.update([det(100, 100, 0)])
    trk = tracker.tracks[1]
    initial = tracker.get_adaptive_threshold(trk)

    for _ in range(5):
        trk.predict(tracker.kf)
        trk.mark_missed()
    assert tracker.get_adaptive_threshold(trk) > initial


def test_head_tail_flip_is_rejected():
    """head/tail が突然入れ替わった検出は対応付けられない。"""
    tracker = FixedIDTracker(max_ids=2, head_tail_jump_thresh=20.0, dist_thresh=50.0)
    tracker.update([det(100, 100, 0, spread=30.0)])
    tracker.update([det(100, 100, 0, spread=30.0)])
    hits_before = tracker.tracks[1].hits

    flipped = (
        100.0,
        100.0,
        {
            "head": np.array([70.0, 100.0]),  # 元は tail の位置
            "middle": np.array([100.0, 100.0]),
            "tail": np.array([130.0, 100.0]),  # 元は head の位置
        },
        0,
    )
    tracker.update([flipped])
    assert tracker.tracks[1].hits == hits_before, "反転した検出は受け入れられないはず"


def test_no_detections_does_not_crash_and_ages_tracks():
    tracker = FixedIDTracker(max_ids=2)
    tracker.update([det(100, 100, 0)])
    tracker.update([])
    assert tracker.tracks[1].time_since_update >= 1


def test_empty_updates_from_the_start_are_safe():
    tracker = FixedIDTracker(max_ids=2)
    for _ in range(5):
        tracker.update([])
    assert tracker.tracks == {}


def test_available_ids_stay_sorted_and_unique():
    tracker = FixedIDTracker(max_ids=3, max_age=2)
    tracker.update([det(100, 100, 0), det(300, 300, 1)])
    for _ in range(20):
        tracker.update([])
    assert tracker.available_ids == sorted(set(tracker.available_ids))
    assert tracker.available_ids == [1, 2, 3]
