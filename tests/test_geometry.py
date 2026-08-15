"""キーポイント幾何プリミティブのテスト。numpy 以外の依存は不要。"""

from __future__ import annotations

import numpy as np
import pytest

from dolo.geometry import (
    angle_signed,
    angle_unsigned,
    check_head_tail_jump,
    cross2d,
    determine_head_tail_by_angle,
    is_suspicious_detection,
    is_suspicious_detection_all_close,
)


def kp(head, middle, tail):
    return {
        "head": np.array(head, dtype=float),
        "middle": np.array(middle, dtype=float),
        "tail": np.array(tail, dtype=float),
    }


# --------------------------------------------------------------------------
# cross2d — np.cross の 2D 版と値が一致すること（NumPy 2.0 非推奨の回避）
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "v1,v2,expected",
    [
        ((1.0, 0.0), (0.0, 1.0), 1.0),
        ((0.0, 1.0), (1.0, 0.0), -1.0),
        ((1.0, 0.0), (1.0, 0.0), 0.0),
        ((3.0, -2.0), (5.0, 7.0), 3.0 * 7.0 - (-2.0) * 5.0),
    ],
)
def test_cross2d(v1, v2, expected):
    assert cross2d(np.array(v1), np.array(v2)) == pytest.approx(expected)


# --------------------------------------------------------------------------
# 角度
# --------------------------------------------------------------------------
def test_angle_straight_body_is_zero():
    """tail-middle-head が一直線なら、符号あり・なしとも 0 度。"""
    tail, middle, head = (0, 0), (10, 0), (20, 0)
    assert angle_signed(tail, middle, head) == pytest.approx(0.0)
    assert angle_unsigned(tail, middle, head) == pytest.approx(0.0)


def test_angle_signed_is_positive_for_counterclockwise_bend():
    tail, middle, head = (0, 0), (10, 0), (10, 10)
    assert angle_signed(tail, middle, head) == pytest.approx(90.0)


def test_angle_signed_is_negative_for_clockwise_bend():
    tail, middle, head = (0, 0), (10, 0), (10, -10)
    assert angle_signed(tail, middle, head) == pytest.approx(-90.0)


def test_angle_unsigned_loses_the_sign():
    """符号なし版は曲がる向きを区別しない。この差が追跡と解析の食い違いの原因。"""
    ccw = angle_unsigned((0, 0), (10, 0), (10, 10))
    cw = angle_unsigned((0, 0), (10, 0), (10, -10))
    assert ccw == pytest.approx(cw) == pytest.approx(90.0)


def test_angle_signed_and_unsigned_agree_in_magnitude():
    rng = np.random.default_rng(0)
    for _ in range(200):
        tail, middle, head = rng.uniform(-100, 100, size=(3, 2))
        if np.linalg.norm(middle - tail) < 1e-6 or np.linalg.norm(head - middle) < 1e-6:
            continue
        assert abs(angle_signed(tail, middle, head)) == pytest.approx(
            angle_unsigned(tail, middle, head), abs=1e-6
        )


def test_angle_signed_range_is_within_180():
    rng = np.random.default_rng(1)
    for _ in range(200):
        tail, middle, head = rng.uniform(-100, 100, size=(3, 2))
        assert -180.0 <= angle_signed(tail, middle, head) <= 180.0


def test_angle_unsigned_handles_zero_length_vector():
    """長さ0のベクトルで ZeroDivision も NaN も出さず 0.0 を返す。"""
    assert angle_unsigned((5, 5), (5, 5), (10, 10)) == 0.0
    assert angle_unsigned((0, 0), (5, 5), (5, 5)) == 0.0


# --------------------------------------------------------------------------
# 潰れた検出の判定
# --------------------------------------------------------------------------
def test_suspicious_when_any_pair_is_close():
    """OR 版: head と middle だけ近くても棄却される（実際に動いている挙動）。"""
    assert is_suspicious_detection((0, 0), (1, 0), (100, 100), overlap_thresh=5.0) is True


def test_not_suspicious_when_all_pairs_are_far():
    assert is_suspicious_detection((0, 0), (20, 0), (40, 0), overlap_thresh=5.0) is False


def test_and_version_needs_all_pairs_close():
    """AND 版（旧・未使用）は1組だけ近くても棄却しない。両者が別物であることの記録。"""
    head, middle, tail = (0, 0), (1, 0), (100, 100)
    assert is_suspicious_detection(head, middle, tail, 5.0) is True
    assert is_suspicious_detection_all_close(head, middle, tail, 5.0) is False


def test_and_version_rejects_fully_collapsed_detection():
    assert is_suspicious_detection_all_close((0, 0), (1, 1), (2, 0), 5.0) is True


def test_suspicious_threshold_is_strict_inequality():
    """距離がちょうど閾値と等しいときは棄却しない。"""
    assert is_suspicious_detection((0, 0), (5, 0), (50, 50), overlap_thresh=5.0) is False


# --------------------------------------------------------------------------
# head/tail のジャンプ検出
# --------------------------------------------------------------------------
def test_jump_check_accepts_when_no_history():
    assert check_head_tail_jump(None, kp((0, 0), (5, 0), (10, 0)), 50.0) is True


def test_jump_check_accepts_small_motion():
    old = kp((0, 0), (5, 0), (10, 0))
    new = kp((3, 0), (8, 0), (13, 0))
    assert check_head_tail_jump(old, new, 50.0) is True


def test_jump_check_rejects_large_head_motion():
    old = kp((0, 0), (5, 0), (10, 0))
    new = kp((500, 0), (8, 0), (13, 0))
    assert check_head_tail_jump(old, new, 50.0) is False


def test_jump_check_rejects_head_tail_swap():
    """head と tail が入れ替わると両方が大きく動くので棄却される。"""
    old = kp(head=(0, 0), middle=(50, 0), tail=(100, 0))
    new = kp(head=(100, 0), middle=(50, 0), tail=(0, 0))
    assert check_head_tail_jump(old, new, 50.0) is False


# --------------------------------------------------------------------------
# head/tail の向き決定
# --------------------------------------------------------------------------
def test_head_tail_kept_when_already_straight():
    head, tail = determine_head_tail_by_angle((20, 0), (10, 0), (0, 0))
    assert tuple(head) == (20.0, 0.0)
    assert tuple(tail) == (0.0, 0.0)


def test_head_tail_swapped_when_reversed_is_straighter():
    """折り返した配置なら、より真っ直ぐになる並びへ入れ替える。"""
    head, tail = determine_head_tail_by_angle(head=(0, 0), middle=(10, 0), tail=(5, 8))
    original = determine_head_tail_by_angle(head=(0, 0), middle=(10, 0), tail=(5, 8))
    assert np.allclose(head, original[0]) and np.allclose(tail, original[1])
    # 入れ替え後の並びのほうが角度が小さい（＝真っ直ぐ）ことを確認
    from dolo.geometry import angle_unsigned as au

    assert au(tail, (10, 0), head) <= au((5, 8), (10, 0), (0, 0))
