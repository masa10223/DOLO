"""キーポイント幾何のプリミティブ。

このモジュールは **numpy にしか依存しない**。torch / ultralytics / opencv を
読み込まないので、GPU もモデルも無い環境（CI など）で import してテストできる。

--------------------------------------------------------------------------
角度の定義について（重要）
--------------------------------------------------------------------------
リファクタ前のコードには `calculate_angle_between_vectors` が4箇所にあり、
**2つの異なる定義**が混在していた:

* 符号なし  ``arccos(dot)``           → 範囲 [0, 180]
* 符号あり  ``arctan2(cross, dot)``   → 範囲 [-180, 180]

`functions_deepsort.py` では同名の関数が同一ファイル内で二重定義されており、
Python の仕様上 **後に書かれた符号ありの定義が実際に使われていた**。
一方、解析側 (`summarize_and_plot_interactions.py`) は符号なしを使っている。

つまり追跡が出力する CSV の ``Angle`` 列は符号あり [-180, 180] である。
本モジュールでは両者を別名で明示的に定義し、既存の挙動を変えないため
``calculate_angle_between_vectors`` は符号あり版のエイリアスとする。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cross2d",
    "angle_signed",
    "angle_unsigned",
    "calculate_angle_between_vectors",
    "is_suspicious_detection",
    "is_suspicious_detection_all_close",
    "check_head_tail_jump",
    "determine_head_tail_by_angle",
]


def cross2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """2次元ベクトルの外積（スカラー）。

    ``np.cross`` は 2次元入力が NumPy 2.0 で非推奨・将来削除されるため、
    同じ値を返す実装をここに持つ。値は完全に一致するので挙動は変わらない。
    """
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def angle_signed(tail, middle, head) -> float:
    """tail→middle と middle→head のなす角。符号あり、範囲 [-180, 180] 度。

    正が反時計回り。追跡が出力する CSV の ``Angle`` 列はこちら。
    """
    tail = np.asarray(tail, dtype=float)
    middle = np.asarray(middle, dtype=float)
    head = np.asarray(head, dtype=float)

    v1 = middle - tail
    v2 = head - middle
    return float(np.degrees(np.arctan2(cross2d(v1, v2), float(np.dot(v1, v2)))))


def angle_unsigned(tail, middle, head) -> float:
    """tail→middle と middle→head のなす角。符号なし、範囲 [0, 180] 度。

    どちらかのベクトルが長さ0の場合は 0.0 を返す。解析側が使う定義。
    """
    tail = np.asarray(tail, dtype=float)
    middle = np.asarray(middle, dtype=float)
    head = np.asarray(head, dtype=float)

    v1 = middle - tail
    v2 = head - middle
    mag1 = float(np.linalg.norm(v1))
    mag2 = float(np.linalg.norm(v2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    cos_theta = float(np.dot(v1, v2)) / (mag1 * mag2)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


# 既存挙動との互換エイリアス（追跡パイプラインが使っていたのは符号あり版）
calculate_angle_between_vectors = angle_signed


def is_suspicious_detection(head, middle, tail, overlap_thresh: float = 5.0) -> bool:
    """検出が潰れている（キーポイントが重なっている）かを判定する。

    **いずれか1組**でも ``overlap_thresh`` より近ければ True（＝棄却する）。

    注意: リファクタ前は同名関数が二重定義されており、先に定義された
    「3組すべてが近いときだけ True」の版は上書きされて使われていなかった。
    ここでは実際に動いていた OR 版を採用する。AND 版が必要なら
    :func:`is_suspicious_detection_all_close` を使うこと。
    """
    head = np.asarray(head, dtype=float)
    middle = np.asarray(middle, dtype=float)
    tail = np.asarray(tail, dtype=float)

    if np.linalg.norm(head - middle) < overlap_thresh:
        return True
    if np.linalg.norm(middle - tail) < overlap_thresh:
        return True
    if np.linalg.norm(head - tail) < overlap_thresh:
        return True
    return False


def is_suspicious_detection_all_close(head, middle, tail, overlap_thresh: float = 5.0) -> bool:
    """3組すべてが ``overlap_thresh`` より近いときのみ True（旧・未使用の定義）。"""
    head = np.asarray(head, dtype=float)
    middle = np.asarray(middle, dtype=float)
    tail = np.asarray(tail, dtype=float)

    return bool(
        np.linalg.norm(head - middle) < overlap_thresh
        and np.linalg.norm(middle - tail) < overlap_thresh
        and np.linalg.norm(head - tail) < overlap_thresh
    )


def check_head_tail_jump(old_kdict, new_kdict, thresh: float = 50.0) -> bool:
    """head/tail が1フレームで飛びすぎていないかを判定する。

    Returns
    -------
    bool
        True なら妥当（対応付けを受け入れてよい）。``old_kdict`` が None の
        場合は比較対象が無いので True。
    """
    if old_kdict is None:
        return True

    old_head = np.asarray(old_kdict["head"], dtype=float)
    old_tail = np.asarray(old_kdict["tail"], dtype=float)
    new_head = np.asarray(new_kdict["head"], dtype=float)
    new_tail = np.asarray(new_kdict["tail"], dtype=float)

    head_dist = float(np.linalg.norm(old_head - new_head))
    tail_dist = float(np.linalg.norm(old_tail - new_tail))
    return head_dist < thresh and tail_dist < thresh


def determine_head_tail_by_angle(head, middle, tail):
    """head と tail の割り当てが正しいかを、関節角の大きさで判定して返す。

    tail-middle-head の角が鈍い（＝より真っ直ぐ）側を正しい向きとみなす。

    Returns
    -------
    tuple
        ``(head, tail)`` の並び。入れ替えが必要と判断した場合は反転して返す。
    """
    head = np.asarray(head, dtype=float)
    middle = np.asarray(middle, dtype=float)
    tail = np.asarray(tail, dtype=float)

    as_is = angle_unsigned(tail, middle, head)
    swapped = angle_unsigned(head, middle, tail)
    if swapped < as_is:
        return tail, head
    return head, tail
