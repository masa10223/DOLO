"""ID を固定数に保つ多個体トラッカー。

numpy と scipy のみに依存する。torch / ultralytics / opencv は不要なので、
モデル無しでユニットテストできる。挙動は `scripts/functions_deepsort.py` の
``FixedIDTracker`` から変更していない。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import check_head_tail_jump, is_suspicious_detection
from .kalman import KalmanFilter, Track

__all__ = ["FixedIDTracker"]


class FixedIDTracker:
    """ID を ``1..max_ids`` の範囲で再利用し、同時トラック数を上限で抑えるトラッカー。

    ゲーティング距離はトラックごとの不確実性と未更新フレーム数に応じて
    動的に広がる（ロスト直後の再捕捉を助けるため）。

    Parameters
    ----------
    max_ids
        同時に存在しうるIDの最大数。個体数が既知ならその値を入れる。
    max_age
        この数だけ連続で未更新になったトラックを削除する。
    n_init
        Tentative から Confirmed へ昇格するのに必要なヒット数。
    dist_thresh
        対応付けの基本距離閾値（ピクセル）。
    head_tail_jump_thresh
        head/tail が1フレームでこれ以上動いた対応付けは棄却する。
    overlap_thresh
        キーポイントがこれより近い検出は潰れているとみなして捨てる。
    adaptive_thresh_factor
        不確実性を距離閾値に上乗せする係数。
    min_confidence
        信頼度がこれを下回ったトラックを削除する。
    """

    def __init__(
        self,
        max_ids: int = 5,
        max_age: int = 15,
        n_init: int = 2,
        dist_thresh: float = 30.0,
        head_tail_jump_thresh: float = 50.0,
        overlap_thresh: float = 5.0,
        adaptive_thresh_factor: float = 2.0,
        min_confidence: float = 0.3,
    ) -> None:
        self.kf = KalmanFilter()
        self.tracks: dict[int, Track] = {}
        self.max_ids = max_ids
        self.available_ids = list(range(1, max_ids + 1))
        self.max_age = max_age
        self.n_init = n_init
        self.dist_thresh = dist_thresh
        self.head_tail_jump_thresh = head_tail_jump_thresh
        self.overlap_thresh = overlap_thresh
        self.adaptive_thresh_factor = adaptive_thresh_factor
        self.min_confidence = min_confidence

    def predict_all(self) -> None:
        for trk in self.tracks.values():
            trk.predict(self.kf)

    def get_adaptive_threshold(self, track: Track) -> float:
        """不確実性と未更新時間に応じて広がる対応付け閾値。"""
        adaptive_thresh = self.dist_thresh + track.get_uncertainty() * self.adaptive_thresh_factor
        time_factor = 1.0 + (track.time_since_update * 0.1)
        return adaptive_thresh * time_factor

    def update(self, detections) -> None:
        """1フレーム分の検出でトラック集合を更新する。

        Parameters
        ----------
        detections
            ``(cx, cy, kdict, det_idx)`` のリスト。``kdict`` は
            ``{"head": (x, y), "middle": (x, y), "tail": (x, y)}``。

        手順:
        1. 潰れている検出を捨てる
        2. 適応的閾値付きのコスト行列を作る
        3. ハンガリアン法で対応付ける
        4. head/tail が飛びすぎた対応は事後的に棄却する
        5. 余った検出は、不確実性の高いトラックへの再対応 → 新規IDの順で処理
        6. 削除されたトラックのIDを解放する
        """
        self.predict_all()

        # 1) 潰れている検出を除去
        detections = [
            (cx, cy, kdict, d_idx)
            for cx, cy, kdict, d_idx in detections
            if not is_suspicious_detection(
                kdict["head"], kdict["middle"], kdict["tail"], self.overlap_thresh
            )
        ]

        # 信頼度が下がりきったトラックを削除対象に
        for trk in self.tracks.values():
            if trk.confidence < self.min_confidence:
                trk.state = "Deleted"

        active_ids = [tid for tid, trk in self.tracks.items() if not trk.is_deleted()]

        if len(active_ids) == 0 and len(detections) > 0:
            for cx, cy, kdict, _ in detections:
                if len(self.available_ids) > 0:
                    new_id = self.available_ids.pop(0)
                    self._initiate_track(new_id, cx, cy, kdict)

        elif len(active_ids) > 0 and len(detections) > 0:
            # 2) コスト行列と閾値行列
            cost_mat = np.zeros((len(active_ids), len(detections)), dtype=np.float32)
            thresh_mat = np.zeros((len(active_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(active_ids):
                trk = self.tracks[tid]
                if trk.is_deleted():
                    cost_mat[i, :] = 999999.0
                    continue
                cxp, cyp = trk.mean[0], trk.mean[1]
                adaptive_thresh = self.get_adaptive_threshold(trk)

                for j, (cx, cy, _, _) in enumerate(detections):
                    cost_mat[i, j] = np.hypot(cxp - cx, cyp - cy)
                    thresh_mat[i, j] = adaptive_thresh

            # 3) ハンガリアン法
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            matched_tracks: set[int] = set()
            matched_dets: set[int] = set()

            for r, c in zip(row_ind, col_ind, strict=False):
                if cost_mat[r, c] > thresh_mat[r, c]:
                    continue

                track_id = active_ids[r]
                cx_det, cy_det, kdict, _ = detections[c]

                # 4) head/tail の飛びを事後チェック
                old_kdict = self.tracks[track_id].last_keypoints
                if old_kdict is not None and not check_head_tail_jump(
                    old_kdict, kdict, self.head_tail_jump_thresh
                ):
                    continue

                self.tracks[track_id].update_track(self.kf, cx_det, cy_det, kdict)
                matched_tracks.add(track_id)
                matched_dets.add(c)

            # 対応付かなかったトラック
            for tid in active_ids:
                if tid not in matched_tracks:
                    self.tracks[tid].mark_missed()

            # 5) 余った検出の救済
            unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
            if unmatched_dets:
                # ロストが長いトラックほど優先して再対応を試みる
                high_uncertainty_tracks = sorted(
                    [
                        (tid, trk)
                        for tid, trk in self.tracks.items()
                        if tid in active_ids
                        and tid not in matched_tracks
                        and trk.time_since_update > 3
                    ],
                    key=lambda x: x[1].get_uncertainty(),
                    reverse=True,
                )

                for j in unmatched_dets:
                    cx, cy, kdict, _ = detections[j]
                    best_match = None
                    best_dist = float("inf")

                    for tid, trk in high_uncertainty_tracks:
                        if trk.is_deleted():
                            continue
                        cxp, cyp = trk.mean[0], trk.mean[1]
                        dist = float(np.hypot(cxp - cx, cyp - cy))
                        extended_thresh = self.get_adaptive_threshold(trk) * 1.5

                        if dist < best_dist and dist < extended_thresh:
                            best_dist = dist
                            best_match = tid

                    if best_match is not None:
                        self.tracks[best_match].update_track(self.kf, cx, cy, kdict)
                        matched_tracks.add(best_match)
                    elif len(self.available_ids) > 0:
                        new_id = self.available_ids.pop(0)
                        self._initiate_track(new_id, cx, cy, kdict)

        else:
            # 検出ゼロ
            for tid in active_ids:
                self.tracks[tid].mark_missed()

        # 6) 削除して ID を解放
        deleted_ids = [tid for tid, trk in self.tracks.items() if trk.is_deleted()]
        for d in deleted_ids:
            del self.tracks[d]
            self.available_ids.append(d)
            self.available_ids.sort()

    def _initiate_track(self, track_id: int, cx: float, cy: float, kdict) -> None:
        mean, cov = self.kf.initiate(cx, cy)
        trk = Track(track_id, mean, cov, self.max_age, self.n_init)
        trk.last_keypoints = kdict
        trk.last_center = (cx, cy)
        self.tracks[track_id] = trk
