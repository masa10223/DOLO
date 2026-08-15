"""等速度モデルの Kalman フィルタと、1個体分のトラック状態。

numpy にしか依存しない。挙動は `scripts/functions_deepsort.py` から変更していない。

状態ベクトルは ``[x, y, vx, vy]``、観測は ``[x, y]``。
"""

from __future__ import annotations

import numpy as np

__all__ = ["KalmanFilter", "Track"]


class KalmanFilter:
    """位置と速度を追う等速度モデル。速度には減衰を掛けている。"""

    def __init__(
        self,
        std_process: float = 3.0,
        std_measure: float = 2.0,
        velocity_damping: float = 0.95,
    ) -> None:
        self.std_process = std_process
        self.std_measure = std_measure
        self.velocity_damping = velocity_damping

    def initiate(self, cx: float, cy: float):
        """観測位置から状態を初期化する。速度は不明なので分散を大きく取る。"""
        mean = np.array([cx, cy, 0, 0], dtype=np.float32)
        cov = np.eye(4, dtype=np.float32)
        cov[0, 0] = 10.0  # x position
        cov[1, 1] = 10.0  # y position
        cov[2, 2] = 100.0  # x velocity
        cov[3, 3] = 100.0  # y velocity
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray, dt: float = 1.0):
        F = np.eye(4, dtype=np.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 2] = self.velocity_damping
        F[3, 3] = self.velocity_damping

        Q = np.eye(4, dtype=np.float32)
        Q[0, 0] = (self.std_process * dt) ** 2
        Q[1, 1] = (self.std_process * dt) ** 2
        Q[2, 2] = (self.std_process * 2) ** 2
        Q[3, 3] = (self.std_process * 2) ** 2

        mean_pred = F @ mean
        cov_pred = F @ cov @ F.T + Q
        return mean_pred, cov_pred

    def update(self, mean_pred: np.ndarray, cov_pred: np.ndarray, z: np.ndarray):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        R = np.eye(2, dtype=np.float32) * (self.std_measure**2)
        S = H @ cov_pred @ H.T + R
        K = cov_pred @ H.T @ np.linalg.inv(S)
        y = z - (H @ mean_pred)
        mean_upd = mean_pred + K @ y
        cov_upd = (np.eye(4, dtype=np.float32) - K @ H) @ cov_pred
        return mean_upd, cov_upd


class Track:
    """1個体分のトラック。状態は Tentative → Confirmed → Deleted と遷移する。"""

    def __init__(
        self,
        track_id: int,
        mean: np.ndarray,
        cov: np.ndarray,
        max_age: int = 15,
        n_init: int = 2,
    ) -> None:
        self.track_id = track_id
        self.mean = mean
        self.cov = cov
        self.max_age = max_age
        self.n_init = n_init
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = "Tentative"
        self.last_keypoints = None
        self.last_center = (mean[0], mean[1])
        self.cumulative_distance = 0.0
        self.confidence = 1.0
        self.last_measured_pos = np.array([mean[0], mean[1]])

    def is_tentative(self) -> bool:
        return self.state == "Tentative"

    def is_confirmed(self) -> bool:
        return self.state == "Confirmed"

    def is_deleted(self) -> bool:
        return self.state == "Deleted"

    def mark_missed(self) -> None:
        """このフレームで対応付かなかった場合に呼ぶ。"""
        self.time_since_update += 1
        self.confidence *= 0.9

        # Tentative なトラックは早めに捨てる
        if self.is_tentative() and self.time_since_update > 2:
            self.state = "Deleted"
        elif self.time_since_update > self.max_age:
            self.state = "Deleted"

    def predict(self, kf: KalmanFilter) -> None:
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update_track(self, kf: KalmanFilter, cx: float, cy: float, keypoints) -> None:
        """観測でトラックを更新する。"""
        mean_upd, cov_upd = kf.update(self.mean, self.cov, np.array([cx, cy], dtype=np.float32))
        self.mean = mean_upd
        self.cov = cov_upd

        self.last_measured_pos = np.array([cx, cy])

        old_cx, old_cy = self.last_center
        self.cumulative_distance += float(np.hypot(cx - old_cx, cy - old_cy))
        self.last_center = (cx, cy)
        self.last_keypoints = keypoints
        self.time_since_update = 0
        self.hits += 1

        self.confidence = min(1.0, self.confidence * 1.1 + 0.1)

        if self.state == "Tentative" and self.hits >= self.n_init:
            self.state = "Confirmed"

    def get_uncertainty(self) -> float:
        """位置の不確実性（位置共分散のトレースの平方根）。"""
        return float(np.sqrt(self.cov[0, 0] + self.cov[1, 1]))
