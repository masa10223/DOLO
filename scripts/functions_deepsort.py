import cv2
import csv
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


############################
# 1) Utility checks
############################
def is_suspicious_detection(head, middle, tail, overlap_thresh=5.0):
    dist_h_m = np.hypot(head[0] - middle[0], head[1] - middle[1])
    dist_m_t = np.hypot(middle[0] - tail[0], middle[1] - tail[1])
    dist_h_t = np.hypot(head[0] - tail[0], head[1] - tail[1])
    if (
        dist_h_m < overlap_thresh
        and dist_m_t < overlap_thresh
        and dist_h_t < overlap_thresh
    ):
        return True
    return False


def check_head_tail_jump(old_kdict, new_kdict, head_tail_jump_thresh=50.0):
    if old_kdict is None:
        return True
    old_head = old_kdict["head"]
    old_tail = old_kdict["tail"]
    new_head = new_kdict["head"]
    new_tail = new_kdict["tail"]
    dist_head = np.hypot(new_head[0] - old_head[0], new_head[1] - old_head[1])
    dist_tail = np.hypot(new_tail[0] - old_tail[0], new_tail[1] - old_tail[1])
    if dist_head > head_tail_jump_thresh or dist_tail > head_tail_jump_thresh:
        return False
    return True


def calculate_angle_between_vectors(tail, middle, head):
    v1 = middle - tail
    v2 = head - middle
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_theta = dot_product / (mag1 * mag2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)



def annotate_frame_with_keypoints_and_angle(frame, keypoints_list, ids_list, angles_list, frame_idx, id_to_color):
    """フレームにキーポイントとIDを描画"""
    annotated = frame.copy()
    
    for kpts, track_id, angle in zip(keypoints_list, ids_list, angles_list):
        head = kpts["head"].astype(int)
        middle = kpts["middle"].astype(int)
        tail = kpts["tail"].astype(int)
        
        # 色を取得
        color = id_to_color.get(track_id, (0, 255, 0))
        color = tuple(int(c * 255) for c in color)
        
        # キーポイントを描画
        cv2.circle(annotated, tuple(head), 5, color, -1)
        cv2.circle(annotated, tuple(middle), 5, color, -1)
        cv2.circle(annotated, tuple(tail), 5, color, -1)
        
        # 線を描画
        cv2.line(annotated, tuple(tail), tuple(middle), color, 2)
        cv2.line(annotated, tuple(middle), tuple(head), color, 2)
        
        # IDと角度を表示
        cv2.putText(annotated, f"ID:{track_id}", tuple(middle - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(annotated, f"{angle:.1f}°", tuple(middle + [0, 20]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # フレーム番号を表示
    cv2.putText(annotated, f"Frame: {frame_idx}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return annotated



def annotate_frame_with_keypoints(img, keypoints_list, ids_list, angles_list, frame_number, id_to_color):
    """
    Annotate the frame with keypoints and IDs.
    Consistent size matching input video resolution.
    """
    frame_height, frame_width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Fix figure size to match video resolution
    fig, ax = plt.subplots(figsize=(frame_width / 100, frame_height / 100), dpi=100)
    ax.imshow(img_rgb)
    
    # Annotate each keypoint set
    for i, keypoints in enumerate(keypoints_list):
        head = keypoints["head"]
        middle = keypoints["middle"]
        tail = keypoints["tail"]
        consistent_id = ids_list[i]
        theta = angles_list[i]

        color = id_to_color.get(consistent_id, (1.0, 0.0, 0.0))
        ax.scatter(head[0], head[1], color=color, s=60, marker="o")
        ax.scatter(middle[0], middle[1], color=color, s=60, marker="x")
        ax.scatter(tail[0], tail[1], color=color, s=60, marker="^")

    ax.axis("off")
    
    # Add legends
    legend_elements = []
    for cid in sorted(id_to_color.keys()):
        color = id_to_color[cid]
        legend_elements.append(
            Line2D(
                [0], [0], marker="o", color="w", label=f"ID {cid}",
                markerfacecolor=color, markersize=10
            )
        )
    marker_elements = [
        Line2D([0], [0], marker="o", color="gray", label="Head", markerfacecolor="gray", markersize=10),
        Line2D([0], [0], marker="x", color="gray", label="Middle", markersize=10),
        Line2D([0], [0], marker="^", color="gray", label="Tail", markerfacecolor="gray", markersize=10),
    ]

    first_legend = ax.legend(handles=legend_elements, loc="upper right", title="IDs")
    ax.add_artist(first_legend)
    ax.legend(handles=marker_elements, loc="upper left", title="Keypoints")

    # Add frame number
    ax.text(
        50, frame_height - 50, f"Frame: {frame_number}", color="red", fontsize=25, fontweight="bold"
    )

    # Fix consistent layout
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Render and convert Matplotlib output to OpenCV format
    fig.canvas.draw()
    annotated_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated_frame = annotated_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    # Convert back to BGR for OpenCV/GIF consistency
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    return annotated_frame


class KalmanFilter:
    def __init__(self):
        # プロセスノイズを増やして不確実性を適切に扱う
        self.std_process = 3.0  # 1.0 -> 3.0に増加
        self.std_measure = 2.0  # 1.0 -> 2.0に増加
        # 速度の減衰係数を追加（速度が徐々に減少する）
        self.velocity_damping = 0.95

    def initiate(self, cx, cy):
        mean = np.array([cx, cy, 0, 0], dtype=np.float32)
        # 初期共分散を位置については小さく、速度については大きく設定
        cov = np.eye(4, dtype=np.float32)
        cov[0, 0] = 10.0  # x position
        cov[1, 1] = 10.0  # y position
        cov[2, 2] = 100.0  # x velocity
        cov[3, 3] = 100.0  # y velocity
        return mean, cov

    def predict(self, mean, cov, dt=1.0):
        F = np.eye(4, dtype=np.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        # 速度の減衰を適用
        F[2, 2] = self.velocity_damping
        F[3, 3] = self.velocity_damping
        
        # プロセスノイズを動的に調整
        Q = np.eye(4, dtype=np.float32)
        Q[0, 0] = (self.std_process * dt) ** 2
        Q[1, 1] = (self.std_process * dt) ** 2
        Q[2, 2] = (self.std_process * 2) ** 2  # 速度の不確実性を大きめに
        Q[3, 3] = (self.std_process * 2) ** 2
        
        mean_pred = F @ mean
        cov_pred = F @ cov @ F.T + Q
        return mean_pred, cov_pred

    def update(self, mean_pred, cov_pred, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        R = np.eye(2, dtype=np.float32) * (self.std_measure**2)
        S = H @ cov_pred @ H.T + R
        K = cov_pred @ H.T @ np.linalg.inv(S)
        y = z - (H @ mean_pred)
        mean_upd = mean_pred + K @ y
        cov_upd = (np.eye(4, dtype=np.float32) - K @ H) @ cov_pred
        return mean_upd, cov_upd


class Track:
    def __init__(self, track_id, mean, cov, max_age=15, n_init=2):  # max_age: 30->15, n_init: 3->2
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
        # 信頼度スコアを追加
        self.confidence = 1.0
        # 最後の実測位置を記録
        self.last_measured_pos = np.array([mean[0], mean[1]])

    def is_tentative(self):
        return self.state == "Tentative"

    def is_confirmed(self):
        return self.state == "Confirmed"

    def is_deleted(self):
        return self.state == "Deleted"

    def mark_missed(self):
        self.time_since_update += 1
        # 信頼度を減少させる
        self.confidence *= 0.9
        
        # Tentativeトラックは早めに削除
        if self.is_tentative() and self.time_since_update > 2:
            self.state = "Deleted"
        elif self.time_since_update > self.max_age:
            self.state = "Deleted"

    def predict(self, kf: KalmanFilter):
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update_track(self, kf: KalmanFilter, cx, cy, keypoints):
        mean_upd, cov_upd = kf.update(
            self.mean, self.cov, np.array([cx, cy], dtype=np.float32)
        )
        self.mean = mean_upd
        self.cov = cov_upd
        
        # 実測位置を更新
        self.last_measured_pos = np.array([cx, cy])
        
        old_cx, old_cy = self.last_center
        dist = np.hypot(cx - old_cx, cy - old_cy)
        self.cumulative_distance += dist
        self.last_center = (cx, cy)
        self.last_keypoints = keypoints
        self.time_since_update = 0
        self.hits += 1
        
        # 信頼度を回復
        self.confidence = min(1.0, self.confidence * 1.1 + 0.1)
        
        if self.state == "Tentative" and self.hits >= self.n_init:
            self.state = "Confirmed"

    def get_uncertainty(self):
        """位置の不確実性（共分散行列のトレース）を返す"""
        return np.sqrt(self.cov[0, 0] + self.cov[1, 1])


class FixedIDTracker:
    """
    Reuses IDs [1..max_ids], limited # of tracks
    With adaptive gating based on uncertainty.
    """

    def __init__(
        self,
        max_ids=5,
        max_age=15,  # 30 -> 15
        n_init=2,     # 3 -> 2
        dist_thresh=30.0,  # 20.0 -> 30.0 (基本閾値を緩和)
        head_tail_jump_thresh=50.0,
        overlap_thresh=5.0,
        adaptive_thresh_factor=2.0,  # 不確実性に基づく適応的閾値の係数
        min_confidence=0.3,  # 最小信頼度
    ):
        self.kf = KalmanFilter()
        self.tracks = {}
        self.max_ids = max_ids
        self.available_ids = list(range(1, max_ids + 1))
        self.max_age = max_age
        self.n_init = n_init
        self.dist_thresh = dist_thresh
        self.head_tail_jump_thresh = head_tail_jump_thresh
        self.overlap_thresh = overlap_thresh
        self.adaptive_thresh_factor = adaptive_thresh_factor
        self.min_confidence = min_confidence

    def predict_all(self):
        for tid, trk in self.tracks.items():
            trk.predict(self.kf)

    def get_adaptive_threshold(self, track):
        """トラックの不確実性に基づいて適応的な閾値を計算"""
        base_thresh = self.dist_thresh
        uncertainty = track.get_uncertainty()
        # 不確実性が高いほど閾値を大きくする
        adaptive_thresh = base_thresh + uncertainty * self.adaptive_thresh_factor
        # time_since_updateが大きいほど閾値を緩和
        time_factor = 1.0 + (track.time_since_update * 0.1)
        return adaptive_thresh * time_factor

    def update(self, detections):
        """
        detections: list of (cx, cy, kdict, det_idx)
        1) skip suspicious ones (head==middle==tail)
        2) build cost matrix with adaptive thresholds
        3) do Hungarian
        4) post-check big head/tail jumps => skip those matches
        5) unmatched => missing or new track
        6) remove deleted => return ID
        """
        self.predict_all()
        
        # Skip suspicious detections
        filtered_detections = []
        for cx, cy, kdict, d_idx in detections:
            if is_suspicious_detection(
                kdict["head"], kdict["middle"], kdict["tail"], self.overlap_thresh
            ):
                continue
            filtered_detections.append((cx, cy, kdict, d_idx))
        detections = filtered_detections

        # 信頼度が低いトラックを削除
        for tid, trk in list(self.tracks.items()):
            if trk.confidence < self.min_confidence:
                trk.state = "Deleted"

        active_ids = [tid for tid, trk in self.tracks.items() if not trk.is_deleted()]

        if len(active_ids) == 0 and len(detections) > 0:
            for cx, cy, kdict, _ in detections:
                if len(self.available_ids) > 0:
                    new_id = self.available_ids.pop(0)
                    self._initiate_track(new_id, cx, cy, kdict)
        elif len(active_ids) > 0 and len(detections) > 0:
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
                    dist_val = np.hypot(cxp - cx, cyp - cy)
                    cost_mat[i, j] = dist_val
                    thresh_mat[i, j] = adaptive_thresh

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                dist_val = cost_mat[r, c]
                adaptive_thresh = thresh_mat[r, c]
                
                if dist_val <= adaptive_thresh:  # 適応的閾値を使用
                    track_id = active_ids[r]
                    cx_det, cy_det, kdict, _ = detections[c]
                    
                    # Check big head/tail jump
                    old_kdict = self.tracks[track_id].last_keypoints
                    if old_kdict is not None and not check_head_tail_jump(
                        old_kdict, kdict, self.head_tail_jump_thresh
                    ):
                        continue
                    
                    self.tracks[track_id].update_track(self.kf, cx_det, cy_det, kdict)
                    matched_tracks.add(track_id)
                    matched_dets.add(c)

            # Unmatched tracks => missed
            for tid in active_ids:
                if tid not in matched_tracks:
                    self.tracks[tid].mark_missed()
                    
            # Unmatched detections => try to match with high-uncertainty tracks first
            unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
            if unmatched_dets:
                # 高い不確実性を持つトラックを優先的に再マッチング
                high_uncertainty_tracks = sorted(
                    [(tid, trk) for tid, trk in self.tracks.items() 
                     if tid in active_ids and tid not in matched_tracks and trk.time_since_update > 3],
                    key=lambda x: x[1].get_uncertainty(),
                    reverse=True
                )
                
                for j in unmatched_dets:
                    cx, cy, kdict, _ = detections[j]
                    best_match = None
                    best_dist = float('inf')
                    
                    for tid, trk in high_uncertainty_tracks:
                        if trk.is_deleted():
                            continue
                        cxp, cyp = trk.mean[0], trk.mean[1]
                        dist = np.hypot(cxp - cx, cyp - cy)
                        extended_thresh = self.get_adaptive_threshold(trk) * 1.5  # より寛容な閾値
                        
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
            # No detections => mark all missed
            for tid in active_ids:
                self.tracks[tid].mark_missed()

        # Remove deleted => free ID
        deleted_ids = []
        for tid, trk in self.tracks.items():
            if trk.is_deleted():
                deleted_ids.append(tid)
        for d in deleted_ids:
            del self.tracks[d]
            self.available_ids.append(d)
            self.available_ids.sort()

    def _initiate_track(self, track_id, cx, cy, kdict):
        mean, cov = self.kf.initiate(cx, cy)
        trk = Track(track_id, mean, cov, self.max_age, self.n_init)
        trk.last_keypoints = kdict
        trk.last_center = (cx, cy)
        self.tracks[track_id] = trk


# ヘルパー関数（元のコードから必要なもの）
def is_suspicious_detection(head, middle, tail, overlap_thresh):
    """頭、中央、尾が重なっているかチェック"""
    if np.linalg.norm(head - middle) < overlap_thresh:
        return True
    if np.linalg.norm(middle - tail) < overlap_thresh:
        return True
    if np.linalg.norm(head - tail) < overlap_thresh:
        return True
    return False


def check_head_tail_jump(old_kdict, new_kdict, thresh):
    """頭尾の大きなジャンプをチェック"""
    if old_kdict is None:
        return True
    old_head = old_kdict["head"]
    old_tail = old_kdict["tail"]
    new_head = new_kdict["head"]
    new_tail = new_kdict["tail"]
    
    head_dist = np.linalg.norm(old_head - new_head)
    tail_dist = np.linalg.norm(old_tail - new_tail)
    
    return head_dist < thresh and tail_dist < thresh


def calculate_angle_between_vectors(tail, middle, head):
    """尾から頭への角度を計算"""
    vec1 = middle - tail
    vec2 = head - middle
    
    dot_product = np.dot(vec1, vec2)
    cross_product = np.cross(vec1, vec2)
    
    angle = np.arctan2(cross_product, dot_product)
    return np.degrees(angle)



def process_video_to_gif_with_angles(
    video_path,
    output_gif_path,
    output_mov_path,
    output_csv_path,
    model_path="yolov11x-pose.pt",
    conf_thres=0.3,
    iou_thres=0.45,
    frame_skip=1,
    device="cuda:3",
    start_frame=0,
    end_frame=None,
    max_ids=5,
    max_age=15,      # 30 -> 15
    n_init=2,        # 3 -> 2
    dist_thresh=30.0,  # 20.0 -> 30.0
    head_tail_jump_thresh=50.0,
    overlap_thresh=5.0,
    adaptive_thresh_factor=2.0,  # 新規追加
    min_confidence=0.3,          # 新規追加
):
    """
    改善されたトラッキング処理
    - より短いmax_ageでフリーズを短縮
    - 適応的な閾値で再検出を改善
    - 信頼度ベースのトラッキング管理
    """
    
    # 0) Initialize YOLO
    model = YOLO(model_path)
    model.to(device)

    # 1) Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    # 2) Initialize tracker with improved parameters
    tracker = FixedIDTracker(
        max_ids=max_ids,
        max_age=max_age,
        n_init=n_init,
        dist_thresh=dist_thresh,
        head_tail_jump_thresh=head_tail_jump_thresh,
        overlap_thresh=overlap_thresh,
        adaptive_thresh_factor=adaptive_thresh_factor,
        min_confidence=min_confidence
    )

    # 3) Prepare outputs
    gif_writer = imageio.get_writer(output_gif_path, mode="I", fps=10)
    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
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
            "Confidence",  # 信頼度を追加
            "TimeSinceUpdate"  # 最終更新からの時間を追加
        ]
    )

    # MOV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ret, sample_frame = cap.read()
    if not ret:
        print(f"[ERROR] Could not read first frame from {video_path}")
        return
    mov_writer = cv2.VideoWriter(output_mov_path, fourcc, 10, 
                                (sample_frame.shape[1], sample_frame.shape[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Reset to start

    # For color assignment per track_id
    id_to_color = {}

    # 4) Frame loop
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # 4.1) YOLO Pose detection
        results = model(frame, conf=conf_thres, iou=iou_thres, device=device)
        dets = results[0]
        
        # If there are boxes & keypoints:
        keypoints_data = None
        confs = []
        if dets and dets.boxes is not None and len(dets.boxes) > 0:
            confs = dets.boxes.conf.cpu().numpy()
        if dets and dets.keypoints is not None:
            keypoints_data = dets.keypoints.data.cpu().numpy()

        # Build detection list
        detection_list = []
        if keypoints_data is not None:
            sorted_inds = np.argsort(-confs)
            for i_d in sorted_inds:
                kp = keypoints_data[i_d]
                if kp.shape[0] < 3:
                    continue
                head_xy = kp[0, :2]
                middle_xy = kp[1, :2]
                tail_xy = kp[2, :2]

                cx, cy = middle_xy
                kdict = {"head": head_xy, "middle": middle_xy, "tail": tail_xy}
                detection_list.append((cx, cy, kdict, i_d))

        # 4.2) Update tracker
        tracker.update(detection_list)

        # 4.3) Gather annotation data from confirmed tracks
        keypoints_list = []
        ids_list = []
        angles_list = []

        for track_id, trk in tracker.tracks.items():
            if (
                trk.is_confirmed()
                and not trk.is_deleted()
                and trk.last_keypoints is not None
            ):
                head = trk.last_keypoints["head"]
                middle = trk.last_keypoints["middle"]
                tail = trk.last_keypoints["tail"]
                angle_val = calculate_angle_between_vectors(tail, middle, head)

                # Distance moved
                dist_moved = trk.cumulative_distance - getattr(trk, "_prev_cumdist", 0.0)
                trk._prev_cumdist = trk.cumulative_distance

                # Center from Kalman filter
                cx_kf = float(trk.mean[0])
                cy_kf = float(trk.mean[1])

                print(
                    f"[LOG] Frame={frame_idx}, ID={track_id} => "
                    f"center=({cx_kf:.2f},{cy_kf:.2f}) "
                    f"Distance={dist_moved:.2f} "
                    f"Confidence={trk.confidence:.2f} "
                    f"TimeSinceUpdate={trk.time_since_update}"
                )

                # Write CSV with additional info
                csv_writer.writerow(
                    [
                        frame_idx,
                        track_id,
                        head[0],
                        head[1],
                        middle[0],
                        middle[1],
                        tail[0],
                        tail[1],
                        angle_val,
                        dist_moved,
                        trk.confidence,
                        trk.time_since_update
                    ]
                )

                # For annotation
                keypoints_list.append(trk.last_keypoints)
                ids_list.append(track_id)
                angles_list.append(angle_val)

                # Color assignment
                if track_id not in id_to_color:
                    cidx = len(id_to_color) % 10
                    color = plt.get_cmap("tab10")(cidx)[:3]
                    id_to_color[track_id] = color

        # 4.4) Annotate frame
        if len(keypoints_list) > 0:
            annotated_frame = annotate_frame_with_keypoints(
                frame, keypoints_list, ids_list, angles_list, frame_idx, id_to_color
            )
        else:
            annotated_frame = frame

        # 4.5) Write to GIF and MOV
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        gif_writer.append_data(annotated_rgb)
        mov_writer.write(annotated_frame)  # BGRのまま書き込み

        frame_idx += 1

    # Cleanup
    gif_writer.close()
    csv_file.close()
    mov_writer.release()
    cap.release()
    print(f"[INFO] Done. GIF={output_gif_path}, CSV={output_csv_path}, MOV={output_mov_path}")
