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
        self.std_process = 1.0
        self.std_measure = 1.0

    def initiate(self, cx, cy):
        mean = np.array([cx, cy, 0, 0], dtype=np.float32)
        cov = np.eye(4, dtype=np.float32) * 100.0
        return mean, cov

    def predict(self, mean, cov, dt=1.0):
        F = np.eye(4, dtype=np.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        Q = np.eye(4, dtype=np.float32) * (self.std_process**2)
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
    def __init__(self, track_id, mean, cov, max_age=30, n_init=3):
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

    def is_tentative(self):
        return self.state == "Tentative"

    def is_confirmed(self):
        return self.state == "Confirmed"

    def is_deleted(self):
        return self.state == "Deleted"

    def mark_missed(self):
        self.time_since_update += 1
        if self.is_tentative():
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
        old_cx, old_cy = self.last_center
        dist = np.hypot(cx - old_cx, cy - old_cy)
        self.cumulative_distance += dist
        self.last_center = (cx, cy)
        self.last_keypoints = keypoints
        self.time_since_update = 0
        self.hits += 1
        if self.state == "Tentative" and self.hits >= self.n_init:
            self.state = "Confirmed"


class FixedIDTracker:
    """
    Reuses IDs [1..max_ids], limited # of tracks
    With stricter gating to avoid big jumps.
    """

    def __init__(
        self,
        max_ids=5,
        max_age=30,
        n_init=3,
        dist_thresh=20.0,
        head_tail_jump_thresh=50.0,
        overlap_thresh=5.0,
    ):
        self.kf = KalmanFilter()
        self.tracks = {}
        self.max_ids = max_ids
        self.available_ids = list(range(1, max_ids + 1))
        self.max_age = max_age
        self.n_init = n_init
        # TIGHTER gating => 20.0 (was 50.0)
        self.dist_thresh = dist_thresh
        self.head_tail_jump_thresh = head_tail_jump_thresh
        self.overlap_thresh = overlap_thresh

    def predict_all(self):
        for tid, trk in self.tracks.items():
            trk.predict(self.kf)

    def update(self, detections):
        """
        detections: list of (cx, cy, kdict, det_idx)
        1) skip suspicious ones (head==middle==tail)
        2) build cost matrix
        3) do Hungarian
        4) post-check big head/tail jumps => skip those matches
        5) unmatched => missing or new track
        6) remove deleted => return ID
        """
        self.predict_all()
        # ### NEW ### skip suspicious
        filtered_detections = []
        for cx, cy, kdict, d_idx in detections:
            if is_suspicious_detection(
                kdict["head"], kdict["middle"], kdict["tail"], self.overlap_thresh
            ):
                # skip
                continue
            filtered_detections.append((cx, cy, kdict, d_idx))
        detections = filtered_detections

        active_ids = [tid for tid, trk in self.tracks.items() if not trk.is_deleted()]

        if len(active_ids) == 0 and len(detections) > 0:
            for cx, cy, kdict, _ in detections:
                if len(self.available_ids) > 0:
                    new_id = self.available_ids.pop(0)
                    self._initiate_track(new_id, cx, cy, kdict)
        elif len(active_ids) > 0 and len(detections) > 0:
            cost_mat = np.zeros((len(active_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(active_ids):
                trk = self.tracks[tid]
                if trk.is_deleted():
                    cost_mat[i, :] = 999999.0
                    continue
                cxp, cyp = trk.mean[0], trk.mean[1]
                for j, (cx, cy, _, _) in enumerate(detections):
                    dist_val = np.hypot(cxp - cx, cyp - cy)
                    cost_mat[i, j] = dist_val

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                dist_val = cost_mat[r, c]
                if dist_val <= self.dist_thresh:
                    # candidate match
                    track_id = active_ids[r]
                    cx_det, cy_det, kdict, _ = detections[c]
                    # ### NEW ### check big head/tail jump
                    old_kdict = self.tracks[track_id].last_keypoints
                    if not check_head_tail_jump(
                        old_kdict, kdict, self.head_tail_jump_thresh
                    ):
                        # too big head/tail jump => skip
                        continue
                    # if okay => finalize match
                    self.tracks[track_id].update_track(self.kf, cx_det, cy_det, kdict)
                    matched_tracks.add(track_id)
                    matched_dets.add(c)

            # unmatched tracks => missed
            for tid in active_ids:
                if tid not in matched_tracks:
                    self.tracks[tid].mark_missed()
            # unmatched det => new track if ID available
            for j, det in enumerate(detections):
                if j not in matched_dets:
                    if len(self.available_ids) > 0:
                        cx, cy, kdict, _ = det
                        new_id = self.available_ids.pop(0)
                        self._initiate_track(new_id, cx, cy, kdict)

        else:
            # no detections => mark all missed
            for tid in active_ids:
                self.tracks[tid].mark_missed()

        # remove deleted => free ID
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


##############################################
# 3) Main function: process_video_to_gif_with_angles
##############################################


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
    max_ids=5,  # how many fish IDs? 1..max_ids
    max_age=50,
    n_init=3,
    dist_thresh=20.0,
    head_tail_jump_thresh=30.0,  ### NEW param
    overlap_thresh=1.0           ### NEW param
):
    """
    1) Loads YOLO Pose model.
    2) Processes frames from start_frame..end_frame, skipping frame_skip.
    3) For each frame:
       - YOLO Pose -> keypoints for each detection.
       - We pass them to `FixedIDTracker` that reuses IDs from [1..max_ids].
       - For each confirmed track, we compute angle, log (ID => center?), and write CSV & annotate.

    CSV columns:
      Frame, Track_ID, Head_X, Head_Y, Middle_X, Middle_Y, Tail_X, Tail_Y,
      Angle, DistMoved

    We also print logs like:
      [LOG] Frame=12, ID=1 => center=(100.2, 205.7)

    If we have more than `max_ids` active tracks, we do NOT create new ones until
    one is deleted (i.e., ID is freed).
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

    # 2) Initialize tracker
    tracker = FixedIDTracker(
        max_ids=max_ids,
        max_age=max_age,
        n_init=n_init,
        dist_thresh=dist_thresh,
        head_tail_jump_thresh=head_tail_jump_thresh,
        overlap_thresh=overlap_thresh
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
        ]
    )

    # MOV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for .mov
    ret, sample_frame = cap.read()
    if not ret:
        print(f"[ERROR] Could not read first frame from {video_path}")
        return
    mov_writer = cv2.VideoWriter(output_mov_path, fourcc, 10, (sample_frame.shape[1], sample_frame.shape[0]))

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
            confs = dets.boxes.conf.cpu().numpy()  # (N,)
        if dets and dets.keypoints is not None:
            keypoints_data = dets.keypoints.data.cpu().numpy()  # (N, #kpts, 3)

        # Build detection list => (cx, cy, keypoints, det_idx)
        # We'll label each detection in the order it appears, or sorted by conf
        detection_list = []
        if keypoints_data is not None:
            # Sort by conf desc, so we pick highest confidence first
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

        # We'll also do a log print of ID => center
        # for each track
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

                # Dist moved is the last segment's distance
                # We'll do: dist_moved = difference since last update
                dist_moved = getattr(trk, "_last_step_dist", 0.0)
                # but we didn't store that per-step. Let's approximate from
                # difference in cumulative distance (like before):
                dist_moved = trk.cumulative_distance - getattr(
                    trk, "_prev_cumdist", 0.0
                )
                trk._prev_cumdist = trk.cumulative_distance

                # For printing the center (cx,cy) from track's Kalman mean
                cx_kf = float(trk.mean[0])
                cy_kf = float(trk.mean[1])

                print(
                    f"[LOG] Frame={frame_idx}, ID={track_id} => center=({cx_kf:.2f},{cy_kf:.2f}) Distance={dist_moved:.2f}"
                )

                # Write CSV
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
                    ]
                )

                # For annotation
                keypoints_list.append(trk.last_keypoints)
                ids_list.append(track_id)
                angles_list.append(angle_val)

                # color
                if track_id not in id_to_color:
                    cidx = len(id_to_color) % 10
                    color = plt.get_cmap("tab20",10)(cidx)[:3]
                    id_to_color[track_id] = color

        # 4.4) Annotate frame
        if len(keypoints_list) > 0:
            annotated_frame = annotate_frame_with_keypoints(
                frame, keypoints_list, ids_list, angles_list, frame_idx, id_to_color
            )
        else:
            annotated_frame = frame

        # 4.5) Write to GIF
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        gif_writer.append_data(annotated_rgb)
        mov_writer.write(annotated_rgb)

        frame_idx += 1

    # Cleanup
    gif_writer.close()
    csv_file.close()
    mov_writer.release()
    cap.release()
    print(f"[INFO] Done. GIF={output_gif_path}, CSV={output_csv_path}, MOV={output_mov_path}")
