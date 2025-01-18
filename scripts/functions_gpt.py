import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from math import hypot
from shapely.geometry import Point
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from matplotlib.lines import Line2D



#==================================================================
# 1) Kalman Filterの簡易実装
#==================================================================

def init_kalman_filter(x, y):
    """
    (x, y, vx, vy) を状態とする簡単なカルマンフィルタを初期化。
    戻り値: (state, covariance)
    """
    # 状態ベクトル [x, y, vx, vy]
    state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
    # 共分散行列
    P = np.eye(4, dtype=np.float32) * 100.0  # 初期は大きめにとる
    return state, P

def predict_kf(state, P, dt=1.0):
    """
    雑に離散モデルを適用: 
        x' = x + vx*dt
        y' = y + vy*dt
        vx' = vx
        vy' = vy
    """
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ], dtype=np.float32)

    # プロセス雑音行列 (適当)
    Q = np.eye(4, dtype=np.float32) * 0.5

    # 予測ステップ
    state_pred = F @ state
    P_pred = F @ P @ F.T + Q
    return state_pred, P_pred

def update_kf(state_pred, P_pred, z):
    """
    観測 z = [x_obs, y_obs]
    """
    # 観測行列
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    R = np.eye(2, dtype=np.float32) * 5.0  # 観測雑音

    # イノベーション
    y = z - (H @ state_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    state_upd = state_pred + K @ y
    I = np.eye(4, dtype=np.float32)
    P_upd = (I - K @ H) @ P_pred
    return state_upd, P_upd

def get_predicted_xy(state):
    """
    カルマンフィルタの状態ベクトル [x, y, vx, vy] から位置だけ返す。
    """
    return state[0:2]


#==================================================================
# 2) ヘルパー関数 (ユーザ提供のものなど)
#==================================================================

def calculate_angle_between_vectors(T, m, H):
    """
    質問文にあるベクトル角度計算
    """
    T_mg = m - T
    mg_H = H - m
    dot_product = np.dot(T_mg, mg_H)
    magnitude_T_mg = np.linalg.norm(T_mg)
    magnitude_mg_H = np.linalg.norm(mg_H)

    if magnitude_T_mg == 0 or magnitude_mg_H == 0:
        return 0.0

    cos_theta = dot_product / (magnitude_T_mg * magnitude_mg_H)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta)
    return theta_degrees

def annotate_frame_with_keypoints(
    img, keypoints_list, ids_list, angles_list, frame_number, id_to_color
):
    """
    現状維持。
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)

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

    legend_elements = []
    for cid in sorted(id_to_color.keys()):
        color = id_to_color[cid]
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"ID {cid}",
                markerfacecolor=color,
                markersize=10,
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

    ax.text(
        100, 900, f"Frame: {frame_number}", color="red", fontsize=25, fontweight="bold"
    )

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.canvas.draw()
    annotated_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated_frame = annotated_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    return annotated_frame


def determine_head_tail_based_on_angle(head, middle, tail):
    """
    角度から head, tail を場合によっては入れ替え
    """
    vector_head = head - middle
    vector_tail = tail - middle
    angle_head = np.arctan2(vector_head[1], vector_head[0])
    angle_tail = np.arctan2(vector_tail[1], vector_tail[0])
    angle_difference = abs(angle_head - angle_tail)
    angle_difference = angle_difference if angle_difference <= np.pi else 2*np.pi - angle_difference

    if angle_difference < np.pi / 2:
        if np.linalg.norm(vector_head) <= np.linalg.norm(vector_tail):
            return head, middle, tail
        else:
            return tail, middle, head
    else:
        angle_at_middle_head = calculate_angle_at_joint(tail, middle, head)
        angle_at_middle_tail = calculate_angle_at_joint(head, middle, tail)
        if angle_at_middle_head <= angle_at_middle_tail:
            return head, middle, tail
        else:
            return tail, middle, head

def calculate_angle_at_joint(p1, p2, p3):
    vector1 = p1 - p2
    vector2 = p3 - p2
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if magnitude_product == 0:
        return 0.0
    cos_angle = dot_product / magnitude_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees


#==================================================================
# 3) Head/Tail の向きをある程度固定する (移動方向ベース)
#==================================================================
def ensure_head_in_direction_of_accumulated_movement(
    head, middle, tail
):
    """
    移動が小さい場合や初期フレームでは angle-based な決定を行うだけ。
    (厳密には ID単位で orientation_fixed を持つロジックでもいいが、
     ここでは簡単な形にしている)
    """
    # ひとまず角度ベースで判断
    head2, middle2, tail2 = determine_head_tail_based_on_angle(head, middle, tail)
    return head2, middle2, tail2


#==================================================================
# 4) メイン処理: カルマンフィルタ + ハンガリアン + manual_assignments
#==================================================================

def process_video_to_gif_with_angles(
    video_path,
    output_gif_path,
    model_path="./runs/pose/train/weights/best.pt",
    frame_skip=1,
    output_csv_path="./output_positions_angles.csv",
    confidence=0.01,
    distance_threshold=50,  # 大きすぎるマッチングを big jump 扱いする基準
    max_missing_frames=30,
    max_consistent_ids=5,   # 5個のIDを追跡
    start_frame=None,
    end_frame=None,
    manual_assignments=None,
):
    """
    5個のIDを追跡し、keypointsを追跡、必要に応じて manual_assignments で強制割り当て。
    カルマンフィルタを用いてIDごとの位置を予測し、マッチングする。
    """

    if manual_assignments is None:
        manual_assignments = {}

    try:
        model = YOLO(model_path)
        print("[DEBUG] YOLOモデルをロードしました。")
    except Exception as e:
        print(f"[ERROR] YOLOモデルの読み込みエラー: {e}")
        return

    try:
        cap = cv2.VideoCapture(video_path)
        print("[DEBUG] 動画ファイルをオープンしました。")
    except Exception as e:
        print(f"[ERROR] ビデオファイルのオープンエラー: {e}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame is None:
        start_frame = 0
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    # ID管理構造
    # tracks[cid] = {
    #   "state": [x, y, vx, vy],
    #   "cov":   P(4x4),
    #   "missing_count": 0
    #   "last_keypoints": {"head":..., "middle":..., "tail":...},
    #   "last_angle": float
    # }
    tracks = {}

    # 使用可能 ID のセット
    available_ids = set(range(1, max_consistent_ids+1))

    # 色のマッピング
    id_to_color = {}
    color_palette = plt.get_cmap("tab20", 10)
    for i, cid in enumerate(sorted(list(available_ids))):
        color_index = i % color_palette.N
        color = color_palette(color_index)[:3]
        id_to_color[cid] = color

    # GIF, CSVライター
    try:
        gif_writer = imageio.get_writer(output_gif_path, mode="I", fps=10, loop=0)
        print(f"[DEBUG] GIFライター初期化: {output_gif_path}")
    except Exception as e:
        print(f"[ERROR] GIFライター初期化エラー: {e}")
        cap.release()
        return

    try:
        with open(output_csv_path, mode="w", newline="") as csv_file:
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
                ]
            )
            print(f"[DEBUG] CSVファイルオープン: {output_csv_path}")

            while cap.isOpened() and frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARN] フレーム {frame_count}: 読み込み失敗または動画終了")
                    break

                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                #=====================================================
                # 1) 全IDについてカルマンフィルタで予測
                #=====================================================
                predicted_positions = {}
                for cid, data in tracks.items():
                    state_pred, P_pred = predict_kf(data["state"], data["cov"], dt=1.0)
                    tracks[cid]["state_pred"] = state_pred
                    tracks[cid]["cov_pred"]   = P_pred
                    pred_xy = get_predicted_xy(state_pred)
                    predicted_positions[cid] = pred_xy

                #=====================================================
                # 2) YOLO推論
                #=====================================================
                try:
                    results = model(frame, conf=confidence)
                except Exception as e:
                    print(f"[ERROR] フレーム {frame_count}: 推論エラー {e}")
                    frame_count += 1
                    continue

                result = results[0]
                keypoints_data = None
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()

                # YOLOから得られた middle の候補と index をまとめる
                current_positions = []
                current_keypoints_dicts = []
                if keypoints_data is not None:
                    for idx in range(keypoints_data.shape[0]):
                        kp = keypoints_data[idx]
                        if kp.shape[0] >= 3:
                            head_xy = kp[0, :2]
                            middle_xy = kp[1, :2]
                            tail_xy = kp[2, :2]
                            current_positions.append(middle_xy)
                            current_keypoints_dicts.append(
                                {"head": head_xy, "middle": middle_xy, "tail": tail_xy}
                            )

                #=====================================================
                # 3) manual_assignments の処理
                #=====================================================
                frame_manual_map = manual_assignments.get(frame_count, {})
                # {ID: index, ...}

                used_det_indices = set()
                assigned_ids = set()

                # manual_assignments があれば先に割り当て
                for forced_id, forced_idx in frame_manual_map.items():
                    if forced_idx < 0 or forced_idx >= len(current_keypoints_dicts):
                        print(f"[WARN] frame={frame_count}: ID={forced_id} に対する manual index={forced_idx} が範囲外")
                        continue
                    # もし tracks にないIDなら初期化（=新規トラック開始）
                    if forced_id not in tracks:
                        if forced_id in available_ids:
                            # まだ使ってないIDなら確保してトラック作成
                            available_ids.remove(forced_id)
                            x0, y0 = current_positions[forced_idx]
                            s, P = init_kalman_filter(x0, y0)
                            tracks[forced_id] = {
                                "state": s,
                                "cov":   P,
                                "missing_count": 0,
                                "last_keypoints": {},
                                "last_angle": 0.0
                            }
                            print(f"[DEBUG] フレーム {frame_count}: manual_assignments で ID={forced_id} を新規作成")
                        else:
                            # IDが既にどこかで使われているが missing_count で削除された？
                            # or 5個以上追跡済み？
                            print(f"[WARN] フレーム {frame_count}: ID={forced_id} を追加できません(上限/状況不明)")
                            continue

                    used_det_indices.add(forced_idx)
                    assigned_ids.add(forced_id)

                    # 観測を更新
                    obs_xy = current_positions[forced_idx]
                    # KF update
                    state_pred = tracks[forced_id].get("state_pred", tracks[forced_id]["state"])
                    cov_pred   = tracks[forced_id].get("cov_pred",  tracks[forced_id]["cov"])
                    state_upd, cov_upd = update_kf(state_pred, cov_pred, obs_xy)
                    tracks[forced_id]["state"] = state_upd
                    tracks[forced_id]["cov"]   = cov_upd
                    tracks[forced_id]["missing_count"] = 0

                    # head-tail 判定
                    keypoints = current_keypoints_dicts[forced_idx]
                    head, middle, tail = (
                        keypoints["head"],
                        keypoints["middle"],
                        keypoints["tail"],
                    )
                    head, middle, tail = ensure_head_in_direction_of_accumulated_movement(
                        head, middle, tail
                    )
                    angle = calculate_angle_between_vectors(tail, middle, head)

                    tracks[forced_id]["last_keypoints"] = {
                        "head": head,
                        "middle": middle,
                        "tail": tail
                    }
                    tracks[forced_id]["last_angle"] = angle

                    # CSVへ書き込み
                    csv_writer.writerow(
                        [
                            frame_count,
                            forced_id,
                            head[0], head[1],
                            middle[0], middle[1],
                            tail[0], tail[1],
                            angle
                        ]
                    )

                    print(f"[DEBUG] フレーム {frame_count} の manual_assignments で ID={forced_id} に index={forced_idx} を強制割り当て")

                #=====================================================
                # 4) 残りID & 残り検出のマッチング (ハンガリアン)
                #=====================================================
                # まだ割り当てていないID (tracks 内にいるが assigned_ids にいない)
                unmatched_ids = [cid for cid in tracks.keys() if cid not in assigned_ids]
                # まだ割り当てていない YOLO index
                unmatched_det_indices = [i for i in range(len(current_positions)) if i not in used_det_indices]

                if unmatched_ids and unmatched_det_indices:
                    cost_matrix = np.zeros((len(unmatched_ids), len(unmatched_det_indices)), dtype=np.float32)
                    for i, cid in enumerate(unmatched_ids):
                        pred_xy = predicted_positions.get(cid, None)
                        if pred_xy is None:
                            # 未初期化なはずはないが
                            pred_xy = get_predicted_xy(tracks[cid]["state"])
                        for j, det_idx in enumerate(unmatched_det_indices):
                            obs_xy = current_positions[det_idx]
                            dist = np.linalg.norm(obs_xy - pred_xy)
                            cost_matrix[i, j] = dist

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    for r, c in zip(row_ind, col_ind):
                        cid = unmatched_ids[r]
                        det_idx = unmatched_det_indices[c]
                        dist_val = cost_matrix[r, c]

                        # big jump判定
                        if dist_val > distance_threshold:
                            print(f"[DEBUG] フレーム {frame_count}: ID={cid} に対してビッグジャンプ (distance={dist_val:.2f})")

                        # KF update
                        obs_xy = current_positions[det_idx]
                        state_pred = tracks[cid].get("state_pred", tracks[cid]["state"])
                        cov_pred   = tracks[cid].get("cov_pred",  tracks[cid]["cov"])
                        state_upd, cov_upd = update_kf(state_pred, cov_pred, obs_xy)

                        tracks[cid]["state"] = state_upd
                        tracks[cid]["cov"]   = cov_upd
                        tracks[cid]["missing_count"] = 0

                        # head-tail 判定
                        keypoints = current_keypoints_dicts[det_idx]
                        head, middle, tail = (
                            keypoints["head"],
                            keypoints["middle"],
                            keypoints["tail"],
                        )
                        head, middle, tail = ensure_head_in_direction_of_accumulated_movement(
                            head, middle, tail
                        )
                        angle = calculate_angle_between_vectors(tail, middle, head)

                        tracks[cid]["last_keypoints"] = {
                            "head": head,
                            "middle": middle,
                            "tail": tail
                        }
                        tracks[cid]["last_angle"] = angle

                        # CSV書き込み
                        csv_writer.writerow(
                            [
                                frame_count,
                                cid,
                                head[0],
                                head[1],
                                middle[0],
                                middle[1],
                                tail[0],
                                tail[1],
                                angle
                            ]
                        )

                        used_det_indices.add(det_idx)
                        assigned_ids.add(cid)

                        print(f"[DEBUG] フレーム {frame_count}: ID={cid} に YOLO index={det_idx} を割り当て (距離={dist_val:.2f})")

                #=====================================================
                # 5) 残った検出に対して、新規IDを割り当て
                #=====================================================
                leftover_det_indices = [i for i in range(len(current_positions)) if i not in used_det_indices]
                for det_idx in leftover_det_indices:
                    if len(available_ids) > 0:
                        new_id = min(available_ids)
                        available_ids.remove(new_id)
                        obs_xy = current_positions[det_idx]
                        s, P = init_kalman_filter(obs_xy[0], obs_xy[1])
                        tracks[new_id] = {
                            "state": s,
                            "cov":   P,
                            "missing_count": 0,
                            "last_keypoints": {},
                            "last_angle": 0.0
                        }

                        # updateステップ
                        state_pred = s
                        cov_pred   = P
                        state_upd, cov_upd = update_kf(state_pred, cov_pred, obs_xy)
                        tracks[new_id]["state"] = state_upd
                        tracks[new_id]["cov"]   = cov_upd

                        # head-tail
                        keypoints = current_keypoints_dicts[det_idx]
                        head, middle, tail = (
                            keypoints["head"],
                            keypoints["middle"],
                            keypoints["tail"],
                        )
                        head, middle, tail = ensure_head_in_direction_of_accumulated_movement(
                            head, middle, tail
                        )
                        angle = calculate_angle_between_vectors(tail, middle, head)
                        tracks[new_id]["last_keypoints"] = {
                            "head": head,
                            "middle": middle,
                            "tail": tail
                        }
                        tracks[new_id]["last_angle"] = angle

                        csv_writer.writerow(
                            [
                                frame_count,
                                new_id,
                                head[0],
                                head[1],
                                middle[0],
                                middle[1],
                                tail[0],
                                tail[1],
                                angle
                            ]
                        )

                        print(f"[DEBUG] フレーム {frame_count}: 新規ID={new_id} を YOLO index={det_idx} に割り当て")
                    else:
                        # ID枠がない場合は無視
                        pass

                #=====================================================
                # 6) 割り当てられなかったIDの missing_count を増やし、超えたら解放
                #=====================================================
                for cid in list(tracks.keys()):
                    if cid not in assigned_ids:
                        tracks[cid]["missing_count"] += 1
                        if tracks[cid]["missing_count"] > max_missing_frames:
                            # ID解放
                            available_ids.add(cid)
                            del tracks[cid]
                            print(f"[DEBUG] フレーム {frame_count}: ID={cid} を解放 (missing_count 超過)")
                
                #=====================================================
                # 7) GIF用の可視化
                #=====================================================
                # フレームにいるIDだけを可視化
                visible_ids = [cid for cid in tracks.keys() if cid in assigned_ids]
                keypoints_list = []
                ids_list = []
                angles_list = []
                for cid in visible_ids:
                    kp = tracks[cid]["last_keypoints"]
                    if len(kp) == 3:  # head, middle, tail
                        keypoints_list.append(kp)
                        ids_list.append(cid)
                        angles_list.append(tracks[cid]["last_angle"])

                if len(keypoints_list) > 0:
                    try:
                        annotated_frame = annotate_frame_with_keypoints(
                            frame,
                            keypoints_list,
                            ids_list,
                            angles_list,
                            frame_count,
                            id_to_color,
                        )
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        gif_writer.append_data(annotated_frame_rgb)
                        print(f"[DEBUG] フレーム {frame_count} をGIFに追加しました。")
                    except Exception as e:
                        print(f"[ERROR] フレーム {frame_count} の注釈中にエラー: {e}")

                frame_count += 1

            print(f"[INFO] 全フレーム処理完了: GIF={output_gif_path}, CSV={output_csv_path}")

    except Exception as e:
        print(f"[ERROR] CSV書き込み時のエラー: {e}")
    finally:
        cap.release()
        gif_writer.close()


#=============================================================
# 角度計算に使う補助関数
#=============================================================
def calculate_angle_at_joint(p1, p2, p3):
    vector1 = p1 - p2
    vector2 = p3 - p2
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if magnitude_product == 0:
        return 0.0
    cos_angle = dot_product / magnitude_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees
