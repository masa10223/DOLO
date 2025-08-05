import os
import glob
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import euclidean
from natsort import natsorted
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob, os, re
from tqdm import tqdm
from pykalman import  KalmanFilter
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import IsolationForest
import networkx as nx
import math
import matplotlib.ticker as ptick 
from scipy.spatial import distance_matrix
from scipy.stats import mannwhitneyu


import matplotlib
# フォントの設定を Arial に変更
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'

def process_contact_events(
    data,
    head_radius=5.0,  # 接触と判定する半径（ピクセル）
    max_gap=1,  # 接触イベント間の許容ギャップフレーム数
    max_displacement=20  # IDスワップ検出の移動距離しきい値
    ):
    """
    データを処理し、個体間の接触イベントを検出します。

    パラメータ:
    data (pd.DataFrame): トラッキングデータ。
    head_radius (float): 接触と判定する頭部座標間の距離のしきい値。
    max_gap (int): 接触イベント間の許容ギャップフレーム数。
    max_displacement (float): IDスワップ検出の移動距離しきい値。

    戻り値:
    pd.DataFrame: 個体ペアごとの接触イベントを含むデータフレーム。
    """
    # ステップ1: データの前処理
    data = data.copy()
    data = data.sort_values(by=['Frame', 'ID']).reset_index(drop=True)

    # 角度の補正
    data = correct_angles_per_id(data)

    # 位置の補完（カルマンフィルタとスプライン補間）
    data = smooth_trajectory(data)

    # IDスワップの検出と修正
    data = detect_id_swaps(data, max_displacement=max_displacement)

    # ステップ2: 接触イベントの検出
    contact_events = detect_contact_events_with_direction(data, head_radius, max_gap)

    return contact_events

def create_kalman_filter(x, y):
    kf = KalmanFilter(initial_state_mean=[x, y, 0, 0],
                      transition_matrices=[[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],
                      observation_matrices=[[1, 0, 0, 0],
                                            [0, 1, 0, 0]],
                      transition_covariance=0.01 * np.eye(4),
                      observation_covariance=10.0 * np.eye(2),
                      initial_state_covariance=100.0 * np.eye(4))
    return kf

def smooth_trajectory(data):
    data = data.copy()
    ids = data['ID'].unique()
    smoothed_data = []

    for id_value in tqdm(ids, desc="Smoothing Trajectories"):
        id_data = data[data['ID'] == id_value].sort_values(by='Frame')
        frames = id_data['Frame'].values
        observations = id_data[['Middle_X', 'Middle_Y']].values

        # 欠損値を線形補間で埋める
        df_interp = id_data[['Middle_X', 'Middle_Y']].interpolate(method='linear', limit_direction='both')
        observations = df_interp.values

        # 観測データの形状を確認
        if observations.ndim != 2 or observations.shape[1] != 2:
            print(f"ID {id_value} の観測データの形状が不正です。スキップします。")
            continue

        kf = create_kalman_filter(observations[0, 0], observations[0, 1])
        state_means, _ = kf.smooth(observations)

        # データポイント数を確認
        m = len(frames)
        # スプラインの次数を設定（デフォルトは3）
        k = 3
        if m <= k:
            # データポイント数が少ない場合は次数を調整
            k = m - 1 if m > 1 else 0  # k は0以上
            if k >= 1:
                # スプライン補間を次数 k で実行
                smoothed_x = UnivariateSpline(frames, state_means[:, 0], k=k, s=0)(frames)
                smoothed_y = UnivariateSpline(frames, state_means[:, 1], k=k, s=0)(frames)
            else:
                # データポイントが1つまたは0の場合、そのまま使用
                smoothed_x = state_means[:, 0]
                smoothed_y = state_means[:, 1]
        else:
            # データポイント数が十分な場合は通常通りスプライン補間
            smoothed_x = UnivariateSpline(frames, state_means[:, 0], s=2)(frames)
            smoothed_y = UnivariateSpline(frames, state_means[:, 1], s=2)(frames)

        id_data['Smoothed_X'] = smoothed_x
        id_data['Smoothed_Y'] = smoothed_y
        smoothed_data.append(id_data)

    return pd.concat(smoothed_data).reset_index(drop=True)

def detect_id_swaps(data, max_displacement=20):
    data = data.copy()
    ids = data['ID'].unique()
    swap_frames = []

    for id_value in tqdm(ids, desc="Detecting ID Swaps"):
        id_data = data[data['ID'] == id_value].sort_values(by='Frame')
        positions = id_data[['Smoothed_X', 'Smoothed_Y']].values

        # 位置の差分を計算
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        # 異常な移動を検出
        abnormal_indices = np.where(displacements > max_displacement)[0] + 1  # 次のフレームが異常

        swap_frames.extend(id_data.iloc[abnormal_indices]['Frame'].tolist())

    # IDスワップの修正
    for frame in swap_frames:
        current_data = data[data['Frame'] == frame]
        previous_data = data[data['Frame'] == frame - 1]

        for id_value in ids:
            curr_row = current_data[current_data['ID'] == id_value]
            prev_row = previous_data[previous_data['ID'] == id_value]

            if curr_row.empty or prev_row.empty:
                continue

            displacement = np.linalg.norm([
                curr_row['Smoothed_X'].values[0] - prev_row['Smoothed_X'].values[0],
                curr_row['Smoothed_Y'].values[0] - prev_row['Smoothed_Y'].values[0]
            ])

            if displacement > max_displacement:
                # 他の個体と位置を比較
                for other_id in ids:
                    if other_id == id_value:
                        continue
                    other_prev_row = previous_data[previous_data['ID'] == other_id]
                    if other_prev_row.empty:
                        continue
                    other_displacement = np.linalg.norm([
                        curr_row['Smoothed_X'].values[0] - other_prev_row['Smoothed_X'].values[0],
                        curr_row['Smoothed_Y'].values[0] - other_prev_row['Smoothed_Y'].values[0]
                    ])
                    if other_displacement < displacement:
                        # IDを交換
                        data.loc[(data['Frame'] == frame) & (data['ID'] == id_value), 'ID'] = other_id
                        data.loc[(data['Frame'] == frame) & (data['ID'] == other_id), 'ID'] = id_value
                        break

    data = data.sort_values(by=['Frame', 'ID']).reset_index(drop=True)
    return data

def correct_angles_per_id(data, threshold=45, accept_threshold=35):
    """
    Corrects sudden large changes in angle by swapping Head and Tail positions when necessary,
    applied separately for each ID in the DataFrame. Invalid frames are dropped.
    """
    def correct_angles(group):
        # Initialize lists to store corrected data
        valid_indices = []
        corrected_angles = []
        Tail_x = []
        Tail_y = []
        Head_x = []
        Head_y = []

        # Extract coordinates
        group_Tail_x = group['Tail_X'].values.copy()
        group_Tail_y = group['Tail_Y'].values.copy()
        group_Mid_x = group['Middle_X'].values
        group_Mid_y = group['Middle_Y'].values
        group_Head_x = group['Head_X'].values.copy()
        group_Head_y = group['Head_Y'].values.copy()

        # Process each frame in the group
        for i in range(len(group)):
            # Original coordinates
            T = np.array([group_Tail_x[i], group_Tail_y[i]])
            m = np.array([group_Mid_x[i], group_Mid_y[i]])
            H = np.array([group_Head_x[i], group_Head_y[i]])

            # Calculate original angle
            angle = calculate_angle_between_vectors(T, m, H)

            if i > 0 and len(corrected_angles) > 0:
                prev_angle = corrected_angles[-1]
                delta = abs(angle - prev_angle)

                if delta > threshold:
                    # Suspect a flip, attempt to swap Head and Tail
                    # Swap Head and Tail
                    T_swapped = np.array([group_Head_x[i], group_Head_y[i]])
                    H_swapped = np.array([group_Tail_x[i], group_Tail_y[i]])

                    # Recalculate angle with swapped positions
                    angle_swapped = calculate_angle_between_vectors(T_swapped, m, H_swapped)
                    delta_swapped = abs(angle_swapped - prev_angle)

                    # Check if swapped angle change is acceptable
                    if delta_swapped < accept_threshold:
                        # Accept swapped positions
                        Tail_x.append(T_swapped[0])
                        Tail_y.append(T_swapped[1])
                        Head_x.append(H_swapped[0])
                        Head_y.append(H_swapped[1])
                        corrected_angles.append(angle_swapped)
                        valid_indices.append(group.index[i])
                    else:
                        # Neither original nor swapped angle acceptable
                        # Drop this frame
                        continue
                else:
                    # Angle change is acceptable, keep original angle
                    Tail_x.append(T[0])
                    Tail_y.append(T[1])
                    Head_x.append(H[0])
                    Head_y.append(H[1])
                    corrected_angles.append(angle)
                    valid_indices.append(group.index[i])
            else:
                # First frame or no previous valid angle, accept the angle
                Tail_x.append(T[0])
                Tail_y.append(T[1])
                Head_x.append(H[0])
                Head_y.append(H[1])
                corrected_angles.append(angle)
                valid_indices.append(group.index[i])

        # Create a new DataFrame with only valid rows
        corrected_group = group.loc[valid_indices].copy()
        corrected_group['Tail_X'] = Tail_x
        corrected_group['Tail_Y'] = Tail_y
        corrected_group['Head_X'] = Head_x
        corrected_group['Head_Y'] = Head_y
        corrected_group['Corrected_Angle'] = corrected_angles

        return corrected_group

    # Apply the correction function to each group identified by 'ID'
    corrected_data = data.groupby('ID', group_keys=False).apply(correct_angles)

    return corrected_data

def calculate_angle_between_vectors(T, m, H):
    """
    Calculates the angle θ between the vectors Tm_g and m_gH using vector math.
    """
    # Calculate vectors Tm_g and m_gH
    T_mg = m - T  # Vector from Tail to Mid: Tm_g = m - T
    mg_H = H - m  # Vector from Mid to Head: m_gH = H - m

    # Calculate the dot product of the vectors
    dot_product = np.dot(T_mg, mg_H)

    # Calculate magnitudes (norms) of the vectors
    magnitude_T_mg = np.linalg.norm(T_mg)
    magnitude_mg_H = np.linalg.norm(mg_H)

    # Check for zero magnitude to avoid division by zero
    if magnitude_T_mg == 0 or magnitude_mg_H == 0:
        return np.nan

    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (magnitude_T_mg * magnitude_mg_H)

    # Clip cos_theta to avoid potential numerical issues outside [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    theta = np.arccos(cos_theta)  # Radians
    theta_degrees = np.degrees(theta)  # Convert to degrees

    return theta_degrees


def detect_contact_events_with_direction(data, head_radius, max_gap):
    """
    接触イベントを検出し、接触の方向性を考慮します。

    パラメータ:
    data (pd.DataFrame): トラッキングデータ。
    head_radius (float): 接触と判定する頭部座標と他個体の体の距離のしきい値。
    max_gap (int): 接触イベント間の許容ギャップフレーム数。

    戻り値:
    pd.DataFrame: 接触イベントのデータフレーム。
    """
    data = data.copy()
    frames = data['Frame'].unique()
    ids = data['ID'].unique()

    # 個体ペアごとの接触フレームを格納する辞書
    contact_dict = {}

    # フレームごとに処理
    for frame in tqdm(frames, desc='Detecting Contacts with Direction'):
        frame_data = data[data['Frame'] == frame]
        for i in range(len(frame_data)):
            id1 = frame_data.iloc[i]['ID']
            head_x1 = frame_data.iloc[i]['Head_X']
            head_y1 = frame_data.iloc[i]['Head_Y']

            # id1 の頭部座標
            head_pos1 = np.array([head_x1, head_y1])

            for j in range(len(frame_data)):
                if i == j:
                    continue  # 同じ個体はスキップ
                id2 = frame_data.iloc[j]['ID']
                # id2 の体の各部位の座標
                tail_pos2 = np.array([frame_data.iloc[j]['Tail_X'], frame_data.iloc[j]['Tail_Y']])
                middle_pos2 = np.array([frame_data.iloc[j]['Middle_X'], frame_data.iloc[j]['Middle_Y']])
                head_pos2 = np.array([frame_data.iloc[j]['Head_X'], frame_data.iloc[j]['Head_Y']])

                # id1 の頭部と id2 の各部位との距離を計算
                distances = [
                    np.linalg.norm(head_pos1 - tail_pos2),
                    np.linalg.norm(head_pos1 - middle_pos2),
                    np.linalg.norm(head_pos1 - head_pos2)
                ]

                min_distance = min(distances)

                # 接触判定
                if min_distance <= head_radius:
                    key = (id1, id2)
                    if key in contact_dict:
                        contact_dict[key].append(frame)
                    else:
                        contact_dict[key] = [frame]

    # 接触イベントを整理
    contact_events = []
    for pair, frames_list in contact_dict.items():
        grouped_frames = group_consecutive_frames(frames_list, max_gap)
        for group in grouped_frames:
            start_frame = group[0]
            end_frame = group[-1]
            event = {
                'id1': pair[0],
                'id2': pair[1],
                'start_frame': start_frame,
                'end_frame': end_frame
            }
            contact_events.append(event)

    contact_events_df = pd.DataFrame(contact_events, columns=['id1', 'id2', 'start_frame', 'end_frame'])
    return contact_events_df

def group_consecutive_frames(frames_list, max_gap):
    """
    連続したフレームをグループ化します。

    パラメータ:
    frames_list (list): フレーム番号のリスト。
    max_gap (int): 許容する最大ギャップフレーム数。

    戻り値:
    list: グループ化されたフレームのリスト。
    """
    grouped_frames = []
    frames_list = sorted(frames_list)
    group = [frames_list[0]]
    for i in range(1, len(frames_list)):
        if frames_list[i] - frames_list[i - 1] <= max_gap + 1:
            group.append(frames_list[i])
        else:
            grouped_frames.append(group)
            group = [frames_list[i]]
    grouped_frames.append(group)
    return grouped_frames

def summarize_interactions(contact_events_df):
    """
    contact_events_df を使用して、各個体ごとのインタラクション割合を計算します。

    パラメータ:
    contact_events_df (pd.DataFrame): 'id1', 'id2', 'start_frame', 'end_frame' を含むデータフレーム。

    戻り値:
    interaction_summary (pd.DataFrame): 個体ごとのインタラクション概要を含むデータフレーム。
    """
    # 接触イベントの持続時間を計算
    contact_events_df = contact_events_df.copy()
    contact_events_df['duration'] = contact_events_df['end_frame'] - contact_events_df['start_frame'] + 1

    # 全ての個体IDを取得
    ids = pd.unique(contact_events_df[['id1', 'id2']].values.ravel('K'))

    # 個体間の接触時間を格納するデータフレームを作成
    interaction_matrix = pd.DataFrame(0, index=natsorted(ids), columns=natsorted(ids))

    # 個体ペアごとに接触時間を集計
    for _, row in contact_events_df.iterrows():
        id1 = row['id1']
        id2 = row['id2']
        duration = row['duration']
        interaction_matrix.loc[id1, id2] += duration
        #interaction_matrix.loc[id2, id1] += duration  # 対称性を持たせる


    # 個体ごとの総接触時間を計算
    total_interactions = interaction_matrix.sum(axis=1)

    # インタラクション割合を計算
    interaction_ratio = interaction_matrix.div(total_interactions, axis=0).fillna(0)

    return interaction_matrix, interaction_ratio

def plot_interaction_heatmap(interaction_matrix):
    """
    個体間の接触時間をヒートマップで可視化します。

    パラメータ:
    interaction_matrix (pd.DataFrame): 個体間の接触時間を含むデータフレーム。
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Interaction Time Heatmap")
    plt.xlabel("ID")
    plt.ylabel("ID")
    plt.show()

def plot_interaction_network(interaction_matrix, threshold=0):
    """
    個体間のインタラクションをネットワークグラフで可視化します。

    パラメータ:
    interaction_matrix (pd.DataFrame): 個体間の接触時間を含むデータフレーム。
    threshold (float): エッジを表示するための接触時間のしきい値。
    """
    G = nx.Graph()

    # ノードを追加
    for node in interaction_matrix.index:
        G.add_node(node)

    # エッジを追加（しきい値以上の接触時間のみ）
    for i in interaction_matrix.index:
        for j in interaction_matrix.columns:
            if i < j:
                weight = interaction_matrix.loc[i, j]
                if weight > threshold:
                    G.add_edge(i, j, weight=weight)

    # ノードの位置を決定
    pos = nx.circular_layout(G)

    # エッジの太さを接触時間に比例させる
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    widths = [weight / max_weight * 5 for weight in weights]  # スケーリング

    # ノードのサイズを接触時間の合計に比例させる
    node_weights = interaction_matrix.sum(axis=1)  # 各ノードの接触時間の合計
    max_node_weight = max(node_weights) if not node_weights.empty else 1  # 最大値で正規化
    # ノードサイズを計算
    node_sizes = [node_weights[node] / max_node_weight * 5000 for node in G.nodes()]  # スケーリングして表示



    # グラフを描画
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes ,  node_color='skyblue')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray')
    plt.title("Interaction Network Graph")
    plt.axis('off')
    plt.show()

def plot_contact_count_heatmap(contact_events_df):
    """
    接触回数を上三角行列のヒートマップとして表示します（方向性を考慮）。

    パラメータ:
    contact_events_df (pd.DataFrame): 'id_from', 'id_to', 'start_frame', 'end_frame' を含むデータフレーム。
    """
    # 接触回数を計算
    contact_counts = contact_events_df.groupby(['id1', 'id2']).size().reset_index(name='Contact_Count')

    # 個体IDのリストを取得し、ソート
    ids = sorted(set(contact_counts['id1']).union(set(contact_counts['id2'])))

    # 接触回数の行列を作成
    interaction_matrix = pd.DataFrame(0, index=ids, columns=ids)

    # 接触回数を行列に埋め込む
    for _, row in contact_counts.iterrows():
        id_from = row['id1']
        id_to = row['id2']
        count = row['Contact_Count']
        interaction_matrix.loc[id_from, id_to] = count


    # ヒートマップの描画
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, annot=True, fmt=".0f", cmap="YlGnBu", mask=interaction_matrix.isnull())
    plt.title("Directional Contact Counts Heatmap (Upper Triangular)")
    plt.xlabel("ID To")
    plt.ylabel("ID From")
    plt.show()

def select_frames(thr_frame = 5000): 
    control_matrix_files = []
    for path in natsorted(glob.glob('./csvs/trajectory/C*')):
        print(f"Selecting frames from {path}")
        tmp_df = pd.read_csv(path)
        total_frames = tmp_df['Frame'].max() + 1
        if total_frames > thr_frame:
            print(path)
            control_matrix_files.append(path)

    mutant_matrix_files = []
    for path in natsorted(glob.glob('./csvs/trajectory/D*')):
        print(f"Selecting frames from {path}")
        tmp_df = pd.read_csv(path)
        total_frames = tmp_df['Frame'].max() + 1
        if total_frames > thr_frame:
            print(path)
            mutant_matrix_files.append(path)
            
    return control_matrix_files, mutant_matrix_files

def selected_frames(thr_frame = 5000): 
    control_matrix_files = []
    for path in natsorted(glob.glob('./csvs/selected_trajectory/C*')):
        print(f"Selecting frames from {path}")
        tmp_df = pd.read_csv(path)
        total_frames = tmp_df['Frame'].max() + 1
        if total_frames > thr_frame:
            print(path)
            control_matrix_files.append(path)

    mutant_matrix_files = []
    for path in natsorted(glob.glob('./csvs/selected_trajectory/D*')):
        print(f"Selecting frames from {path}")
        tmp_df = pd.read_csv(path)
        total_frames = tmp_df['Frame'].max() + 1
        if total_frames > thr_frame:
            print(path)
            mutant_matrix_files.append(path)
            
    return control_matrix_files, mutant_matrix_files

def summarize_initeractions_controls(control_matrix_files, head_radius=15, max_gap=5, max_displacement=20):
    for control_path in tqdm(
        control_matrix_files, desc="Collecting Controls..."
    ):
        traj_csv = pd.read_csv(control_path)
        filename = os.path.splitext(os.path.basename(control_path))[0]
        contact_events_df = process_contact_events(
            traj_csv,
            head_radius=head_radius,  # 接触と判定する半径（ピクセル）
            max_gap=max_gap,  # 許容する最大ギャップフレーム数
            max_displacement=max_displacement,  # IDスワップ検出の移動距離しきい値
        )

        # インタラクション割合を計算
        interaction_matrix, interaction_ratio = summarize_interactions(
            contact_events_df
        )

        interaction_matrix.to_csv(
            "./csvs/interaction_matrcies/inter_mat_{}.csv".format(filename)
        )
        contact_events_df.to_csv(
            "./csvs/interaction_matrcies/contact_events_{}.csv".format(filename)
        )
        print('Saved interaction matricies of controls!')

def summarize_initeractions_mutants(mutant_matrix_files, head_radius=15, max_gap=5, max_displacement=20):
    for control_path in tqdm(
        mutant_matrix_files, desc="Collecting Mutants..."
    ):
        traj_csv = pd.read_csv(control_path)
        filename = os.path.splitext(os.path.basename(control_path))[0]
        contact_events_df = process_contact_events(
            traj_csv,
            head_radius=head_radius,  # 接触と判定する半径（ピクセル）
            max_gap=max_gap,  # 許容する最大ギャップフレーム数
            max_displacement=max_displacement,  # IDスワップ検出の移動距離しきい値
        )

        # インタラクション割合を計算
        interaction_matrix, interaction_ratio = summarize_interactions(
            contact_events_df
        )

        interaction_matrix.to_csv(
            "./csvs/interaction_matrcies/inter_mat_{}.csv".format(filename)
        )
        contact_events_df.to_csv(
            "./csvs/interaction_matrcies/contact_events_{}.csv".format(filename)
        )
        
        print('Saved interaction matricies of mutants!')


def summarize_initeractions_controls_and_mutants(control_matrix_files, mutant_matrix_files, head_radius=15, max_gap=5, max_displacement=20):

    summarize_initeractions_controls(
        control_matrix_files=control_matrix_files,
        head_radius=head_radius,  # 接触と判定する半径（ピクセル）
        max_gap=max_gap,  # 許容する最大ギャップフレーム数
        max_displacement=max_displacement,
    )
    summarize_initeractions_mutants(
        mutant_matrix_files=mutant_matrix_files,
        head_radius=head_radius,  # 接触と判定する半径（ピクセル）
        max_gap=max_gap,  # 許容する最大ギャップフレーム数
        max_displacement=max_displacement,
    )


def calculate_distances(df, PIXEL_TO_CM=0.006):
    distances = {track_id: 0 for track_id in df["ID"].unique()}
    total_frame_counts = {track_id: 0 for track_id in df["ID"].unique()}
    for track_id in df["ID"].unique():
        track_data = df[df["ID"] == track_id].sort_values("Frame")
        previous_position = None
        previous_frame = None
        frames = track_data["Frame"].values
        if len(frames) == 0:
            continue
        first_frame = frames[0]
        last_frame = frames[-1]
        total_frames = last_frame - first_frame + 1
        total_frame_counts[track_id] = total_frames
        for idx, row in track_data.iterrows():
            current_position = (row["Middle_X"], row["Middle_Y"])
            current_frame = row["Frame"]
            if previous_position is not None:
                frame_diff = current_frame - previous_frame
                if frame_diff > 0:
                    # 距離をフレーム差分で割って、1フレームあたりの移動距離を計算
                    distance = (
                        euclidean(previous_position, current_position) * PIXEL_TO_CM
                    )
                    distance_per_frame = distance / frame_diff
                    # 総移動距離に加算
                    distances[track_id] += distance
            previous_position = current_position
            previous_frame = current_frame
    return distances, total_frame_counts


def distance_compute(PIXEL_TO_CM=0.006):
    # 結果を格納するリスト
    results = []
    # for path in control_paths:
    for path in natsorted(glob.glob("./csvs/interaction_matrcies/contact_events_C*")):
        filename = os.path.splitext(os.path.basename(path))[0][15:]
        df = pd.read_csv("./csvs/trajectory/{}.csv".format(filename))
        distances, frame_counts = calculate_distances(df, PIXEL_TO_CM)
        print(path)
        print("Total distances traveled by each track ID:")
        for track_id in distances.keys():
            distance = distances[track_id]
            frame_count = frame_counts[track_id]
            avg_distance = distance / frame_count if frame_count > 0 else 0
            results.append(
                [
                    os.path.basename(path),
                    track_id,
                    distance,
                    avg_distance,
                    "$white^{1118}$",
                ]
            )
            print(
                f"Track ID {track_id}: Total Distance = {distance:.2f} cm, Frames = {frame_count}, Avg Distance = {avg_distance:.4f} cm/frame"
            )
        print("")

    results_df = pd.DataFrame(
        results, columns=["path", "track_id", "distance", "avg_distance", "Group"]
    )

    # 同様に 'nagoya_d*.csv' に対しても処理
    results_d = []
    # for path in mutant_paths:
    for path in natsorted(glob.glob("./csvs/interaction_matrcies/contact_events_D*")):
        filename = os.path.splitext(os.path.basename(path))[0][15:]
        df = pd.read_csv("./csvs/trajectory/{}.csv".format(filename))
        distances, frame_counts = calculate_distances(df, PIXEL_TO_CM)
        print(path)
        print("Total distances traveled by each track ID:")
        for track_id in distances.keys():
            distance = distances[track_id]
            frame_count = frame_counts[track_id]
            avg_distance = distance / frame_count if frame_count > 0 else 0
            results_d.append(
                [
                    os.path.basename(path),
                    track_id,
                    distance,
                    avg_distance,
                    "$orco^2$ , $Gr63a^1$",
                ]
            )
            print(
                f"Track ID {track_id}: Total Distance = {distance:.2f} cm, Frames = {frame_count}, Avg Distance = {avg_distance:.4f} cm/frame"
            )
        print("")

    results_df_d = pd.DataFrame(
        results_d, columns=["path", "track_id", "distance", "avg_distance", "Group"]
    )
    results_combine = pd.concat([results_df, results_df_d])

    return results_combine, results_df, results_df_d


def plot_distance_compare(results_combine, results_df, results_df_d, filename = None):
    mutant = results_df_d["avg_distance"]
    control = results_df["avg_distance"]

    # 1. コルモゴロフ・スミルノフ検定
    ks_stat, ks_p_value = stats.ks_2samp(control, mutant)
    print("KS検定:")
    print(f"統計量 = {ks_stat}, p値 = {ks_p_value}\n")

    # 2. マン・ホイットニーU検定
    mw_stat, mw_p_value = stats.mannwhitneyu(control, mutant, alternative="two-sided")
    print("マン・ホイットニーU検定:")
    print(f"統計量 = {mw_stat}, p値 = {mw_p_value}\n")

    # 3. カイ二乗適合度検定
    # ヒストグラムのビンを計算
    # control_hist, bins = np.histogram(control, bins=10)
    # mutant_hist, _ = np.histogram(mutant, bins=bins)
    # chi2_stat, chi2_p_value = stats.chisquare(control_hist, f_exp=mutant_hist)
    # print("カイ二乗適合度検定:")
    # print(f"統計量 = {chi2_stat}, p値 = {chi2_p_value}\n")

    # 4. クラスカル・ウォリス検定
    kw_stat, kw_p_value = stats.kruskal(control, mutant)
    print("クラスカル・ウォリス検定:")
    print(f"統計量 = {kw_stat}, p値 = {kw_p_value}")

    fig, ax = plt.subplots(figsize=(9, 6))
    results_combine = pd.concat([results_df, results_df_d])
    sns.histplot(
        data=results_combine,
        x="avg_distance",
        hue="Group",
        palette="coolwarm",
        multiple="dodge",
        # kde=True,
        # fill=False
    )
    plt.xlabel("Velocity [cm/s]", fontsize=20)
    plt.yticks(np.arange(0, 24, 5))
    plt.ylabel("Counts", fontsize=20)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, -2))
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", labelsize=15)
    plt.savefig(f"./Fig_paper/{filename}/velocity_compare_{filename}.pdf")
    print(f"PDF saved at ./Fig_paper/{filename}/velocity_compare_{filename}.pdf")


    sns.displot(
        data=results_combine,
        x="avg_distance",
        hue="Group",
        palette="coolwarm",
        kind="kde",
        height=5,
        aspect=1.6,
    )
    plt.xlabel("Velocity [cm/s]", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, -2))
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", labelsize=15)
    plt.savefig(f"./Fig_paper/{filename}/contact_time_compare_density_all_{filename}.pdf")
    print(f"PDF saved at ./Fig_paper/{filename}/contact_time_compare_density_all_{filename}.pdf")
    
    

def compute_body_angle(head, middle, tail):
    """
    Head, Middle, Tail が (x, y) のタプルで与えられた場合、
    Middle における角度（Head-Middle と Tail-Middle のなす角）を返す。
    体が一直線なら約 π、曲がっていればこの値は小さくなる。
    """
    v1 = (head[0] - middle[0], head[1] - middle[1])
    v2 = (tail[0] - middle[0], tail[1] - middle[1])
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 == 0 or norm2 == 0:
        return np.nan  # 異常値対策
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos_angle = max(min(dot / (norm1 * norm2), 1.0), -1.0)
    angle = math.acos(cos_angle)
    return angle

def is_local_minimum(curvatures, idx, window=2):
    """
    curvatures: 各フレームの curvature のリスト
    idx: 判定対象のインデックス
    window: 前後何フレームと比較するか（例：window=2なら idx-2～idx+2内で比較）
    
    → idx の値が指定窓内で最も低い（局所最小）なら True を返す
    """
    current = curvatures[idx]
    for i in range(max(0, idx - window), min(len(curvatures), idx + window + 1)):
        if i == idx:
            continue
        if curvatures[i] is None or current is None or np.isnan(curvatures[i]) or np.isnan(current):
            continue
        if current > curvatures[i]:
            return False
    return True

def group_consecutive_frames(frames, max_gap=1):
    """
    ソート済みのフレーム番号リスト frames を、各フレーム間のギャップが max_gap 以下なら同一グループとし、
    連続しているフレーム群のリストを返す。
    """
    groups = []
    current_group = []
    for frame in frames:
        if not current_group:
            current_group.append(frame)
        else:
            if frame - current_group[-1] <= max_gap:
                current_group.append(frame)
            else:
                groups.append(current_group)
                current_group = [frame]
    if current_group:
        groups.append(current_group)
    return groups

# ===== 回転イベント検出＆集計関数 =====

def detect_rotation_events_with_direction(data, 
                                          min_event_length=15, 
                                          curvature_threshold=2.5, 
                                          local_window=0, 
                                          max_gap=1):
    """
    トラッキングデータ（DataFrame）から各個体の回転（体が「くの字」になる状態）イベントを検出します。
    
    パラメータ:
      data (pd.DataFrame): 以下のカラムを含む必要があります。
          'Frame', 'ID', 'Head_X', 'Head_Y', 'Middle_X', 'Middle_Y', 'Tail_X', 'Tail_Y'
      min_event_length (int): 連続して「くの字状態」が続く最低フレーム数
      curvature_threshold (float): curvature がこの値未満なら十分に曲がっていると判定
      local_window (int): 局所最小判定に用いる前後フレーム数（0なら単純に閾値判定のみ）
      max_gap (int): 連続グループとして許容するフレーム間ギャップ
    
    戻り値:
      rotation_events_df (pd.DataFrame): 各イベントについて、ID, start_frame, end_frame, duration,
                                           event_frames（リスト）などを含む DataFrame
    """
    data = data.copy()
    
    # 各行に対して curvature を計算
    def compute_curv_row(row):
        head = (row['Head_X'], row['Head_Y'])
        middle = (row['Middle_X'], row['Middle_Y'])
        tail = (row['Tail_X'], row['Tail_Y'])
        return compute_body_angle(head, middle, tail)
    
    data['Curvature'] = data.apply(compute_curv_row, axis=1)
    
    # 各個体 (ID) ごとに、フレーム順にソートし、KuShape フラグを計算
    def compute_ku_shape(group):
        group = group.sort_values('Frame').copy()
        curvatures = group['Curvature'].tolist()
        ku_shape_flags = []
        for i, curv in enumerate(curvatures):
            if curv is None or np.isnan(curv):
                ku_shape_flags.append(False)
            else:
                flag = (curv < curvature_threshold and 
                        is_local_minimum(curvatures, i, window=local_window))
                ku_shape_flags.append(flag)
        group['KuShape'] = ku_shape_flags
        return group
    
    data = data.groupby('ID', group_keys=False).apply(compute_ku_shape)
    
    # 各個体ごとに、KuShape が True となっているフレーム番号を取得し、連続グループ化する
    events = []
    for id_val, group in data.groupby('ID'):
        group = group.sort_values('Frame')
        frames_true = group.loc[group['KuShape'], 'Frame'].tolist()
        groups = group_consecutive_frames(frames_true, max_gap=max_gap)
        for g in groups:
            if len(g) >= min_event_length:
                event = {
                    'ID': id_val,
                    'start_frame': g[0],
                    'end_frame': g[-1],
                    'duration': g[-1] - g[0] + 1,
                    'event_frames': g
                }
                events.append(event)
    rotation_events_df = pd.DataFrame(events)
    return rotation_events_df

def summarize_rotation_events(rotation_events_df):
    """
    rotation_events_df を用いて、各個体ごとの回転イベント概要（イベント数、総持続時間、平均・中央値持続時間など）を算出します。
    
    戻り値:
      summary_df (pd.DataFrame): 各個体ごとのサマリー情報を含む DataFrame
    """
    if rotation_events_df.empty:
        return pd.DataFrame()
    
    summary_df = rotation_events_df.groupby('ID').agg(
        event_count=('ID', 'count'),
        total_duration=('duration', 'sum'),
        mean_duration=('duration', 'mean'),
        median_duration=('duration', 'median')
    ).reset_index()
    return summary_df

def compute_rotation_file_summary(file_list, rotation_events_df):
    """
    file_list 内の各ファイルについて、CSVから総フレーム数（Frameカラムの最大値＋1）を算出し、
    rotation_events_df におけるそのファイル内の回転イベント数と、全体フレームに対する割合を計算する。
    
    file_list: 対象ファイルパスのリスト
    rotation_events_df: 既に各ファイルごとに検出された回転イベントの DataFrame（'File'カラムあり）
    
    戻り値:
      rotation_file_summary_df (pd.DataFrame): 各ファイルごとの 'File', 'Group', 'total_frames',
                                                 'event_count', 'ratio'（イベント発生割合）の情報を持つ DataFrame
    """
    summary_list = []
    for f in file_list:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
        # 総フレーム数はFrameカラムの最大値＋1（フレーム番号が0始まりの場合）
        total_frames = df['Frame'].max() + 1
        file_name = os.path.basename(f)
        ## IDごとに集計 (2025/2/4 Update)
        for ids in range(1,6):
        # rotation_events_df に同一ファイル名のイベントを抽出
            events_for_file = rotation_events_df[(rotation_events_df['File'] == file_name)
            & (rotation_events_df['ID'] ==ids)]
            event_count = len(events_for_file)
            ratio = event_count / total_frames if total_frames > 0 else np.nan
            # グループはファイル名の先頭文字で判定（例：'C'→Control, 'D'→Mutant）
            if file_name.startswith('C'):
                group = 'Control'
            elif file_name.startswith('D'):
                group = 'Mutant'
            else:
                group = 'Unknown'
            summary_list.append({
                'File': file_name,
                'Group': group,
                'total_frames': total_frames,
                'event_count': event_count,
                'id': ids,
                'ratio': ratio
            })
    return pd.DataFrame(summary_list)

# =============================================================================
# 【プロット関数：ファイル別回転イベントの散布図と箱ひげ図】
# =============================================================================
def add_stat_annotation(ax, x1, x2, y, p_val, h):
    """
    ax: matplotlib の Axes オブジェクト
    x1, x2: 比較対象の箱（x座標；ここでは0と1と仮定）
    y: 横線の開始y座標
    p_val: 統計検定のp値
    h: オフセット（y軸方向）
    """
    color = 'black'
    # 横線の描画
    ax.plot([x1, x1, x2, x2],
            [y, y + h * 1.2, y + h * 1.2, y],
            color=color)
    # p値に応じたテキスト設定
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "n.s."
    # 横線中央上部にテキストを描画
    ax.text((x1 + x2) * 0.5, y + h, text,
            ha='center', va='bottom', color=color, fontsize=20)

def plot_rotation_events_file_summary(rotation_file_summary_df, filename = None):
    """
    rotation_file_summary_df の内容を用いて、各ファイルごとの回転イベント数と発生割合を
    散布図と箱ひげ図で可視化する。
    さらに、各箱ひげ図に有意差のアノテーションを追加する。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import mannwhitneyu

    if rotation_file_summary_df.empty:
        print("ファイル別の回転イベントサマリーが存在しません。")
        return
    
    # --- 散布図 ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rotation_file_summary_df, x='event_count', y='ratio', hue='Group', palette="coolwarm", s=100,
                    edgecolor="black", lw = 2
                    )
    plt.title("Turn Events: Count vs. Ratio (per file)", fontsize=20)
    plt.xlabel("Turn Event Count", fontsize=20)
    plt.ylabel("Event Ratio (events / total frames)", fontsize=20)
    plt.legend(title='Group')
    plt.savefig(f"./Fig_paper/{filename}/turn_event_count_vs_ratio_{filename}.pdf")
    
    # --- 箱ひげ図：グループ別のイベント数 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Group', y='event_count', data=rotation_file_summary_df, palette="coolwarm", ax=ax,
                linecolor="k",
                linewidth=2,)
    sns.stripplot(x='Group', y='event_count', data=rotation_file_summary_df, palette="coolwarm", ax=ax,
                  edgecolor="k",
                s = 20, linewidth=2)
    ax.set_title("Turn Event Count Distribution by Group", fontsize=20)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Rotation Event Count", fontsize=20)
    labels = ['$\mathit{white^{1118}}$', '$\mathit{orco^2}$ , $\mathit{Gr63a^1}$']
    ax.set_xticklabels(labels, fontsize=15)
    
    # 統計検定（Mann‐Whitney U検定）による有意差判定
    control_counts = rotation_file_summary_df[rotation_file_summary_df['Group']=='Control']['event_count']
    mutant_counts  = rotation_file_summary_df[rotation_file_summary_df['Group']=='Mutant']['event_count']
    stat, p_val = mannwhitneyu(control_counts, mutant_counts, alternative='two-sided')
    # アノテーションの位置を設定（x座標は0と1と仮定）
    y_max = rotation_file_summary_df['event_count'].max()
    y_min = rotation_file_summary_df['event_count'].min()
    h_offset = (y_max - y_min) * 0.1  # 10%の高さをオフセットとして設定
    add_stat_annotation(ax, 0, 1, y_max + h_offset, p_val, h_offset)
    plt.savefig(f"./Fig_paper/{filename}/turn_event_count_distribution_{filename}.pdf")

    
    # --- 箱ひげ図：グループ別のイベント割合 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2,-2))
    sns.boxplot(x='Group', y='ratio', data=rotation_file_summary_df, palette="coolwarm", ax=ax,
                linecolor="k",
                linewidth=2,)
    sns.stripplot(x='Group', y='ratio', data=rotation_file_summary_df, palette="coolwarm", ax=ax,
                  edgecolor="k",
                s = 20, linewidth=2)
    ax.set_title("Turn Event Ratio Distribution by Group", fontsize=20)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Event Ratio (events / total frames)", fontsize=20)
    labels = ['$\mathit{white^{1118}}$', '$\mathit{orco^2}$ , $\mathit{Gr63a^1}$']
    ax.set_xticklabels(labels, fontsize=15)
    # 統計検定
    control_ratio = rotation_file_summary_df[rotation_file_summary_df['Group']=='Control']['ratio']
    mutant_ratio  = rotation_file_summary_df[rotation_file_summary_df['Group']=='Mutant']['ratio']
    stat, p_val = mannwhitneyu(control_ratio, mutant_ratio, alternative='two-sided')
    y_max = rotation_file_summary_df['ratio'].max()
    y_min = rotation_file_summary_df['ratio'].min()
    h_offset = (y_max - y_min) * 0.1
    add_stat_annotation(ax, 0, 1, y_max + h_offset, p_val, h_offset)
    plt.savefig(f"./Fig_paper/{filename}/turn_event_ratio_distribution_{filename}.pdf")


def plot_rotation_events_by_files(control_matrix_files, mutant_matrix_files,
                                  min_event_length=15, curvature_threshold=2.5,
                                  local_window=0, max_gap=1, max_displacement=20,
                                  filename = None):
    """
    control_matrix_files と mutant_matrix_files の全ファイルについて、
    回転（くの字）イベントの検出・集計を行い、グループ別（Control / Mutant）の結果を
    ボックスプロットおよび散布図で表示する関数です。

    パラメータ:
      control_matrix_files (list): Control群のCSVファイルパスのリスト
      mutant_matrix_files (list): Mutant群のCSVファイルパスのリスト
      min_event_length (int): 連続して「くの字状態」が続く最低フレーム数
      curvature_threshold (float): 曲がり（curvature）の閾値（この値未満なら十分に曲がっていると判定）
      local_window (int): 局所最小判定に用いる前後フレーム数（0なら単純に閾値判定のみ）
      max_gap (int): 連続グループとして許容するフレーム間ギャップ
      max_displacement (int): IDスワップ検出用の移動距離しきい値
    """

    # --- Control群の全ファイルを処理 ---
    control_rotation_events_list = []
    for path in tqdm(control_matrix_files, desc="Processing Control Rotation Events"):
        data = pd.read_csv(path)
        # 前処理：角度補正、スムージング、IDスワップ修正
        data = correct_angles_per_id(data)
        data = smooth_trajectory(data)
        data = detect_id_swaps(data, max_displacement=max_displacement)
        # 回転イベント検出
        rotation_events_df = detect_rotation_events_with_direction(
            data,
            min_event_length=min_event_length,
            curvature_threshold=curvature_threshold,
            local_window=local_window,
            max_gap=max_gap
        )
        if not rotation_events_df.empty:
            rotation_events_df['Group'] = 'Control'
            rotation_events_df['File'] = os.path.basename(path)
            control_rotation_events_list.append(rotation_events_df)

    # --- Mutant群の全ファイルを処理 ---
    mutant_rotation_events_list = []
    for path in tqdm(mutant_matrix_files, desc="Processing Mutant Rotation Events"):
        data = pd.read_csv(path)
        # 前処理
        data = correct_angles_per_id(data)
        data = smooth_trajectory(data)
        data = detect_id_swaps(data, max_displacement=max_displacement)
        # 回転イベント検出
        rotation_events_df = detect_rotation_events_with_direction(
            data,
            min_event_length=min_event_length,
            curvature_threshold=curvature_threshold,
            local_window=local_window,
            max_gap=max_gap
        )
        if not rotation_events_df.empty:
            rotation_events_df['Group'] = 'Mutant'
            rotation_events_df['File'] = os.path.basename(path)
            mutant_rotation_events_list.append(rotation_events_df)

    # --- 全ファイルの結果を結合 ---
    if control_rotation_events_list or mutant_rotation_events_list:
        all_rotation_events_df = pd.concat(control_rotation_events_list + mutant_rotation_events_list, ignore_index=True)
    else:
        all_rotation_events_df = pd.DataFrame()

    all_files = control_matrix_files + mutant_matrix_files
    rotation_file_summary_df = compute_rotation_file_summary(all_files, all_rotation_events_df)

    plot_rotation_events_file_summary(rotation_file_summary_df, filename=filename)
    
    return all_rotation_events_df, rotation_file_summary_df
    
    
def compute_proximity_ratio(df, threshold, id_col="ID", frame_col="Frame", x_col="Head_X", y_col="Head_Y"):
    """
    df: 軌跡データの DataFrame
    threshold: 距離のしきい値（例: 50）
    Returns: 各個体ごとに、近接状態（最近接距離 < threshold）となるフレームの割合
    """
    # 各フレームごとに処理して、最近接距離を計算
    results = []
    for frame, group in df.groupby(frame_col):
        group = group.copy()
        pts = group[[x_col, y_col]].values
        if len(pts) < 2:
            group['nnd'] = np.nan
        else:
            dmat = distance_matrix(pts, pts)
            np.fill_diagonal(dmat, np.inf)
            group['nnd'] = dmat.min(axis=1)
        results.append(group)
    df2 = pd.concat(results, ignore_index=True)
    summary = df2.groupby(id_col).apply(lambda g: pd.Series({
        'proximity_ratio': np.mean(g['nnd'] < threshold)
    })).reset_index()
    return summary

def plot_proximity_ratio_by_files(control_matrix_files, mutant_matrix_files,
                                  distance_threshold=50, filename = None):
    summary_df_control = pd.DataFrame()
    for path in control_matrix_files:
        df = pd.read_csv(path)
        proximity_summary = compute_proximity_ratio(df, threshold=distance_threshold, id_col="ID", frame_col="Frame", x_col="Head_X", y_col="Head_Y")
        summary_df_control = pd.concat([summary_df_control, proximity_summary])

    summary_df_mutant = pd.DataFrame()
    for path in mutant_matrix_files:
        df = pd.read_csv(path)
        proximity_summary = compute_proximity_ratio(df, threshold=distance_threshold, id_col="ID", frame_col="Frame", x_col="Head_X", y_col="Head_Y")
        summary_df_mutant = pd.concat([summary_df_mutant, proximity_summary])
        
    summary_df_control['Group'] = 'control'
    summary_df_mutant['Group'] = 'mutant'
    control_values = summary_df_control['proximity_ratio'].dropna().values
    mutant_values  = summary_df_mutant['proximity_ratio'].dropna().values
    summary_df_concat = pd.concat([summary_df_control, summary_df_mutant])
    stat, p_val = mannwhitneyu(control_values, mutant_values, alternative='two-sided')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Group', y='proximity_ratio', data=summary_df_concat, palette="coolwarm", ax=ax,
            linecolor="k",
            linewidth=2,)
    sns.stripplot(x='Group', y='proximity_ratio', data=summary_df_concat, palette="coolwarm", ax=ax,
                  edgecolor="k",
                s = 20, linewidth=2)
    ax.set_title(f"Distribution of Proximity Ratio of each individual\n Threshold = {distance_threshold}", fontsize=20)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Proximity ratio", fontsize=20)
    labels = ['$\mathit{white^{1118}}$', '$\mathit{orco^2}$ , $\mathit{Gr63a^1}$']
    ax.set_xticklabels(labels, fontsize=15)

    y_max = summary_df_concat['proximity_ratio'].max()
    y_min = summary_df_concat['proximity_ratio'].min()
    h_offset = (y_max - y_min) * 0.1
    add_stat_annotation(ax, 0, 1, y_max + h_offset, p_val, h_offset)
    plt.savefig(f"./Fig_paper/{filename}/proximity_ratio_distribution_{filename}.pdf")

# =============================================================================
# 【プロット関数：Contact + Turn】
# =============================================================================

def select_frames_of_contact_events(thr_frame = 5000): 
    control_matrix_files = []
    for path in natsorted(glob.glob('./csvs/interaction_matrcies/contact_events_C*')):
        df = pd.read_csv(path)
        ## 最大フレームの取得
        filename = (os.path.splitext(os.path.basename(path))[0][15:])
        tmp_df = pd.read_csv('./csvs/trajectory/{}.csv'.format(filename))
        total_frames = tmp_df['Frame'].max() + 1
        if total_frames > thr_frame:
            print(path)
            control_matrix_files.append(path)

    mutant_matrix_files = []
    for path in natsorted(glob.glob('./csvs/interaction_matrcies/contact_events_D*')):
        df = pd.read_csv(path)
        ## 最大フレームの取得
        filename = (os.path.splitext(os.path.basename(path))[0][15:])
        tmp_df = pd.read_csv('./csvs/trajectory/{}.csv'.format(filename))
        total_frames = tmp_df['Frame'].max() + 1
        if total_frames > thr_frame:
            print(path)
            mutant_matrix_files.append(path)
            
    return control_matrix_files, mutant_matrix_files

def collect_contact_files(thr_frame = 5000):
    contact_control_matrix_files, contact_mutant_matrix_files = select_frames_of_contact_events(thr_frame = thr_frame)
    all_contact_events_df = pd.DataFrame()
    for path in contact_control_matrix_files:
        df = pd.read_csv(path)
        df['File'] = os.path.basename(path)
        df['Group'] = 'Control'
        all_contact_events_df = pd.concat([all_contact_events_df,df])

    for path in contact_mutant_matrix_files:
        df = pd.read_csv(path)
        df['File'] = os.path.basename(path)
        df['Group'] = 'Mutant'
        all_contact_events_df = pd.concat([all_contact_events_df,df])
        
    return all_contact_events_df.reset_index(drop=True)


def check_turn_after_contact(contact_df, turn_df, threshold=30, target="id1"):
    """
    接触イベントの各行について、指定された個体（target列の値）が、
    同じファイル内で、接触終了後 threshold フレーム以内にTurnイベントを起こすかを判定する。
    
    Parameters:
      contact_df: 接触イベントのDataFrame。必須カラム: 'id1', 'id2', 'start_frame', 'end_frame', 'File'
      turn_df: TurnイベントのDataFrame。必須カラム: 'ID', 'start_frame', 'end_frame', 'duration', 'event_frames', 'Group', 'File'
      threshold: 接触終了後、何フレーム以内にTurnイベントが開始すれば「追跡」とみなすかの閾値（例: 30）
      target: 接触イベント中、どちらの個体を対象にTurnの有無をチェックするか（例: "id1" または "id2"）
    
    Returns:
      contact_df: 各接触イベントに対して、新たに以下のカラムを追加したDataFrame  
          - "turn_within_threshold": True/False  
          - "turn_start_frame": 該当するTurnイベントの開始フレーム（該当しない場合はNaN）
    """
    # DataFrameのコピーを作成
    contact_df = contact_df.copy()
    
    # 新たなカラムを初期化
    contact_df["turn_within_threshold"] = False
    contact_df["turn_start_frame"] = pd.NA
    
    # 各接触イベント行ごとにチェック
    for idx, row in contact_df.iterrows():
        file_name = row["File"][15:]
        contact_end = row["end_frame"]
        target_id = row[target]
        
        # turn_dfから、同じファイルかつ対象個体のTurnイベントを抽出
        df_turn = turn_df[(turn_df["File"] == file_name) & (turn_df["ID"] == target_id)]
        # 接触終了後、かつ threshold フレーム以内に開始するTurnイベントを抽出
        df_turn_within = df_turn[(df_turn["start_frame"] > contact_end) & 
                                  (df_turn["start_frame"] <= contact_end + threshold)]
        if not df_turn_within.empty:
            # 最初のTurnイベントのstart_frameを取得（昇順ソートして）
            first_turn_start = df_turn_within["start_frame"].min()
            contact_df.at[idx, "turn_within_threshold"] = True
            contact_df.at[idx, "turn_start_frame"] = first_turn_start
        else:
            contact_df.at[idx, "turn_within_threshold"] = False
            contact_df.at[idx, "turn_start_frame"] = pd.NA
    return contact_df

def aggregate_turn_events(df_sub, target):
    """
    make_sub_event_rows() で作成した DataFrame (File, Group, ID, turn_within_threshold) から、
    個体ごとに合計接触イベント数(total_events) と Turn イベント数(turn_events)をカウントし、
    turn_ratio を算出する。
    """
    agg = df_sub.groupby(['File','Group',target]).agg(
        total_events = (target, 'count'),
        turn_events = ('turn_within_threshold', lambda x: x.sum())
    ).reset_index()
    agg['turn_ratio'] = agg['turn_events'] / agg['total_events']
    return agg


def plot_turn_after_contact(control_matrix_files, mutant_matrix_files, threshold=30, target="id1", thr_frame = 5000, filename = None):
    contact_df = collect_contact_files(thr_frame = thr_frame)
    turn_df, _ = plot_rotation_events_by_files(control_matrix_files, mutant_matrix_files, filename=filename)
    
    contact_df = check_turn_after_contact(contact_df, turn_df, threshold=threshold, target=target)
    final_agg = aggregate_turn_events(contact_df, target)
    
    control_values = final_agg[final_agg['Group']=='Control']['turn_ratio'].dropna().values
    mutant_values  = final_agg[final_agg['Group']=='Mutant']['turn_ratio'].dropna().values
    
    stat, p_val = mannwhitneyu(control_values, mutant_values, alternative='two-sided')
    print("Mann–Whitney U test p-value:", p_val)
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x='Group', y='turn_ratio', data=final_agg, palette='coolwarm', ax=ax,
                linecolor="k",
                linewidth=2,)
    sns.stripplot(x='Group', y='turn_ratio', data=final_agg, palette='coolwarm', ax=ax,
                  edgecolor='k', s=20, linewidth=2)
    ax.set_title(f"Turn After Contact Ratio, threshold={threshold}", fontsize=16)
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("Turn Ratio", fontsize=14)
    # example of custom x-ticks
    labels = ['$\mathit{white^{1118}}$', '$\mathit{orco^2}$ , $\mathit{Gr63a^1}$']
    ax.set_xticklabels(labels, fontsize=14)
    
    y_max = final_agg['turn_ratio'].max()
    y_min = final_agg['turn_ratio'].min()
    h_offset = (y_max - y_min)*0.1
    add_stat_annotation(ax, 0, 1, y_max + h_offset, p_val, h_offset)
    plt.savefig(f'./Fig_paper/{filename}/turn_after_contact_ratio_{filename}.pdf')

                                    


if __name__ == "__main__":
    PIXEL_TO_CM = 0.006  # ピクセルからセンチメートルへの変換定数
    Head_radius = 25
    Thr_frame = 5500
    import matplotlib
    # フォントの設定を Arial に変更
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
    control_matrix_files, mutant_matrix_files = select_frames(thr_frame = Thr_frame)
    summarize_initeractions_controls_and_mutants(control_matrix_files, mutant_matrix_files,
                                                 head_radius=Head_radius, max_gap=5, max_displacement=20)
    
    results_combine, results_df, results_df_d = distance_compute(
        PIXEL_TO_CM=PIXEL_TO_CM
    )
    plot_distance_compare(results_combine, results_df, results_df_d)
    
    _,_ = plot_rotation_events_by_files(control_matrix_files, mutant_matrix_files,
                                  min_event_length=15, curvature_threshold=2.5,
                                  local_window=0, max_gap=1, max_displacement=20)
