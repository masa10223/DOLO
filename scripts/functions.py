import cv2
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage, Keypoint
import numpy as np
import random
import imageio
import matplotlib.pyplot as plt
from ultralytics import YOLO
from matplotlib.lines import Line2D  # 凡例作成のために追加
from scipy.optimize import linear_sum_assignment  # 追加

import csv


############################
#### Augmentation Part  ####
############################


def create_yolo_annotations_with_mask(
    df, video_path, mask_dir, annotations_dir, augment=False, target_size=1000
):
    """
    Converts coordinate data into YOLO format annotations for YOLO-Pose,
    using both video frames and corresponding mask images to enhance texture detection.
    Optionally applies augmentation to increase the dataset size.

    Parameters:
    df (pd.DataFrame): DataFrame containing the coordinates data.
    video_path (str): Path to the video file.
    mask_dir (str): Directory containing the mask images.
    annotations_dir (str): Path to save the YOLO format annotation files.
    augment (bool): Whether to apply augmentation to the data.
    target_size (int): Target number of augmented images for the training set.

    Returns:
    None: Saves annotation files in the specified directory.
    """
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Augmentation settings
    aug_seq = (
        iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
                iaa.Affine(
                    rotate=(-15, 15)
                ),  # Random rotation between -25 and 25 degrees
                iaa.Affine(translate_px={"x": (-30, 30), "y": (-10, 10)}),
                iaa.ScaleX((0.9, 1.1)),  # Random scaling along X-axis
                iaa.ScaleY((0.9, 1.1)),  # Random scaling along Y-axis
            ]
        )
        if augment
        else None
    )

    # グループ化（frame_idxごとに処理）
    grouped = df.groupby("frame_idx")

    for frame_idx, group in grouped:
        # ビデオフレームを設定
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # フレームを読み込み
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read the frame at index {frame_idx}.")
            continue

        # マスク画像の読み込み
        mask_path = os.path.join(mask_dir, f"{frame_idx}.tif")
        if not os.path.exists(mask_path):
            print(f"Warning: Mask image for frame {frame_idx} not found. Skipping.")
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # マスク画像のサイズをフレームサイズに合わせる
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # マスクを適用してテクスチャを強調
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # フレームとアノテーションを保存
        save_frame_and_annotation(masked_frame, group, annotations_dir, frame_idx, 0)

        # データ拡張が有効な場合、ターゲットサイズまで拡張
        if augment:
            current_count = len(group)
            while current_count < target_size:
                aug_frame, aug_group = apply_augmentation(masked_frame, group, aug_seq)
                save_frame_and_annotation(
                    aug_frame,
                    aug_group,
                    annotations_dir,
                    frame_idx,
                    current_count,
                    augmented=True,
                )
                current_count += len(aug_group)

    # ビデオキャプチャを解放
    cap.release()

    print(f"Processing completed. Annotations saved in '{annotations_dir}'.")


def create_yolo_annotations(
    df, video_path, annotations_dir, augment=False, target_size=1000
):
    """
    Converts coordinate data into YOLO format annotations for YOLO-Pose,
    matching each annotation to the correct video frame. Optionally applies augmentation
    to increase the dataset size.

    Parameters:
    df (pd.DataFrame): DataFrame containing the coordinates data.
    video_path (str): Path to the video file.
    annotations_dir (str): Path to save the YOLO format annotation files.
    augment (bool): Whether to apply augmentation to the data.
    target_size (int): Target number of augmented images for the training set.

    Returns:
    None: Saves annotation files in the specified directory.
    """
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    aug_seq = (
        iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
                iaa.Affine(
                    rotate=(-25, 25)
                ),  # Random rotation between -25 and 25 degrees
                iaa.Affine(translate_px={"x": (-30, 30), "y": (-20, 20)}),
                iaa.ScaleX((0.95, 1.05)),  # Random scaling along X-axis
                iaa.ScaleY((0.95, 1.05)),  # Random scaling along Y-axis
            ]
        )
        if augment
        else None
    )

    # Group by frame index to handle multiple objects per frame
    grouped = df.groupby("frame_idx")

    for frame_idx, group in grouped:
        # Set the video to the specified frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the specified frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read the frame at index {frame_idx}.")
            continue

        # Save the original frame and annotation
        save_frame_and_annotation(frame, group, annotations_dir, frame_idx, 0)

        # Apply augmentation if enabled and target size not reached
        if augment:
            current_count = len(group)
            while current_count < target_size:
                aug_frame, aug_group = apply_augmentation(frame, group, aug_seq)
                save_frame_and_annotation(
                    aug_frame,
                    aug_group,
                    annotations_dir,
                    frame_idx,
                    current_count,
                    augmented=True,
                )
                current_count += len(aug_group)

    # Release the video capture object
    cap.release()


def create_yolo_annotations_from_images(
    df, image_dir, annotations_dir, augment=False, target_size=1000
):
    """
    Converts coordinate data into YOLO format annotations for YOLO-Pose,
    matching each annotation to the corresponding image. Optionally applies augmentation
    to increase the dataset size.

    Parameters:
    df (pd.DataFrame): DataFrame containing the coordinates data.
    image_dir (str): Path to the directory containing images.
    annotations_dir (str): Path to save the YOLO format annotation files.
    augment (bool): Whether to apply augmentation to the data.
    target_size (int): Target number of augmented images for the training set.

    Returns:
    None: Saves annotation files in the specified directory.
    """
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    aug_seq = (
        iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
                iaa.Affine(
                    rotate=(-25, 25)
                ),  # Random rotation between -25 and 25 degrees
                iaa.Affine(translate_px={"x": (-30, 30), "y": (-20, 20)}),
                iaa.ScaleX((0.95, 1.05)),  # Random scaling along X-axis
                iaa.ScaleY((0.95, 1.05)),  # Random scaling along Y-axis
            ]
        )
        if augment
        else None
    )

    # Group by image name to handle multiple objects per image
    grouped = df.groupby("frame_idx")

    for frame_idx, group in grouped:
        image_path = os.path.join(image_dir, f"{frame_idx}.tif")
        if not os.path.exists(image_path):
            print(f"Error: Image file {frame_idx} not found in {image_dir}.")
            continue

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read the image {frame_idx}.")
            continue

        # Save the original image and annotation
        save_frame_and_annotation(image, group, annotations_dir, frame_idx, 0)

        # Apply augmentation if enabled and target size not reached
        if augment:
            current_count = len(group)
            while current_count < target_size:
                aug_image, aug_group = apply_augmentation(image, group, aug_seq)
                save_frame_and_annotation(
                    aug_image,
                    aug_group,
                    annotations_dir,
                    frame_idx,
                    current_count,
                    augmented=True,
                )
                current_count += len(aug_group)


def save_frame_and_annotation(
    frame, group, annotations_dir, frame_idx, count, augmented=False
):
    """
    Saves the frame as an image and writes the corresponding YOLO annotation file.

    Parameters:
    frame (ndarray): Image frame from the video.
    group (DataFrame): DataFrame group containing keypoints for multiple objects.
    annotations_dir (str): Directory to save the annotations.
    frame_idx (int): Frame index from the video.
    count (int): Counter for file naming.
    augmented (bool): Whether the frame is augmented.

    Returns:
    None
    """
    suffix = "_aug" if augmented else ""
    image_filename = f"{frame_idx}_{count}{suffix}.jpg"
    image_path = os.path.join(annotations_dir, image_filename)
    cv2.imwrite(image_path, frame)

    # Initialize a list to store annotation lines
    annotation_lines = []

    frame_height, frame_width = frame.shape[:2]

    # Iterate over each object in the group
    for _, row in group.iterrows():
        # Calculate bounding box and keypoints (normalized to image dimensions)
        x_center = (row["Head.x"] + row["Tail.x"]) / 2 / frame_width
        y_center = (row["Head.y"] + row["Tail.y"]) / 2 / frame_height
        width = abs(row["Tail.x"] - row["Head.x"]) / frame_width
        height = abs(row["Tail.y"] - row["Head.y"]) / frame_height

        # Define keypoints (normalized x, y, visibility)
        keypoints = [
            row["Head.x"] / frame_width,
            row["Head.y"] / frame_height,
            2,
            row["mid.x"] / frame_width,
            row["mid.y"] / frame_height,
            2,
            row["Tail.x"] / frame_width,
            row["Tail.y"] / frame_height,
            2,
        ]

        # Combine into a single line for YOLO format
        annotation_line = f"0 {x_center} {y_center} {width} {height} " + " ".join(
            map(str, keypoints)
        )
        annotation_lines.append(annotation_line)

    # Write all annotation lines to a file
    annotation_filename = f"{frame_idx}_{count}{suffix}.txt"
    annotation_path = os.path.join(annotations_dir, annotation_filename)
    with open(annotation_path, "w") as f:
        f.write("\n".join(annotation_lines))


def apply_augmentation(frame, group, aug_seq):
    """
    Applies augmentation to a frame and corresponding keypoints.

    Parameters:
    frame (ndarray): Original frame image.
    group (DataFrame): DataFrame group containing keypoints for multiple objects.
    aug_seq (imgaug.augmenters.Sequential): Augmentation sequence.

    Returns:
    aug_frame (ndarray): Augmented frame image.
    aug_group (DataFrame): DataFrame group with augmented keypoints data.
    """
    keypoints_list = []

    # Prepare keypoints for all objects in the group
    for _, row in group.iterrows():
        keypoints_list.extend(
            [
                Keypoint(x=row["Head.x"], y=row["Head.y"]),
                Keypoint(x=row["mid.x"], y=row["mid.y"]),
                Keypoint(x=row["Tail.x"], y=row["Tail.y"]),
            ]
        )

    # Wrap keypoints in KeypointsOnImage with image shape
    kps = KeypointsOnImage(keypoints_list, shape=frame.shape)

    # Apply augmentation to both image and keypoints
    aug_frame, aug_kps = aug_seq(image=frame, keypoints=kps)

    # Extract augmented keypoints
    aug_keypoints = aug_kps.keypoints

    # Update the group with augmented keypoints
    aug_group = group.copy()
    for i, (_, row) in enumerate(aug_group.iterrows()):
        aug_group.at[_, "Head.x"], aug_group.at[_, "Head.y"] = (
            aug_keypoints[i * 3].x,
            aug_keypoints[i * 3].y,
        )
        aug_group.at[_, "mid.x"], aug_group.at[_, "mid.y"] = (
            aug_keypoints[i * 3 + 1].x,
            aug_keypoints[i * 3 + 1].y,
        )
        aug_group.at[_, "Tail.x"], aug_group.at[_, "Tail.y"] = (
            aug_keypoints[i * 3 + 2].x,
            aug_keypoints[i * 3 + 2].y,
        )

    return aug_frame, aug_group


def create_yolo_pose_yaml(train_dir, val_dir, yaml_path):
    """
    Creates a YAML file for YOLO-Pose training configuration.

    Parameters:
    train_dir (str): Path to the training images directory.
    val_dir (str): Path to the validation images directory.
    yaml_path (str): Path to save the YAML configuration file.

    Returns:
    None: Saves the YAML configuration file.
    """
    data = {
        "train": f"/cellpose/scripts/{train_dir}",
        "val": f"/cellpose/scripts/{val_dir}",
        "nc": 1,  # Number of classes
        "names": ["pose"],  # Class names
        "kpt_shape": [3, 3],  # 3 keypoints, each with x, y, visibility
        "kpts": 3,  # Number of keypoints (Head, Mid, Tail)
    }

    # Save YAML data
    with open(yaml_path, "w") as file:
        yaml.dump(data, file, sort_keys=False)


def plot_coordinates_on_frame(video_path, csv_path, track_ids=None, frame_idx=None):
    """
    Extracts specific frames from a video and plots the given coordinates for specified tracks and frames.

    Parameters:
    video_path (str): Path to the video file.
    csv_path (str): Path to the CSV file containing coordinates data.
    track_ids (list or None): List of track identifiers to filter coordinates by (e.g., ['track_0', 'track_1']).
                              If None, all tracks will be selected.
    frame_idx (int or None): The index of the frame to extract and plot. If None, all frames will be selected.

    Returns:
    None: Displays the plots with coordinates.
    """
    # Load the CSV data into a DataFrame
    df = pd.read_csv(csv_path)

    # Filter by track_ids if provided
    if track_ids is not None:
        df = df[df["track_id"].isin(track_ids)]

    # Filter by frame_idx if provided
    if frame_idx is not None:
        df = df[df["frame_idx"] == frame_idx]

    # Check if the filtered DataFrame is empty
    if df.empty:
        print("No coordinates found for the specified tracks and frame index.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop over each unique frame index in the filtered DataFrame
    for idx in df["frame_idx"].unique():
        # Set the video to the specified frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # Read the specified frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read the frame at index {idx}.")
            continue

        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Filter the DataFrame for the current frame index
        frame_data = df[df["frame_idx"] == idx]

        # Plot the frame
        plt.imshow(frame)

        # Plot each set of coordinates for the current frame index
        for _, row in frame_data.iterrows():
            plt.scatter(
                [row["Head.x"], row["mid.x"], row["Tail.x"]],
                [row["Head.y"], row["mid.y"], row["Tail.y"]],
                color="red",
                marker="o",
            )
            plt.text(
                row["Head.x"],
                row["Head.y"],
                "Head",
                color="white",
                fontsize=12,
                ha="right",
            )
            plt.text(
                row["mid.x"],
                row["mid.y"],
                "Mid",
                color="white",
                fontsize=12,
                ha="right",
            )
            plt.text(
                row["Tail.x"],
                row["Tail.y"],
                "Tail",
                color="white",
                fontsize=12,
                ha="right",
            )

        plt.title(
            f"Tracks {track_ids if track_ids is not None else 'All'}, Frame {idx} with Coordinates"
        )
        plt.axis("off")
        plt.show()

    # Release the video capture object
    cap.release()


#######
#### Animationm Part
######


def calculate_angle_between_vectors(T, m, H):
    """
    Calculates the angle θ between the vectors Tm_g and m_gH using vector math.

    Parameters:
    T (np.array): Coordinates of the Tail (T) [x, y].
    m (np.array): Coordinates of the Mid (m) [x, y].
    H (np.array): Coordinates of the Head (H) [x, y].

    Returns:
    float: Angle θ in degrees.
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
        return 0.0

    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (magnitude_T_mg * magnitude_mg_H)

    # Clip cos_theta to avoid potential numerical issues outside [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    theta = np.arccos(cos_theta)  # Radians
    theta_degrees = np.degrees(theta)  # Convert to degrees

    return theta_degrees


def annotate_frame_with_keypoints(
    img, keypoints_list, ids_list, angles_list, frame_number, id_to_color
):
    """
    Annotates the image with keypoints for multiple instances and displays angles.
    Returns the annotated image as an array.

    Parameters:
    img (ndarray): Original image in BGR format.
    keypoints_list (list): List of keypoints dictionaries for each instance.
    ids_list (list): List of consistent IDs for each instance.
    angles_list (list): List of angles θ for each instance.
    frame_number (int): The current frame number.
    id_to_color (dict): Mapping from consistent IDs to colors.

    Returns:
    annotated_frame (ndarray): Annotated image in BGR format.
    """
    # Convert BGR to RGB for plotting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plotting with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)

    # Plot keypoints and angles
    for i, keypoints in enumerate(keypoints_list):
        head = keypoints["head"]
        middle = keypoints["middle"]
        tail = keypoints["tail"]
        consistent_id = ids_list[i]
        theta = angles_list[i]

        color = id_to_color.get(
            consistent_id, (1.0, 0.0, 0.0)
        )  # Get color from mapping, default to red

        # Plot keypoints
        ax.scatter(head[0], head[1], color=color, s=60, marker="o")
        ax.scatter(middle[0], middle[1], color=color, s=60, marker="x")
        ax.scatter(tail[0], tail[1], color=color, s=60, marker="^")

        # Display angle θ near the middle point
        ax.text(
            middle[0] + 10,
            middle[1] + 10,
            f"ID:{consistent_id},\n θ={theta:.1f}°",
            color=color,
            fontsize=12,
        )

    # Remove axis
    ax.axis("off")

    # Add legend with IDs and marker explanations
    legend_elements = []
    # 修正開始：IDをソートして凡例を作成
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
    # 修正終了

    # Add marker explanations
    marker_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            label="Head",
            markerfacecolor="gray",
            markersize=10,
        ),
        Line2D([0], [0], marker="x", color="gray", label="Middle", markersize=10),
        Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            label="Tail",
            markerfacecolor="gray",
            markersize=10,
        ),
    ]

    # Combine legends
    first_legend = ax.legend(handles=legend_elements, loc="upper right", title="IDs")
    ax.add_artist(first_legend)
    ax.legend(handles=marker_elements, loc="upper left", title="Keypoints")

    # Add frame number in red color
    ax.text(
        10, 90, f"Frame: {frame_number}", color="red", fontsize=16, fontweight="bold"
    )

    # Remove margins
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)

    # Convert Matplotlib figure to a numpy array
    fig.canvas.draw()
    annotated_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated_frame = annotated_frame.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    plt.close()

    # Convert RGB to BGR for consistency with OpenCV
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    return annotated_frame


def determine_head_tail_based_on_angle(head, middle, tail):
    """
    Determines the head and tail based on the angles at the middle point.
    Returns the keypoints with head and tail possibly swapped.
    """
    # Vectors from middle to head and tail
    vector_head = head - middle
    vector_tail = tail - middle

    # Calculate the angles
    angle_head = np.arctan2(vector_head[1], vector_head[0])
    angle_tail = np.arctan2(vector_tail[1], vector_tail[0])

    # Calculate the absolute difference between the angles
    angle_difference = np.abs(angle_head - angle_tail)
    angle_difference = (
        angle_difference if angle_difference <= np.pi else 2 * np.pi - angle_difference
    )

    # Use the smaller angle to determine the head
    if angle_difference < np.pi / 2:
        # The side with the smaller vector magnitude is the head
        if np.linalg.norm(vector_head) <= np.linalg.norm(vector_tail):
            return head, middle, tail
        else:
            return tail, middle, head  # Swap head and tail
    else:
        # The side forming the sharper angle is the head
        angle_at_middle_head = calculate_angle_at_joint(tail, middle, head)
        angle_at_middle_tail = calculate_angle_at_joint(head, middle, tail)

        if angle_at_middle_head <= angle_at_middle_tail:
            return head, middle, tail
        else:
            return tail, middle, head  # Swap head and tail


def calculate_angle_at_joint(p1, p2, p3):
    """
    Calculates the angle at point p2 formed by the lines p2p1 and p2p3.
    """
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


def ensure_head_in_direction_of_accumulated_movement(
    head,
    middle,
    tail,
    consistent_id,
    positions_history,
    orientation_fixed,
    head_tail_mapping,
):
    N = 30  # Number of frames to accumulate movement
    angle_threshold = 45  # Angle threshold in degrees to reset orientation
    movement_threshold = (
        5.0  # Minimum movement distance to consider movement vector reliable
    )

    if consistent_id not in positions_history:
        positions_history[consistent_id] = []
        orientation_fixed[consistent_id] = False

    positions_history[consistent_id].append(middle)

    # Keep only the last 2N positions
    if len(positions_history[consistent_id]) > 2 * N:
        positions_history[consistent_id] = positions_history[consistent_id][-2 * N :]

    # Calculate movement vector
    if len(positions_history[consistent_id]) >= N:
        movement_vector = (
            positions_history[consistent_id][-1] - positions_history[consistent_id][-N]
        )
        movement_distance = np.linalg.norm(movement_vector)

        if movement_distance >= movement_threshold:
            movement_vector = movement_vector / (movement_distance + 1e-6)

            # オリエンテーションが未固定の場合
            if not orientation_fixed[consistent_id]:
                # Vectors from middle to head and tail
                head_vector = head - middle
                head_vector = head_vector / (np.linalg.norm(head_vector) + 1e-6)
                tail_vector = tail - middle
                tail_vector = tail_vector / (np.linalg.norm(tail_vector) + 1e-6)

                # Compare dot products
                head_similarity = np.dot(movement_vector, head_vector)
                tail_similarity = np.dot(movement_vector, tail_vector)

                # Decide correct orientation
                if head_similarity >= tail_similarity:
                    head_tail_mapping[consistent_id] = {"head": "head", "tail": "tail"}
                else:
                    head_tail_mapping[consistent_id] = {"head": "tail", "tail": "head"}
                    head, tail = tail, head  # Swap head and tail

                orientation_fixed[consistent_id] = True

            else:
                # 移動方向の変化を監視
                if len(positions_history[consistent_id]) >= 2 * N:
                    previous_movement_vector = (
                        positions_history[consistent_id][-N]
                        - positions_history[consistent_id][-2 * N]
                    )
                    previous_movement_vector = previous_movement_vector / (
                        np.linalg.norm(previous_movement_vector) + 1e-6
                    )

                    # 角度の変化を計算
                    angle_change = np.arccos(
                        np.clip(
                            np.dot(movement_vector, previous_movement_vector), -1.0, 1.0
                        )
                    )
                    angle_change_degrees = np.degrees(angle_change)

                    if angle_change_degrees > angle_threshold:
                        orientation_fixed[consistent_id] = False  # Reset orientation

        else:
            # Movement is too small, use angle-based method
            head, middle, tail = determine_head_tail_based_on_angle(head, middle, tail)
            orientation_fixed[consistent_id] = (
                True  # Assume orientation is fixed for now
            )

    else:
        # Not enough data, use angle-based method
        head, middle, tail = determine_head_tail_based_on_angle(head, middle, tail)
        orientation_fixed[consistent_id] = True  # Assume orientation is fixed for now

    # Use fixed orientation if available
    if orientation_fixed[consistent_id] and consistent_id in head_tail_mapping:
        if head_tail_mapping[consistent_id]["head"] == "tail":
            head, tail = tail, head

    return head, middle, tail


def process_video_to_gif_with_angles(
    video_path,
    output_gif_path,
    model_path="./runs/pose/train/weights/best.pt",
    frame_skip=1,
    distance_threshold=30,
    max_consistent_id=15,
    output_csv_path="./output_positions_angles.csv",
    confidence=0.01,
):
    """
    動画を処理し、各フレームに対して推論を行い、角度を計算し、
    キーポイントと角度でフレームに注釈を付け、結果をループするGIFに変換します。
    さらに、各IDの位置と角度のデータをCSVファイルに保存します。
    """
    # YOLOモデルをロード
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"YOLOモデルの読み込みエラー: {e}")
        return

    # ビデオファイルを開く
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        print(f"ビデオファイルのオープンエラー: {e}")
        return
    frame_count = 0

    # マッピングとデータ構造の初期化
    consistentid_to_last_keypoints = {}
    consistentid_to_last_position = {}
    consistentid_to_last_theta = {}
    consistentid_to_velocity = {}
    # 一貫した色割り当ての初期化
    id_to_color = {}
    color_palette = plt.get_cmap("tab20", max_consistent_id)
    for consistent_id in range(1, max_consistent_id + 1):
        color_index = (consistent_id - 1) % color_palette.N
        color = color_palette(color_index)[:3]
        id_to_color[consistent_id] = color

    # 新しいデータ構造の初期化
    positions_history = {}  # 各IDの過去のmiddle位置を保存
    orientation_fixed = {}  # 各IDの方向が固定されたかどうか
    head_tail_mapping = {}  # 各IDのheadとtailのマッピング

    max_missing_frames = 50
    consistentid_to_last_seen = {}
    available_consistent_ids = set(range(1, max_consistent_id + 1))
    active_consistent_ids = set()

    # GIFとCSVの準備
    try:
        gif_writer = imageio.get_writer(output_gif_path, mode="I", fps=10, loop=0)
    except Exception as e:
        print(f"GIFライターの初期化中にエラーが発生しました: {e}")
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

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(
                        f"フレーム {frame_count}: フレームを読み込めないか、ビデオの終了です。"
                    )
                    break

                # フレームスキップ
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                # YOLOモデル推論
                try:
                    results = model(frame, conf=confidence)
                except Exception as e:
                    print(f"フレーム {frame_count}: モデル推論中のエラー - {e}")
                    frame_count += 1
                    continue

                keypoints_list = []
                ids_list = []
                angles_list = []
                consistent_ids_in_frame = set()

                if results:
                    result = results[0]
                    if result.keypoints is not None:
                        keypoints_data = result.keypoints.data.cpu().numpy()
                        num_instances = keypoints_data.shape[0]

                        current_positions = []
                        current_keypoints_list = []

                        for idx in range(num_instances):
                            keypoints = keypoints_data[idx]
                            if keypoints.shape[0] >= 3:
                                tail = keypoints[2, :2]
                                middle = keypoints[1, :2]
                                head = keypoints[0, :2]

                                current_positions.append(middle)
                                current_keypoints_list.append(
                                    {"head": head, "middle": middle, "tail": tail}
                                )

                        # IDの一致処理
                        if current_positions:
                            # 予測位置の計算
                            predicted_positions = []
                            for cid in consistentid_to_last_position:
                                last_position = consistentid_to_last_position[cid]
                                velocity = consistentid_to_velocity.get(
                                    cid, np.array([0, 0])
                                )
                                predicted_position = last_position + velocity
                                predicted_positions.append(predicted_position)

                            # 変数を初期化
                            assigned_indices = set()
                            assigned_cids = set()
                            cid_list = list(consistentid_to_last_position.keys())

                            if predicted_positions:
                                cost_matrix = np.zeros(
                                    (len(predicted_positions), len(current_positions))
                                )

                                for i, predicted_position in enumerate(
                                    predicted_positions
                                ):
                                    for j, current_position in enumerate(
                                        current_positions
                                    ):
                                        distance = np.linalg.norm(
                                            current_position - predicted_position
                                        )
                                        cost_matrix[i, j] = distance

                                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                                # 既存のIDを一致させる処理
                                for i, j in zip(row_ind, col_ind):
                                    if cost_matrix[i, j] < distance_threshold:
                                        cid = cid_list[i]
                                        consistent_id = cid
                                        assigned_cids.add(cid)
                                        assigned_indices.add(j)
                                        consistentid_to_last_seen[consistent_id] = (
                                            frame_count
                                        )
                                        consistent_ids_in_frame.add(consistent_id)

                                        keypoints = current_keypoints_list[j]
                                        head, middle, tail = (
                                            keypoints["head"],
                                            keypoints["middle"],
                                            keypoints["tail"],
                                        )

                                        # 進行方向に合わせてheadとtailを修正
                                        head, middle, tail = (
                                            ensure_head_in_direction_of_accumulated_movement(
                                                head,
                                                middle,
                                                tail,
                                                consistent_id,
                                                positions_history,
                                                orientation_fixed,
                                                head_tail_mapping,
                                            )
                                        )

                                        # 速度の更新
                                        previous_middle = consistentid_to_last_position[
                                            consistent_id
                                        ]
                                        velocity = middle - previous_middle
                                        consistentid_to_velocity[consistent_id] = (
                                            velocity
                                        )

                                        color = id_to_color[consistent_id]

                                        theta = calculate_angle_between_vectors(
                                            tail, middle, head
                                        )

                                        angles_list.append(theta)
                                        consistentid_to_last_keypoints[
                                            consistent_id
                                        ] = {
                                            "head": head,
                                            "middle": middle,
                                            "tail": tail,
                                        }
                                        consistentid_to_last_position[consistent_id] = (
                                            middle
                                        )
                                        consistentid_to_last_theta[consistent_id] = (
                                            theta
                                        )
                                        keypoints_list.append(
                                            {
                                                "head": head,
                                                "middle": middle,
                                                "tail": tail,
                                            }
                                        )
                                        ids_list.append(consistent_id)

                                        # CSVにデータ保存
                                        csv_writer.writerow(
                                            [
                                                frame_count,
                                                consistent_id,
                                                head[0],
                                                head[1],
                                                middle[0],
                                                middle[1],
                                                tail[0],
                                                tail[1],
                                                theta,
                                            ]
                                        )

                            # 新規IDの割り当て
                            for idx in range(len(current_positions)):
                                if idx not in assigned_indices:
                                    if available_consistent_ids:
                                        consistent_id = min(available_consistent_ids)
                                        available_consistent_ids.remove(consistent_id)
                                        active_consistent_ids.add(consistent_id)
                                        consistentid_to_last_seen[consistent_id] = (
                                            frame_count
                                        )
                                        consistent_ids_in_frame.add(consistent_id)

                                        keypoints = current_keypoints_list[idx]
                                        head, middle, tail = (
                                            keypoints["head"],
                                            keypoints["middle"],
                                            keypoints["tail"],
                                        )

                                        # 速度はゼロで初期化
                                        consistentid_to_velocity[consistent_id] = (
                                            np.array([0, 0])
                                        )

                                        # 位置関係でheadとtailを修正（初期フレーム）
                                        head, middle, tail = (
                                            ensure_head_in_direction_of_accumulated_movement(
                                                head,
                                                middle,
                                                tail,
                                                consistent_id,
                                                positions_history,
                                                orientation_fixed,
                                                head_tail_mapping,
                                            )
                                        )

                                        # 色を割り当て
                                        color = id_to_color[consistent_id]

                                        # データ更新
                                        theta = calculate_angle_between_vectors(
                                            tail, middle, head
                                        )
                                        angles_list.append(theta)
                                        consistentid_to_last_keypoints[
                                            consistent_id
                                        ] = {
                                            "head": head,
                                            "middle": middle,
                                            "tail": tail,
                                        }
                                        consistentid_to_last_position[consistent_id] = (
                                            middle
                                        )
                                        consistentid_to_last_theta[consistent_id] = (
                                            theta
                                        )
                                        keypoints_list.append(
                                            {
                                                "head": head,
                                                "middle": middle,
                                                "tail": tail,
                                            }
                                        )
                                        ids_list.append(consistent_id)

                                        # CSVに保存
                                        csv_writer.writerow(
                                            [
                                                frame_count,
                                                consistent_id,
                                                head[0],
                                                head[1],
                                                middle[0],
                                                middle[1],
                                                tail[0],
                                                tail[1],
                                                theta,
                                            ]
                                        )

                else:
                    print(
                        f"フレーム {frame_count}: モデルからの結果がありませんでした。"
                    )

                if keypoints_list:
                    try:
                        annotated_frame = annotate_frame_with_keypoints(
                            frame,
                            keypoints_list,
                            ids_list,
                            angles_list,
                            frame_count,
                            id_to_color,
                        )
                        # OpenCVの画像としてGIFに追加
                        annotated_frame_rgb = cv2.cvtColor(
                            annotated_frame, cv2.COLOR_BGR2RGB
                        )
                        gif_writer.append_data(annotated_frame_rgb)
                        print(f"フレーム {frame_count} をGIFに追加しました。")
                    except Exception as e:
                        print(f"フレーム {frame_count} の注釈中のエラー: {e}")
                        continue

                # 一定時間検出されていないIDの解放
                for consistent_id in list(active_consistent_ids):
                    if (
                        frame_count - consistentid_to_last_seen[consistent_id]
                        > max_missing_frames
                    ):
                        active_consistent_ids.remove(consistent_id)
                        available_consistent_ids.add(consistent_id)
                        # データの削除（popを使用してKeyErrorを防止）
                        consistentid_to_last_position.pop(consistent_id, None)
                        consistentid_to_velocity.pop(consistent_id, None)
                        consistentid_to_last_keypoints.pop(consistent_id, None)
                        consistentid_to_last_theta.pop(consistent_id, None)
                        # id_to_color.pop(consistent_id, None)
                        positions_history.pop(consistent_id, None)
                        orientation_fixed.pop(consistent_id, None)
                        head_tail_mapping.pop(consistent_id, None)
                frame_count += 1

            print(
                f"処理が完了しました。GIFは {output_gif_path} に、CSVは {output_csv_path} に保存されました。"
            )

    except Exception as e:
        print(f"CSVファイルのオープンまたは書き込み中にエラーが発生しました: {e}")
    finally:
        cap.release()
        gif_writer.close()


######################################
######### Avoidance Detection
####################################


def detect_avoidance_behavior_gradual(
    data,
    min_increase_frames=10,
    max_increase_frames=30,
    angle_increase_threshold=30.0,
    plateau_threshold=5.0,
    plateau_frames=5,
):
    """
    Corrected_Angleの徐々な増加とプラトー化による回避行動を検出します。

    パラメータ:
    data (pd.DataFrame): 'ID', 'Frame', 'Middle_X', 'Middle_Y', 'Corrected_Angle' を含むデータフレーム。
    min_increase_frames (int): 角度が増加する最小フレーム数。
    max_increase_frames (int): 角度が増加する最大フレーム数。
    angle_increase_threshold (float): 角度の総増加量のしきい値（度数法）。
    plateau_threshold (float): プラトー期間中の角度変化の最大値（度数法）。
    plateau_frames (int): プラトー期間の最小フレーム数。

    戻り値:
    pd.DataFrame: 回避行動イベントのリストを含むデータフレーム。
    列は 'ID', 'Start_Frame', 'End_Frame', 'Duration'。
    """
    # 必要な列が存在するか確認
    required_columns = ["ID", "Frame", "Middle_X", "Middle_Y", "Corrected_Angle"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"必要な列 '{col}' がデータに存在しません。")

    # 個体ごとに処理
    avoidance_events = []
    ids = data["ID"].unique()

    for id_value in tqdm(ids, desc="Processing IDs"):
        individual_data = data[data["ID"] == id_value].copy()
        individual_data = individual_data.sort_values(by="Frame").reset_index(drop=True)

        # Corrected_Angleのスムージング（移動平均）
        window_size = 5  # スムージングのウィンドウサイズ（必要に応じて調整）
        individual_data["Smoothed_Angle"] = (
            individual_data["Corrected_Angle"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

        # 角度の差分を計算
        individual_data["Angle_Diff"] = individual_data["Smoothed_Angle"].diff()

        # 回避行動の期間を検出
        frames = individual_data["Frame"].values
        angles = individual_data["Smoothed_Angle"].values
        angle_diffs = individual_data["Angle_Diff"].values

        increasing = False
        start_increase_idx = None
        for i in range(1, len(angles)):
            if np.isnan(angle_diffs[i]):
                continue

            # 角度が増加しているか確認
            if angle_diffs[i] > 0:
                if not increasing:
                    # 増加開始
                    increasing = True
                    start_increase_idx = i - 1
            else:
                if increasing:
                    # 増加終了
                    increasing = False
                    end_increase_idx = i - 1
                    increase_duration = end_increase_idx - start_increase_idx + 1

                    # 増加期間が条件を満たすか確認
                    if min_increase_frames <= increase_duration <= max_increase_frames:
                        total_angle_increase = (
                            angles[end_increase_idx] - angles[start_increase_idx]
                        )

                        if total_angle_increase >= angle_increase_threshold:
                            # プラトー期間を探す
                            plateau_start_idx = end_increase_idx + 1
                            plateau_end_idx = plateau_start_idx

                            while plateau_end_idx < len(angles):
                                angle_change = np.abs(
                                    angles[plateau_end_idx] - angles[plateau_start_idx]
                                )
                                if angle_change <= plateau_threshold:
                                    plateau_end_idx += 1
                                else:
                                    break

                            plateau_duration = plateau_end_idx - plateau_start_idx

                            # プラトー期間が条件を満たすか確認
                            if plateau_duration >= plateau_frames:
                                # 回避行動の期間を記録
                                event = {
                                    "ID": id_value,
                                    "Start_Frame": frames[start_increase_idx],
                                    "End_Frame": frames[plateau_end_idx - 1],
                                    "Duration": frames[plateau_end_idx - 1]
                                    - frames[start_increase_idx]
                                    + 1,
                                }
                                avoidance_events.append(event)

        # 増加が継続している場合の処理
        if increasing:
            end_increase_idx = len(angles) - 1
            increase_duration = end_increase_idx - start_increase_idx + 1

            if min_increase_frames <= increase_duration <= max_increase_frames:
                total_angle_increase = (
                    angles[end_increase_idx] - angles[start_increase_idx]
                )

                if total_angle_increase >= angle_increase_threshold:
                    # プラトー期間を探す
                    plateau_start_idx = end_increase_idx + 1
                    plateau_end_idx = plateau_start_idx

                    while plateau_end_idx < len(angles):
                        angle_change = np.abs(
                            angles[plateau_end_idx] - angles[plateau_start_idx]
                        )
                        if angle_change <= plateau_threshold:
                            plateau_end_idx += 1
                        else:
                            break

                    plateau_duration = plateau_end_idx - plateau_start_idx

                    if plateau_duration >= plateau_frames:
                        event = {
                            "ID": id_value,
                            "Start_Frame": frames[start_increase_idx],
                            "End_Frame": frames[plateau_end_idx - 1],
                            "Duration": frames[plateau_end_idx - 1]
                            - frames[start_increase_idx]
                            + 1,
                        }
                        avoidance_events.append(event)

    # 回避行動イベントをデータフレームに変換
    avoidance_df = pd.DataFrame(avoidance_events)

    return avoidance_df


def correct_angles_per_id(data, threshold=45, accept_threshold=35):
    """
    Corrects sudden large changes in angle by swapping Head and Tail positions when necessary,
    applied separately for each ID in the DataFrame. Invalid frames are dropped.

    Parameters:
    data (pd.DataFrame): DataFrame containing the coordinates of Tail, Mid, Head, Angle, and ID.
    threshold (float): Angle change threshold to detect sudden changes (default 45 degrees).
    accept_threshold (float): Acceptable angle difference after correction (default 35 degrees).

    Returns:
    pd.DataFrame: DataFrame with corrected coordinates and angles for each ID, excluding invalid frames.
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
        group_Tail_x = group["Tail_X"].values.copy()
        group_Tail_y = group["Tail_Y"].values.copy()
        group_Mid_x = group["Middle_X"].values
        group_Mid_y = group["Middle_Y"].values
        group_Head_x = group["Head_X"].values.copy()
        group_Head_y = group["Head_Y"].values.copy()

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
                    angle_swapped = calculate_angle_between_vectors(
                        T_swapped, m, H_swapped
                    )
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
        corrected_group["Tail_X"] = Tail_x
        corrected_group["Tail_Y"] = Tail_y
        corrected_group["Head_X"] = Head_x
        corrected_group["Head_Y"] = Head_y
        corrected_group["Corrected_Angle"] = corrected_angles

        return corrected_group

    # Apply the correction function to each group identified by 'ID'
    corrected_data = data.groupby("ID", group_keys=False).apply(correct_angles)

    return corrected_data
