import pandas as pd
from functions import *
from sklearn.model_selection import train_test_split


def main():

    # Load your CSV data into a DataFrame
    # csv_file_path = '../nagi_annotation/annotation_241114.csv'  # Replace with your actual CSV file path
    # csv_file_path = '../OneDrive_1_2024-9-6/labels_white3rd_10fps-5min_test_1011.csv'
    csv_file_path = "../nagi_annotation_1115/annotation_241115.csv"
    df = pd.read_csv(csv_file_path)
    # Group by frame_idx
    grouped = [group for _, group in df.groupby("frame_idx")]
    # Split the groups into train and test sets
    train_groups, val_groups = train_test_split(
        grouped, test_size=0.25, random_state=42
    )
    # Concatenate groups back into DataFrames
    train_df = pd.concat(train_groups).reset_index(drop=True)
    val_df = pd.concat(val_groups).reset_index(drop=True)

    # Define paths
    ## Original Train data
    # video_path = '../OneDrive_1_2024-9-6/white3rd_10fps-5min_test.avi'
    # train_annotations_dir = './train_annotations'
    # val_annotations_dir = './val_annotations'  # Path to save validation YOLO annotations and images
    # yaml_path = './yolo_pose_config_1115.yaml'

    ## New Train Data
    # video_path = '../nagi_annotation/Training_data_Background-proc.avi'  # Path to your video file
    # train_annotations_dir = './train_annotations'  # Path to save training YOLO annotations and images
    # mask_dir = '../nagi_annotation/mask_figs'
    # val_annotations_dir = './val_annotations'  # Path to save validation YOLO annotations and images
    # yaml_path = './yolo_pose_config_1115.yaml'  # Path to save the YAML configuration file

    ## 20241115 version
    video_path = "../nagi_annotation_1115/white3rd_10fps-5min_test.avi"
    train_annotations_dir = "train_annotations_1125"
    val_annotations_dir = (
        "val_annotations_1125"  # Path to save validation YOLO annotations and images
    )
    yaml_path = "./yolo_pose_config_1125.yaml"
    mask_dir = "../nagi_annotation_1125/mask_figs"

    image_df = ""

    # Create YOLO annotations and YAML config
    create_yolo_annotations_with_mask(
        train_df,
        video_path,
        mask_dir,
        train_annotations_dir,
        augment=True,
        target_size=10000,
    )
    create_yolo_annotations_with_mask(
        val_df, video_path, mask_dir, val_annotations_dir, augment=False
    )
    # create_yolo_annotations(train_df, video_path,  train_annotations_dir, augment=True, target_size=10000)
    # create_yolo_annotations(val_df, video_path,  val_annotations_dir, augment=False)
    create_yolo_annotations_from_images(
        train_df, image_df, train_annotations_dir, augment=True, target_size=10000
    )
    create_yolo_annotations_from_images(
        val_df, image_df, val_annotations_dir, augment=False
    )
    create_yolo_pose_yaml(train_annotations_dir, val_annotations_dir, yaml_path)

    print(
        f"Training annotations saved in {train_annotations_dir}, validation annotations in {val_annotations_dir}, and YAML configuration saved at {yaml_path}."
    )


main()
