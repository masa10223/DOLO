import pandas as pd
from functions import create_yolo_annotations_with_mask, create_yolo_pose_yaml
from sklearn.model_selection import train_test_split


def main():
    train_annotations_dir = "train_annotations_1212"
    # Path to save validation YOLO annotations and images
    val_annotations_dir = "val_annotations_1212"
    yaml_path = "./yolo_pose_config_1212.yaml"

    csv_files = [
        "../annotations/annotation_241124double.csv",
        "../annotations/annotation_241124white.csv",
        "../annotations/annotation_241115.csv",
    ]

    mask_dirs = [
        "../annotations/maskfig_1124double",
        "../annotations/maskfig_1124white",
        "../annotations/maskfig_1115",
    ]

    video_dirs = [
        "../video/Training_1124double.avi",
        "../video/Training_1124white.avi",
        "../video/white3rd_10fps-5min_test.avi",
    ]

    for csv_file_path, mask_dir, video_dir in zip(csv_files, mask_dirs, video_dirs):
        # Load your CSV data into a DataFrame
        df = pd.read_csv(csv_file_path)
        # Group by frame_idx
        grouped = [group for _, group in df.groupby("frame_idx")]
        # Split the groups into train and test sets
        train_groups, val_groups = train_test_split(
            grouped, test_size=0.10, random_state=42
        )
        # Concatenate groups back into DataFrames
        train_df = pd.concat(train_groups).reset_index(drop=True)
        val_df = pd.concat(val_groups).reset_index(drop=True)

        create_yolo_annotations_with_mask(
            train_df,
            video_dir,
            mask_dir,
            train_annotations_dir,
            augment=True,
            target_size=1000,
        )
        create_yolo_annotations_with_mask(
            val_df, video_dir, mask_dir, val_annotations_dir, augment=False
        )

    create_yolo_pose_yaml(train_annotations_dir, val_annotations_dir, yaml_path)

    print(
        f"Training annotations saved in {train_annotations_dir}, validation annotations in {val_annotations_dir}, and YAML configuration saved at {yaml_path}."
    )


main()


# for csv_file_path, img_dir in zip(csv_files, img_dirs):
#     # Load your CSV data into a DataFrame
#     df = pd.read_csv(csv_file_path)
#     # Group by frame_idx
#     grouped = [group for _, group in df.groupby("frame_idx")]
#     # Split the groups into train and test sets
#     train_groups, val_groups = train_test_split(
#         grouped, test_size=0.25, random_state=42
#     )
#     # Concatenate groups back into DataFrames
#     train_df = pd.concat(train_groups).reset_index(drop=True)
#     val_df = pd.concat(val_groups).reset_index(drop=True)


#     create_yolo_annotations_from_images(
#         train_df, img_dir, train_annotations_dir, augment=True, target_size=2000
#     )
#     create_yolo_annotations_from_images(
#         val_df, img_dir, val_annotations_dir, augment=False
#     )
