import pandas as pd
from functions import create_yolo_annotations_with_mask, create_yolo_pose_yaml
from sklearn.model_selection import train_test_split
import cv2
import tifffile
import os


def create_masktif(arguments):
    unique_name = arguments.unique_name
    annotation_csv_path = (
        f"../annotations/overfittings/csvs/annotation_{unique_name}_manual.csv"
    )
    movie_path = f"../video/250703_melano/{unique_name}.avi"
    tif_annotation_dir = f"../annotations/overfittings/tiffs/{unique_name}"
    annot_df = pd.read_csv(annotation_csv_path)
    annot_df = annot_df.dropna()
    # annot_df['frame_idx'] = annot_df['frame_idx'].astype(int)
    for frame_idx in annot_df["frame_idx"].unique():
        cap = cv2.VideoCapture(movie_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        _, frame = cap.read()
        os.makedirs(tif_annotation_dir, exist_ok=True)
        tifffile.imwrite(f"{tif_annotation_dir}/{int(frame_idx)}.tif", frame)


def create_yolo(arguments):
    unique_name = arguments.unique_name
    annotation_csv_path = (
        f"../annotations/overfittings/csvs/annotation_{unique_name}_manual.csv"
    )
    movie_path = f"../video/250703_melano/{unique_name}.avi"
    tif_annotation_dir = f"../annotations/overfittings/tiffs/{unique_name}"
    train_annotations_dir = (
        f"../annotations/overfittings/train_annotations/{unique_name}"
    )
    val_annotations_dir = f"../annotations/overfittings/val_annotations/{unique_name}"
    yaml_path = f"../annotations/overfittings/yamls/{unique_name}.yaml"

    # Load your CSV data into a DataFrame
    df = pd.read_csv(annotation_csv_path)
    # Group by frame_idx
    grouped = [group for _, group in df.groupby("frame_idx")]
    # Split the groups into train and test sets
    # split 0.1 for validation (2025/08/04 fixed)
    # split 0.01 for validation (2025/08/05 fixed)
    train_groups, val_groups = train_test_split(
        grouped, test_size=0.01  # random_state=11
    )
    # Concatenate groups back into DataFrames
    train_df = pd.concat(train_groups).reset_index(drop=True)
    val_df = pd.concat(val_groups).reset_index(drop=True)

    create_yolo_annotations_with_mask(
        train_df,
        movie_path,
        tif_annotation_dir,
        train_annotations_dir,
        augment=True,
        target_size=100,
    )
    create_yolo_annotations_with_mask(
        val_df, movie_path, tif_annotation_dir, val_annotations_dir, augment=False
    )

    create_yolo_pose_yaml(train_annotations_dir, val_annotations_dir, yaml_path)

    print(
        f"Training annotations saved in {train_annotations_dir}, validation annotations in {val_annotations_dir}, and YAML configuration saved at {yaml_path}."
    )


def train_overfits(arguments):
    from ultralytics import YOLO
    from ultralytics import settings

    unique_name = arguments.unique_name
    load_model_path = arguments.load_model_path
    gpu1 = arguments.gpu1
    gpu2 = arguments.gpu2

    yaml_path = f"../annotations/overfittings/yamls/{unique_name}.yaml"
    settings.update({"datasets_dir": "/cellpose/scripts"})
    if load_model_path is None:
        model = YOLO("./runs/pose/train/weights/best.pt")
    else:
        model = YOLO(load_model_path)

    model.train(
        data=yaml_path,
        epochs=15,
        batch=20,
        device=[int(gpu1), int(gpu2)],
        project="../annotations/overfittings/overfits_weights",
        exist_ok=True,
        name=f"{unique_name}",
    )
    model.export(format="onnx")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--unique_name", type=str)
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--gpu1", type=int)
    parser.add_argument("--gpu2", type=int)
    arguments = parser.parse_args()

    create_masktif(arguments)
    create_yolo(arguments)
    train_overfits(arguments)


if __name__ == "__main__":
    main()
