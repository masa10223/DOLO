import time
from functions import process_video_to_gif_with_angles

# from functions_test import process_video_to_gif_with_angles_and_tracking
import argparse
import pytz
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Input questionnare name and column range"
    )
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--output_gif_path", type=str)
    parser.add_argument("--output_csv_path", type=str)
    parser.add_argument(
        "--model_path",
        type=str,
        default="./runs/pose/train/weights/best.pt",
    )
    parser.add_argument("--max_id", type=int, default=10)
    parser.add_argument("--conf", type=float, default=1e-4)
    parser.add_argument("--max_missing_frames", type=int, default=30)
    arguments = parser.parse_args()

    video_path = arguments.video_path
    output_gif_path = arguments.output_gif_path
    output_csv_path = arguments.output_csv_path
    max_consistent_id = arguments.max_id
    confidence = arguments.conf
    max_missing_frames = arguments.max_missing_frames
    model_path = "./runs/pose/train20241204/weights/best.pt"

    start_now = datetime.now(pytz.timezone("Asia/Tokyo"))
    start = time.time()
    print(
        "Creating Gif...: {}".format(start_now.strftime("%Y-%m-%d %H:%M:%S")),
        flush=True,
    )

    # Convert video to annotated GIF with angles
    process_video_to_gif_with_angles(
        video_path=video_path,
        output_gif_path=output_gif_path,
        model_path=model_path,
        confidence=confidence,
        max_consistent_ids=max_consistent_id,
        output_csv_path=output_csv_path,
        max_missing_frames=max_missing_frames,
    )

    # process_video_to_gif_with_angles_and_tracking(
    #     video_path=video_path,
    #     output_gif_path=output_gif_path,
    #     # model_path = model_path,
    #     confidence=confidence,
    #     max_consistent_id=max_consistent_id,
    #     output_csv_path=output_csv_path,
    # )

    print("Done in {} sec".format(time.time() - start), flush=True)


if __name__ == "__main__":
    main()
