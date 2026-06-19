import time
# from functions import process_video_to_gif_with_angles, process_video_to_gif_with_manual
#from functions_gpt import process_video_to_gif_with_angles
from functions_deepsort import process_video_to_gif_with_angles
import argparse
import pytz
from datetime import datetime
import json
import seaborn as sns
import os

sns.set()

def main():
    parser = argparse.ArgumentParser(
        description="Input questionnare name and column range"
    )
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--output_gif_path", type=str)
    parser.add_argument("--output_mov_path", type=str)
    parser.add_argument("--output_csv_path", type=str)
    parser.add_argument(
        "--model_path",
        type=str,
        default="./runs/pose/train/weights/best.pt",
    )
    parser.add_argument("--max_id", type=int, default=10)
    parser.add_argument("--conf", type=float, default=1e-3)
    parser.add_argument("--iou_thr", type=float, default=0.45)
    parser.add_argument("--max_missing_frames", type=int, default=15)
    parser.add_argument("--manual_assignments_file", type=str, default=None,
                        help="Path to a JSON file containing manual assignments")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--dist_thresh", type=int, default=30)
    parser.add_argument("--head_tail_jump_thresh", type=int, default=50)
    parser.add_argument("--overlap_thresh", type=int, default=5)
    arguments = parser.parse_args()

    video_path = arguments.video_path
    output_gif_path = arguments.output_gif_path
    output_mov_path = arguments.output_mov_path
    output_csv_path = arguments.output_csv_path
    max_consistent_id = arguments.max_id
    confidence = arguments.conf
    iou_thr = arguments.iou_thr
    max_missing_frames = arguments.max_missing_frames
    start_frame = arguments.start_frame
    end_frame = arguments.end_frame
    model_path = arguments.model_path
    dist_thresh = arguments.dist_thresh
    head_tail_jump_thresh = arguments.head_tail_jump_thresh
    overlap_thresh = arguments.overlap_thresh
    manual_assignments = None
    
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found or path is empty: {video_path}")
    
    if arguments.manual_assignments_file:
        try:
            with open(arguments.manual_assignments_file, "r") as f:
                data = json.load(f)
                # json.loadした直後
            manual_assignments = {
                int(frame_str): {int(cid_str): box_idx
                                for cid_str, box_idx in assignments.items()}
                for frame_str, assignments in data.items()
            }
        except Exception as e:
            print(f"手動割り当てファイルの読み込みに失敗: {e}")

    start_now = datetime.now(pytz.timezone("Asia/Tokyo"))
    start = time.time()
    print(
        "Creating Gif...: {}".format(start_now.strftime("%Y-%m-%d %H:%M:%S")),
        flush=True,
    )

    # Convert video to annotated GIF with angles
    # process_video_to_gif_with_angles(
    #     video_path=video_path,
    #     output_gif_path=output_gif_path,
    #     model_path=model_path,
    #     confidence=confidence,
    #     max_consistent_ids=max_consistent_id,
    #     output_csv_path=output_csv_path,
    #     max_missing_frames=max_missing_frames,
    #     manual_assignments=manual_assignments,
    #     start_frame=start_frame,
    #     end_frame=end_frame,
    # )

    process_video_to_gif_with_angles(
        video_path=video_path,
        output_gif_path=output_gif_path,
        output_mov_path=output_mov_path,
        model_path=model_path,
        output_csv_path=output_csv_path,
        conf_thres=confidence,
        iou_thres=iou_thr,
        frame_skip=1,
        device="cuda:3",
        start_frame=start_frame,
        end_frame=end_frame,
        max_age=max_missing_frames,
        max_ids=max_consistent_id,   
        n_init=2, ## how many frames to detect trajectories
        dist_thresh=dist_thresh,
        head_tail_jump_thresh=head_tail_jump_thresh,  ### NEW param
        overlap_thresh=overlap_thresh           ## how close (head, middle, tail) can be before we consider them “identical.”
        )


    print("Done in {} sec".format(time.time() - start), flush=True)


if __name__ == "__main__":
    main()
