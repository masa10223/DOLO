from summarize_and_plot_contacts import (
    select_frames_of_contact_events,
    summarize_contacts_time_controls_and_mutants,
    summarize_contacts_count_controls_and_mutants,
    summarize_contacts_time_per_touch_controls_and_mutants,
    plot_contact_time_rate_compare,
    plot_contact_count_compare,
    plot_contact_time_per_touch_compare,
)
from summarize_and_plot_interactions import (
    summarize_initeractions_controls_and_mutants,
    select_frames,
    distance_compute,
    plot_distance_compare,
)
from scripts.summarize_and_plot_reinforcement_learn import (
    summarize_reinforce_learn_controls_mutants,
    plot_value_funcs_controls_mutants,
)
from datetime import datetime
import pytz


def main():
    PIXEL_TO_CM = 0.006
    import argparse

    parser = argparse.ArgumentParser(
        description="Input questionnare name and column range"
    )
    parser.add_argument("--thr_frame", type=int, default=5500)
    parser.add_argument("--radius", type=int, default=15)
    parser.add_argument("--max_gap", type=int, default=5)
    parser.add_argument("--max_displacement", type=int, default=20)
    parser.add_argument("--angle_threshold", type=int, default=35)
    parser.add_argument("--speed_threshold", type=float, default=0.3)
    parser.add_argument("--frame_interval", type=int, default=5)
    arguments = parser.parse_args()

    Thr_frame = arguments.thr_frame
    Head_radius = arguments.radius
    max_gap = arguments.max_gap
    max_displacement = arguments.max_displacement
    angle_threshold = arguments.angle_threshold
    speed_threshold = arguments.speed_threshold
    frame_interval = arguments.frame_interval

    import matplotlib

    # フォントの設定を Arial に変更
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["font.sans-serif"] = ["Arial"]
    matplotlib.rcParams["mathtext.it"] = "Arial:italic"

    ### 抽出したデータフレームの確認
    start_now = datetime.now(pytz.timezone("Asia/Tokyo"))
    print(
        "Extracting frames...{}".format(start_now.strftime("%Y-%m-%d %H:%M:%S")),
        flush=True,
    )
    control_matrix_files, mutant_matrix_files = select_frames(thr_frame=Thr_frame)
    ##　相互作用の確認
    start_now = datetime.now(pytz.timezone("Asia/Tokyo"))
    print(
        "Checking interactions...{}".format(start_now.strftime("%Y-%m-%d %H:%M:%S")),
        flush=True,
    )
    summarize_initeractions_controls_and_mutants(
        control_matrix_files,
        mutant_matrix_files,
        head_radius=Head_radius,
        max_gap=max_gap,
        max_displacement=max_displacement,
    )
    results_combine, results_df, results_df_d = distance_compute(
        PIXEL_TO_CM=PIXEL_TO_CM
    )
    plot_distance_compare(results_combine, results_df, results_df_d)
    ## Contactの確認
    start_now = datetime.now(pytz.timezone("Asia/Tokyo"))
    print(
        "Checking contacts...{}".format(start_now.strftime("%Y-%m-%d %H:%M:%S")),
        flush=True,
    )
    contact_time_rate_df = summarize_contacts_time_controls_and_mutants(
        control_matrix_files, mutant_matrix_files
    )
    plot_contact_time_rate_compare(contact_time_rate_df)
    contact_count_rate_df = summarize_contacts_count_controls_and_mutants(
        control_matrix_files, mutant_matrix_files
    )
    plot_contact_count_compare(contact_count_rate_df)
    contact_time_per_touch_df = summarize_contacts_time_per_touch_controls_and_mutants(
        control_matrix_files, mutant_matrix_files
    )
    plot_contact_time_per_touch_compare(contact_time_per_touch_df)

    ## 逆強化学習
    start_now = datetime.now(pytz.timezone("Asia/Tokyo"))
    print(
        "Excecuting reinfrocement learning...{}".format(
            start_now.strftime("%Y-%m-%d %H:%M:%S")
        ),
        flush=True,
    )
    control_results_dfs, mutants_results_dfs = (
        summarize_reinforce_learn_controls_mutants(
            control_matrix_files,
            mutant_matrix_files,
            contact_radius=Head_radius,
            angle_threshold=angle_threshold,
            speed_threshold=speed_threshold,
            frame_interval=frame_interval,
        )
    )
    plot_value_funcs_controls_mutants(control_results_dfs, mutants_results_dfs)


if __name__ == "__main__":
    main()
