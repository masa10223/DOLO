from tqdm import tqdm
from natsort import natsorted
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functions import calculate_value_funcs


def select_frames(thr_frame=5000):
    control_matrix_files = []
    for path in natsorted(glob.glob("./csvs/trajectory/C*")):
        print(f"Selecting frames from {path}")
        tmp_df = pd.read_csv(path)
        total_frames = tmp_df["Frame"].max() + 1
        if total_frames > thr_frame:
            print(f"Selected frames above threshold from {path}")
            control_matrix_files.append(path)

    mutant_matrix_files = []
    for path in natsorted(glob.glob("./csvs/trajectory/D*")):
        print(f"Selecting frames from {path}")
        tmp_df = pd.read_csv(path)
        total_frames = tmp_df["Frame"].max() + 1
        if total_frames > thr_frame:
            print(f"Selected frames above threshold from {path}")
            mutant_matrix_files.append(path)

    return control_matrix_files, mutant_matrix_files


def summarize_reinforce_learn_controls_mutants(
    control_matrix_files,
    mutant_matrix_files,
    contact_radius=15,
    angle_threshold=35,
    speed_threshold=0.3,
    frame_interval=5,
):
    mutants_results_dfs = pd.DataFrame()
    for path in tqdm(natsorted(mutant_matrix_files)):
        print(path)
        _, results_dataframe = calculate_value_funcs(
            path,
            contact_radius=contact_radius,
            angle_threshold=angle_threshold,
            speed_threshold=speed_threshold,
            frame_interval=frame_interval,
        )
        mutants_results_dfs = pd.concat([mutants_results_dfs, results_dataframe])
        mutants_results_dfs.to_csv(
            "./csvs/reinforce_results/mutants_results_calculatingnow.csv"
        )

    mutants_results_dfs.columns = [
        "filename",
        "id",
        "non_contact_crawl",
        "non_contact_turn",
        "non_contact_pause",
        "contact_crawl",
        "contact_turn",
        "contact_pause",
    ]
    mutants_results_dfs["group"] = "mutant"
    mutants_results_dfs.to_csv("./csvs/reinforce_results/mutants_results.csv")

    control_results_dfs = pd.DataFrame()
    for path in tqdm(natsorted(control_matrix_files)):
        _, results_dataframe = calculate_value_funcs(path)
        control_results_dfs = pd.concat([control_results_dfs, results_dataframe])
        control_results_dfs.to_csv(
            "./csvs/reinforce_results/control_results_calculatingnow.csv"
        )

    control_results_dfs.columns = [
        "filename",
        "id",
        "non_contact_crawl",
        "non_contact_turn",
        "non_contact_pause",
        "contact_crawl",
        "contact_turn",
        "contact_pause",
    ]
    control_results_dfs["group"] = "control"
    control_results_dfs.to_csv("./csvs/reinforce_results/control_results.csv")

    return control_results_dfs, mutants_results_dfs


def plot_value_funcs_controls_mutants(
    control_results_dfs, mutants_results_dfs, filter=True, filename=None
):

    df = pd.concat([mutants_results_dfs, control_results_dfs])
    df_compare = df[df.columns[5:]]
    if filter:
        df_filtered = df_compare[
            (df_compare[["contact_crawl", "contact_turn", "contact_pause"]] != 0).any(
                axis=1
            )
        ]

        df_filtered["group"] = df_filtered["group"].map(
            {
                "control": "$white^{1118}$",
                "mutant": "$orco^2$ , $Gr63a^1$",
            }
        )
        df_compare_melt = df_filtered.melt(
            id_vars="group", var_name="contact_type", value_name="value"
        )

    else:
        df_compare["group"] = df_compare["group"].map(
            {
                "control": "$white^{1118}$",
                "mutant": "$orco^2$ , $Gr63a^1$",
            }
        )
        df_filtered = df_compare.copy()
        df_compare_melt = df_compare.melt(
            id_vars="group", var_name="contact_type", value_name="value"
        )

    plt.figure(figsize=(13, 10))
    grped_bplot = sns.catplot(
        x="contact_type",
        y="value",
        data=df_compare_melt,
        kind="box",
        hue="group",
        palette="coolwarm",
        legend=False,
        height=6,
        aspect=1.3,
    )
    plt.xticks([0, 1, 2], ["Crawl", "Turn", "Pause"], fontsize=15)
    plt.xlabel("", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.ylim([-1, 1])
    plt.savefig(f"./Fig_paper/{filename}value_compare_boxplot_{filename}.pdf")

    ## Mean Compare
    df_filtered_mean = df_filtered.groupby("group").mean()

    plt.figure(figsize=(24, 6))
    plt.subplot(121)
    sns.heatmap(
        df_filtered_mean,
        vmax=0.5,
        vmin=-0.5,
        annot=True,
        cmap="coolwarm",
        xticklabels=["Crawl", "Turn", "Pause"],
        annot_kws={"size": 25},
        linewidth=2,
        # linecolor = 'k',
        square=True,
    )
    plt.title("Mean Compare", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("")
    ## Median comapare
    df_filtered_median = df_filtered.groupby("group").median()

    plt.subplot(122)
    sns.heatmap(
        df_filtered_median,
        annot=True,
        vmax=0.5,
        vmin=-0.5,
        cmap="coolwarm",
        xticklabels=["Crawl", "Turn", "Pause"],
        annot_kws={"size": 25},
        linewidth=2,
        # linecolor = 'k',
        square=True,
    )
    plt.title("Median Compare", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("")
    plt.savefig(f"./Fig_paper/{filename}/value_compare_heatmap_{filename}.pdf")

    df_filtered_melt = df_filtered.melt(
        id_vars="group", var_name="contact_type", value_name="value"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_filtered_melt,
        x="value",
        y="contact_type",
        orient="y",
        errorbar=("se"),
        capsize=0.4,
        err_kws={"color": "k", "linewidth": 1.5},
        linewidth=0,
        palette="coolwarm_r",
        hue="group",
    )
    handler, label = ax.get_legend_handles_labels()
    ax.legend(
        handler,
        ["$orco^2$ , $Gr63a^1$", "$white^{1118}$"],
        fontsize=15,
        loc="upper left",
        bbox_to_anchor=(1.0, 1),
    )
    plt.yticks([0, 1, 2], ["Crawl", "Turn", "Pause"], fontsize=15)
    plt.ylabel("")
    plt.xticks(fontsize=10)
    plt.xlabel("Mean of Value Function", fontsize=16)
    plt.savefig(f"./Fig_paper/{filename}/value_compare_mean_barplot_se_{filename}.pdf")


if __name__ == "__main__":
    Thr_frame = 5500
    contact_radius = 25
    angle_threshold = 35
    speed_threshold = 0.3
    frame_interval = 5
    control_matrix_files, mutant_matrix_files = select_frames(thr_frame=Thr_frame)
    control_results_dfs, mutants_results_dfs = (
        summarize_reinforce_learn_controls_mutants(
            control_matrix_files,
            mutant_matrix_files,
            contact_radius=contact_radius,
            angle_threshold=angle_threshold,
            speed_threshold=speed_threshold,
            frame_interval=frame_interval,
        )
    )
    plot_value_funcs_controls_mutants(control_results_dfs, mutants_results_dfs)
