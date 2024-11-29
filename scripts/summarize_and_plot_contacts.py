import glob
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import euclidean
from natsort import natsorted
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import matplotlib.ticker as ptick 

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

 # 有意差を示す関数

def add_stat_annotation(ax, x1, x2, y, p_val, h ):
    """有意差のアノテーションを追加"""  # 線の高さ
    color = 'black'
    ax.plot([x1, x1, x2, x2], [y, y + h * 1.2 , y + h * 1.2, y], color=color)
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "n.s."
    ax.text((x1 + x2) * 0.5, y + h , text, ha='center', va='bottom', color=color, fontsize = 20)

def summarize_contacts_time_controls_and_mutants(control_matrix_files, mutant_matrix_files):
    # 接触行列のデータを格納するリスト
    contact_data_list = []
    # 各接触行列を読み込み、データを抽出
    for idx, file in enumerate(control_matrix_files):
        ## 最大フレームの取得
        filename = (os.path.splitext(os.path.basename(file))[0][15:])
        tmp_df = pd.read_csv('./csvs/trajectory/{}.csv'.format(filename))
        total_frames = tmp_df['Frame'].max() + 1

        df = pd.read_csv(file)
        df['duration'] = df['end_frame'] - df['start_frame'] + 1
        # 2. 個体ペアごとに duration を合計
        contact_time_df = df.groupby(['id1', 'id2'])['duration'].sum().reset_index()
        # 3. カラム名を変更
        contact_time_df.rename(columns={
            'id_min': 'id1',
            'id_max': 'id2',
            'duration': 'Total_Contact_time'
        }, inplace=True)
        for _, row in contact_time_df.iterrows():
            contact_rate = row['Total_Contact_time'] / total_frames
            contact_data_list.append({
                'Group': 'control',
                'Matrix_ID': idx,
                'ID1': row['id1'],
                'ID2': row['id2'],
                'Contact_time_rate': contact_rate
            })

    for idx, file in enumerate(mutant_matrix_files):
        ## 最大フレームの取得
        filename = (os.path.splitext(os.path.basename(file))[0][15:])
        tmp_df = pd.read_csv('./csvs/trajectory/{}.csv'.format(filename))
        total_frames = tmp_df['Frame'].max() + 1
        
        df = pd.read_csv(file)
        df['duration'] = df['end_frame'] - df['start_frame'] + 1
        # 2. 個体ペアごとに duration を合計
        contact_time_df = df.groupby(['id1', 'id2'])['duration'].sum().reset_index()
        # 3. カラム名を変更
        contact_time_df.rename(columns={
            'id_min': 'id1',
            'id_max': 'id2',
            'duration': 'Total_Contact_time'
        }, inplace=True)
        
        for _, row in contact_time_df.iterrows():
            contact_rate = row['Total_Contact_time'] / total_frames
            contact_data_list.append({
                'Group': 'mutant',
                'Matrix_ID': idx,
                'ID1': row['id1'],
                'ID2': row['id2'],
                'Contact_time_rate': contact_rate
            })

    # データフレームを作成
    contact_time_rate_df = pd.DataFrame(contact_data_list)
    
    return contact_time_rate_df

def plot_contact_time_rate_compare(contact_time_rate_df):
    # グループごとの平均接触率を計算
    group_contact_rates = contact_time_rate_df.groupby(['Group', 'Matrix_ID'])['Contact_time_rate'].mean().reset_index()
    # コントロール群と変異体群の接触率データ
    control_rates = group_contact_rates[group_contact_rates['Group'] == 'control']['Contact_time_rate']
    mutant_rates = group_contact_rates[group_contact_rates['Group'] == 'mutant']['Contact_time_rate']
    # 正規性の検定
    stat_c, p_c = shapiro(control_rates)
    stat_m, p_m = shapiro(mutant_rates)
    print('Contact_time_rate: Control group normality p-value:', p_c)
    print('Contact_time_rate: Mutant group normality p-value:', p_m)
    # 分散の等質性の検定
    stat_levene, p_levene = levene(control_rates, mutant_rates)
    print('Contact_time_rate: Levene test p-value:', p_levene)
    if p_c > 0.05 and p_m > 0.05 and p_levene > 0.05:
        # t検定を実施
        stat, p_value = ttest_ind(control_rates, mutant_rates)
        print('Contact_time_rate: t-test p-value:', p_value)
    else:
        # マン・ホイットニーのU検定を実施
        stat, p_value = mannwhitneyu(control_rates, mutant_rates, alternative='two-sided')
        print('Contact_time_rate: Mann-Whitney U test p-value:', p_value)


    fig, ax = plt.subplots(figsize=(7, 6))
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2,-2))

    sns.boxplot(x='Group',
                y='Contact_time_rate', 
                data=group_contact_rates,
                linecolor="k",
                linewidth=2,
                palette = 'coolwarm')
    # フォントサイズやラベルの設定
    ax.set_title('Average Contact Time by Group', fontsize=20)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Contact Time per Frame [sec]", fontsize=20)
    labels = ['$\mathit{white^{1118}}$', '$\mathit{orco^2}$ , $\mathit{Gr63a^1}$']
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    labels_get = ax.get_xticklabels()
    for lbl in labels_get:
        lbl.set_style('italic')

    # 有意差のアノテーション追加
    y_max = group_contact_rates['Contact_time_rate'].max()
    ymin = group_contact_rates['Contact_time_rate'].min()
    h = y_max - ymin
    add_stat_annotation(ax, 0, 1, y_max + h / 30, p_value, h / 10 )
    plt.savefig('./Fig_paper/contact_time_compare.pdf')
    #plt.show()

def summarize_contacts_count_controls_and_mutants(control_matrix_files, mutant_matrix_files):
    # 接触行列のデータを格納するリスト
    contact_data_list = []
    # 各接触行列を読み込み、データを抽出
    for idx, file in enumerate(control_matrix_files):
        ## 最大フレームの取得
        filename = (os.path.splitext(os.path.basename(file))[0][15:])
        tmp_df = pd.read_csv('./csvs/trajectory/{}.csv'.format(filename))
        total_frames = tmp_df['Frame'].max() + 1

        df = pd.read_csv(file)
        df['duration'] = df['end_frame'] - df['start_frame'] + 1
        # 2. 個体ペアごとに duration を合計
        contact_time_df = df.groupby(['id1', 'id2'])['duration'].size().reset_index()
        # 3. カラム名を変更
        contact_time_df.rename(columns={
            'id_min': 'id1',
            'id_max': 'id2',
            'duration': 'Total_Contact_counts'
        }, inplace=True)
        for _, row in contact_time_df.iterrows():
            contact_rate = row['Total_Contact_counts'] / total_frames
            contact_data_list.append({
                'Group': 'control',
                'Matrix_ID': idx,
                'ID1': row['id1'],
                'ID2': row['id2'],
                'Contact_count_rate': contact_rate
            })

    for idx, file in enumerate(mutant_matrix_files):
        ## 最大フレームの取得
        filename = (os.path.splitext(os.path.basename(file))[0][15:])
        tmp_df = pd.read_csv('./csvs/trajectory/{}.csv'.format(filename))
        total_frames = tmp_df['Frame'].max() + 1
        
        df = pd.read_csv(file)
        df['duration'] = df['end_frame'] - df['start_frame'] + 1
        # 2. 個体ペアごとに duration を合計
        contact_time_df = df.groupby(['id1', 'id2'])['duration'].size().reset_index()
        # 3. カラム名を変更
        contact_time_df.rename(columns={
            'id_min': 'id1',
            'id_max': 'id2',
            'duration': 'Total_Contact_counts'
        }, inplace=True)
        
        for _, row in contact_time_df.iterrows():
            contact_rate = row['Total_Contact_counts'] / total_frames
            contact_data_list.append({
                'Group': 'mutant',
                'Matrix_ID': idx,
                'ID1': row['id1'],
                'ID2': row['id2'],
                'Contact_count_rate': contact_rate
            })

    # データフレームを作成
    contact_count_rate_df = pd.DataFrame(contact_data_list)
    
    return contact_count_rate_df

def plot_contact_count_compare(contact_count_rate_df):
    # グループごとの平均接触率を計算
    group_contact_rates = contact_count_rate_df.groupby(['Group', 'Matrix_ID'])['Contact_count_rate'].mean().reset_index()
    # コントロール群と変異体群の接触率データ
    control_rates = group_contact_rates[group_contact_rates['Group'] == 'control']['Contact_count_rate']
    mutant_rates = group_contact_rates[group_contact_rates['Group'] == 'mutant']['Contact_count_rate']

    # 正規性の検定
    stat_c, p_c = shapiro(control_rates)
    stat_m, p_m = shapiro(mutant_rates)
    print('Cotact counts: Control group normality p-value:', p_c)
    print('Cotact counts: Mutant group normality p-value:', p_m)

    # 分散の等質性の検定
    stat_levene, p_levene = levene(control_rates, mutant_rates)
    print('Cotact counts: Levene test p-value:', p_levene)


    if p_c > 0.05 and p_m > 0.05 and p_levene > 0.05:
        # t検定を実施
        stat, p_value = ttest_ind(control_rates, mutant_rates)
        print('Cotact counts: t-test p-value:', p_value)
    else:
        # マン・ホイットニーのU検定を実施
        stat, p_value = mannwhitneyu(control_rates, mutant_rates, alternative='two-sided')
        print('Cotact counts: Mann-Whitney U test p-value:', p_value)



    fig, ax = plt.subplots(figsize=(7, 6))
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2,-2))


    sns.boxplot(x='Group',
                y='Contact_count_rate', 
                data=group_contact_rates,
                linecolor="k",
                linewidth=2,
                palette = 'coolwarm')
    # フォントサイズやラベルの設定
    ax.set_title('Average Contact Counts by Group', fontsize=20)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Contact Counts per Frame", fontsize=20)
    labels = ['$\mathit{white^{1118}}$', '$\mathit{orco^2}$ , $\mathit{Gr63a^1}$']
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    labels_get = ax.get_xticklabels()
    for lbl in labels_get:
        lbl.set_style('italic')

    # 有意差のアノテーション追加
    y_max = group_contact_rates['Contact_count_rate'].max()
    ymin = group_contact_rates['Contact_count_rate'].min()
    h = y_max - ymin
    add_stat_annotation(ax, 0, 1, y_max + h / 30, p_value, h / 15 )
    plt.savefig('./Fig_paper/contact_count_compare.pdf')

def summarize_contacts_time_per_touch_controls_and_mutants(control_matrix_files, mutant_matrix_files):
    # 接触行列のデータを格納するリスト
    contact_data_list = []
    # 各接触行列を読み込み、データを抽出
    for idx, file in enumerate(control_matrix_files):
        ## 最大フレームの取得
        filename = (os.path.splitext(os.path.basename(file))[0][15:])
        tmp_df = pd.read_csv('./csvs/trajectory/{}.csv'.format(filename))
        total_frames = tmp_df['Frame'].max() + 1

        df = pd.read_csv(file)
        df['duration'] = df['end_frame'] - df['start_frame'] + 1
        # 2. 個体ペアごとに duration を合計
        contact_time_df = df.groupby(['id1', 'id2'])['duration'].mean().reset_index()
        # 3. カラム名を変更
        contact_time_df.rename(columns={
            'id_min': 'id1',
            'id_max': 'id2',
            'duration': 'Total_Contact_time_per_touch'
        }, inplace=True)
        for _, row in contact_time_df.iterrows():
            contact_rate = row['Total_Contact_time_per_touch']
            contact_data_list.append({
                'Group': 'control',
                'Matrix_ID': idx,
                'ID1': row['id1'],
                'ID2': row['id2'],
                'Total_Contact_time_per_touch': contact_rate
            })

    for idx, file in enumerate(mutant_matrix_files):
        ## 最大フレームの取得
        filename = (os.path.splitext(os.path.basename(file))[0][15:])
        tmp_df = pd.read_csv('./csvs/trajectory/{}.csv'.format(filename))
        total_frames = tmp_df['Frame'].max() + 1
        
        df = pd.read_csv(file)
        df['duration'] = df['end_frame'] - df['start_frame'] + 1
        # 2. 個体ペアごとに duration を合計
        contact_time_df = df.groupby(['id1', 'id2'])['duration'].mean().reset_index()
        # 3. カラム名を変更
        contact_time_df.rename(columns={
            'id_min': 'id1',
            'id_max': 'id2',
            'duration': 'Total_Contact_time_per_touch'
        }, inplace=True)
        
        for _, row in contact_time_df.iterrows():
            contact_rate = row['Total_Contact_time_per_touch']
            contact_data_list.append({
                'Group': 'mutant',
                'Matrix_ID': idx,
                'ID1': row['id1'],
                'ID2': row['id2'],
                'Total_Contact_time_per_touch': contact_rate
            })

    # データフレームを作成
    contact_time_per_touch_df = pd.DataFrame(contact_data_list)
    
    return contact_time_per_touch_df

def plot_contact_time_per_touch_compare(contact_time_per_touch_df):
    # グループごとの平均接触率を計算
    group_contact_rates = contact_time_per_touch_df.groupby(['Group', 'Matrix_ID'])['Total_Contact_time_per_touch'].mean().reset_index()
    # コントロール群と変異体群の接触率データ
    control_rates = group_contact_rates[group_contact_rates['Group'] == 'control']['Total_Contact_time_per_touch']
    mutant_rates = group_contact_rates[group_contact_rates['Group'] == 'mutant']['Total_Contact_time_per_touch']

    # 正規性の検定
    stat_c, p_c = shapiro(control_rates)
    stat_m, p_m = shapiro(mutant_rates)
    print('Contact time per touch: Control group normality p-value:', p_c)
    print('Contact time per touch: Mutant group normality p-value:', p_m)

    # 分散の等質性の検定
    stat_levene, p_levene = levene(control_rates, mutant_rates)
    print('Contact time per touch: Levene test p-value:', p_levene)


    if p_c > 0.05 and p_m > 0.05 and p_levene > 0.05:
        # t検定を実施
        stat, p_value = ttest_ind(control_rates, mutant_rates)
        print('Contact time per touch: t-test p-value:', p_value)
    else:
        # マン・ホイットニーのU検定を実施
        stat, p_value = mannwhitneyu(control_rates, mutant_rates, alternative='two-sided')
        print('Contact time per touch: Mann-Whitney U test p-value:', p_value)



    fig, ax = plt.subplots(figsize=(7, 6))
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
    #ax.ticklabel_format(style="sci", axis="y", scilimits=(-2,-2))


    sns.boxplot(x='Group',
                y='Total_Contact_time_per_touch', 
                data=group_contact_rates,
                linecolor="k",
                linewidth=2,
                palette = 'coolwarm')
    # フォントサイズやラベルの設定
    ax.set_title('Average Contact Times per Touch by Group', fontsize=20)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Contact Times per Touch [sec]", fontsize=20)

    import matplotlib.font_manager as fm
    ax.set_xticks([0, 1])
    labels = ['$\mathit{white^{1118}}$', '$\mathit{orco^2}$ , $\mathit{Gr63a^1}$']
    ax.set_xticklabels(labels, fontsize=15,)
    ax.tick_params(axis='y', labelsize=15)

    # 有意差のアノテーション追加
    y_max = group_contact_rates['Total_Contact_time_per_touch'].max()
    ymin = group_contact_rates['Total_Contact_time_per_touch'].min()
    h = y_max - ymin
    add_stat_annotation(ax, 0, 1, y_max + h / 30, p_value, h / 15 )
    plt.savefig('./Fig_paper/contact_time_per_touch_compare.pdf')


if __name__ == "__main__":
    import matplotlib
    # フォントの設定を Arial に変更
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
    control_matrix_files, mutant_matrix_files = select_frames_of_contact_events(thr_frame= 5500)
    contact_time_rate_df = summarize_contacts_time_controls_and_mutants(control_matrix_files, mutant_matrix_files)
    plot_contact_time_rate_compare(contact_time_rate_df)
    contact_count_rate_df = summarize_contacts_count_controls_and_mutants(control_matrix_files, mutant_matrix_files)
    plot_contact_count_compare(contact_count_rate_df)
    contact_time_per_touch_df = summarize_contacts_time_per_touch_controls_and_mutants(control_matrix_files, mutant_matrix_files)
    plot_contact_time_per_touch_compare(contact_time_per_touch_df)