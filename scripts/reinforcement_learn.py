from tqdm import tqdm
from natsort import natsorted
import glob
import pandas as pd

from functions import calculate_value_funcs

mutants_results_dfs = pd.DataFrame()
for path in tqdm(natsorted(glob.glob("./csvs/trajectory/D*.csv"))):
    _, results_dataframe = calculate_value_funcs(path)
    mutants_results_dfs = pd.concat([mutants_results_dfs, results_dataframe])
    mutants_results_dfs.to_csv("./csvs/reinforce_results/mutants_results_calculatingnow.csv")

mutants_results_dfs.to_csv("./csvs/reinforce_results/mutants_results.csv")

control_results_dfs = pd.DataFrame()
for path in tqdm(natsorted(glob.glob("./csvs/trajectory/C*.csv"))):
    _, results_dataframe = calculate_value_funcs(path)
    control_results_dfs = pd.concat([control_results_dfs, results_dataframe])
    control_results_dfs.to_csv("./csvs/reinforce_results/control_results_calculatingnow.csv")
control_results_dfs.to_csv("./csvs/reinforce_results/control_results.csv")
