#%%
import importlib, random, os, sys
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import time
import webbrowser

import utils
importlib.reload(utils.analysis_utils)
importlib.reload(utils)

np.random.seed(42)
random.seed(42)


#%%

importlib.reload(utils.analysis_utils)
importlib.reload(utils)

path = "results_data/DeepSeek-R1-Distill-Qwen-7B_gpqa_08-28_15-09-01_localisation_run"
config,SAVE_DATA = utils.read_save_data(path,localisation_run=True)

#%% plot 

importlib.reload(utils.analysis_utils)
importlib.reload(utils)

localisation_sample_idxs = list(SAVE_DATA.keys())

for sample_idx in localisation_sample_idxs:
    utils.plot_data(config=config,SAVE_DATA=SAVE_DATA[sample_idx],sample_idxs=[sample_idx],localisation_run=True)

#%% token budget view 

for sample_idx in localisation_sample_idxs:
    utils.token_budget_text_view(config,SAVE_DATA[sample_idx],save=False)

#%% metric tables

sample_idx = 1
mode = "full_view"

metric_types = ["answer_score", "answer_probability", "answer_entropy", "answer_ranking"]
token_budgets = config.token_budgets
mode="full_view"


gpqa_tables = {}
for metric_type in metric_types:
    table = utils.get_metric_table(SAVE_DATA[sample_idx],  metric_type=metric_type,mode=mode)
    gpqa_tables[metric_type] = table

#%%