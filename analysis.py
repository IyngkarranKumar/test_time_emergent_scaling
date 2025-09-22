#%%
import importlib, random, os, sys
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import time
import webbrowser

import utils

from collections import Counter
from transformers import AutoTokenizer


np.random.seed(42)
random.seed(42)


#%% full workflow 
import os 
import importlib
import utils


importlib.reload(utils)
importlib.reload(utils.analysis_utils)
importlib.reload(utils.plot_configs)
all_paths_dir = "results_data/Round_1_21_09"
scaling_config = ("deepseek","gpqa") #config for scaling paths

paths = os.listdir(all_paths_dir)
paths = [os.path.join(all_paths_dir, path) for path in paths]
save_dir = "tmp_dir"
main_scale="32B"
analysis_workflow = utils.analysis_utils.AnalysisWorkflow(paths=paths, 
results_save_dir=save_dir,scaling_config=scaling_config)

#aggregaate analysis
if 1: 
    analysis_workflow.aggregate_plots(SHOW=True,SAVE=True,log_probs_aime=True)

    #correlation_results = analysis_workflow.plot_correlation_heatmap(correlation_type="pearson",metric_pair="score_prob",SHOW=True,SAVE=False)

if 0: 
    #saving not working at the moment

    breakthroughness_lims = [-20,50]
    skewness_lims = [-4,4]

    for model in analysis_workflow.model_names:
        continue
        emergence_score_dist_across_datasets = analysis_workflow.emergence_score_dist_across_datasets(model_name=model,SHOW=True,SAVE=False,breakthroughness_lims=breakthroughness_lims,skewness_lims=skewness_lims)

    model_name = "deepseek"
    datasets_summary_stats = analysis_workflow.emergence_score_summary_stats_across_datasets(model_name=model_name)


    for dataset in analysis_workflow.dataset_names:
        continue
        emergence_score_dist_across_models = analysis_workflow.emergence_score_dist_across_models(dataset_name=dataset,SHOW=True, SAVE=False,breakthroughness_lims=breakthroughness_lims,skewness_lims=skewness_lims)
    
    dataset_name = "gpqa"
    models_summary_stats = analysis_workflow.emergence_score_summary_stats_across_models(dataset_name=dataset_name)

if 0: 
    #specific model to analyse
    model_name = "deepseek"
    dataset_name = "gpqa"


    top_k_samples = analysis_workflow.get_top_k_max_emergence_samples(model_name=model_name,dataset_name=dataset_name,k=4,plot=True,save_plots=False)
     
if 0:
    
    #out = analysis_workflow.plot_scaling_curves(SHOW=False,SAVE=False,SINGLE_PLOT=False)

    #out = analysis_workflow.emergence_scores_dist_across_scales(SHOW=True,SAVE=False)

    scales_summary_stats = analysis_workflow.emergence_score_summary_stats_across_scales()

if 0:
    dataset_name = "gpqa"
    model_name = "deepseek"
    rankings = analysis_workflow.get_emergence_score_ordering(dataset_name=dataset_name,model_name=model_name)

if 0: 
    dataset_name = "gpqa"

    out = analysis_workflow.task_comparison(dataset_name=dataset_name)

#correlation_results = analysis_workflow.correlation_scores(model_name=model_name,dataset_name=dataset_name,method="2D")


#%% get max emergence samples for 6-pager 

importlib.reload(utils.analysis_utils)
importlib.reload(utils)
importlib.reload(utils.plot_configs)

path = ["results_data/DeepSeek-R1-Distill-Qwen-7B_gpqa_08-22_20-01-27"]
config,SAVE_DATA = utils.read_save_data(path[0],localisation_run=False)

analysis_workflow_debug = utils.analysis_utils.AnalysisWorkflow(paths=path,results_save_dir="dummy_dir",scaling_config=None)

breakthroughness_lims = [0,60]
skewness_lims = [-3,3]

gpqa_emergence_metric_data = analysis_workflow_debug.emergence_scores(model_name="deepseek",dataset_name="gpqa")

probs_skew_gpqa, negent_skew_gpqa = gpqa_emergence_metric_data["probs_skewness"], gpqa_emergence_metric_data["negent_skewness"]
abs_probs_break_gpqa, abs_negent_break_gpqa = gpqa_emergence_metric_data["abs_probs_breakthroughness"], gpqa_emergence_metric_data["abs_negent_breakthroughness"]


k = 10
m = 4

argmax_probs_skewness = utils.topk_nanmax_arg(probs_skew_gpqa, k)
argmax_negent_skewness = utils.topk_nanmax_arg(negent_skew_gpqa, k)
argmax_abs_probs_breakthroughness = utils.topk_nanmax_arg(abs_probs_break_gpqa, k)
argmax_abs_negent_breakthroughness = utils.topk_nanmax_arg(abs_negent_break_gpqa, k)

all_topk_idxs = argmax_probs_skewness + argmax_negent_skewness + argmax_abs_probs_breakthroughness + argmax_abs_negent_breakthroughness

top_m_recurring_with_counts = Counter(all_topk_idxs).most_common(m)
top_m_recurring = [idx for idx, _ in top_m_recurring_with_counts]
max_diffs = utils.get_max_diff(SAVE_DATA,config,sample_idxs=top_m_recurring)



#single plot 
sample_idxs = top_m_recurring
save_path="/Users/iyngkarrankumar/Documents/Edinburgh MScR/writing/6pager/figures/max_emergence_samples"
name="max_emergence_samples"
title="Samples with highest emergence scores"
utils.plot_data(config=config,SAVE_DATA=SAVE_DATA,metrics=["probs","entropy","score"],sample_idxs=sample_idxs,logy=False,score_average_type="mean",style_dict=utils.plot_configs.top_k_style,save_path=save_path,fig_title=title,name=name)

#%% check traces for path 

path = "results_data/Round_1_21_09/QwQ-32B_gpqa_09-19_12-20-19"
config,SAVE_DATA = utils.read_save_data(path)

utils.token_budget_text_view(config,SAVE_DATA,save=False)