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

#%% setup 

all_paths_dir = "results_data/Round_1_21_09"
scaling_config = ("deepseek","gpqa","32B") #config for scaling paths

paths = os.listdir(all_paths_dir)
paths = [os.path.join(all_paths_dir, path) for path in paths]
main_scale="32B"


#%% main 
import os 
import importlib
import utils


importlib.reload(utils.analysis_utils)
importlib.reload(utils.plot_configs)
importlib.reload(utils)

save_dir = "/Users/iyngkarrankumar/Documents/Edinburgh MScR/writing/thesis/figures/main"
analysis_workflow = utils.analysis_utils.AnalysisWorkflow(paths=paths, results_save_dir=save_dir,scaling_config=scaling_config)
os.makedirs(save_dir, exist_ok=True)


import warnings
warnings.filterwarnings("ignore")

focus_model_name = "deepseek"
focus_dataset_name = "gpqa"


#aggregate analysis 
if 1: 

    analysis_workflow.aggregate_plots(SHOW=True,SAVE=True,model_name=focus_model_name,dataset_name=focus_dataset_name)

    #correlation_scores = analysis_workflow.correlation_scores(model_name=model_name,dataset_name=dataset_name,method="2D")


    correlation_results = analysis_workflow.plot_correlation_heatmap(correlation_type="pearson",metric_pair="score_prob",SHOW=True,SAVE=True,vmin=0,vmax=1,method="2D")

if 1: 
    #saving not working at the moment

    breakthroughness_lims = [0,50]
    skewness_lims = [-4,4]

    for model in analysis_workflow.model_names:
        if focus_model_name not in model.lower():
            continue
        emergence_score_dist_across_datasets = analysis_workflow.emergence_score_dist_across_datasets(model_name=model,SHOW=True,SAVE=True,breakthroughness_lims=breakthroughness_lims,skewness_lims=skewness_lims)

    model_name = focus_model_name
    datasets_summary_stats = analysis_workflow.emergence_score_summary_stats_across_datasets(model_name=focus_model_name)


    for dataset in analysis_workflow.dataset_names:
        if focus_dataset_name not in dataset.lower():
            continue
        emergence_score_dist_across_models = analysis_workflow.emergence_score_dist_across_models(dataset_name=dataset,SHOW=True, SAVE=True,breakthroughness_lims=breakthroughness_lims,skewness_lims=skewness_lims)
    

    models_summary_stats = analysis_workflow.emergence_score_summary_stats_across_models(dataset_name=focus_dataset_name)
    
if 1:

    breakthroughness_lims = [0,60]
    skewness_lims = [-4,4]
    
    out = analysis_workflow.plot_scaling_curves(SHOW=True,SAVE=True,SINGLE_PLOT=False)

    out = analysis_workflow.emergence_scores_dist_across_scales(SHOW=True,SAVE=True,breakthroughness_lims=breakthroughness_lims,skewness_lims=skewness_lims)

    scales_summary_stats = analysis_workflow.emergence_score_summary_stats_across_scales()

if 1: 
    #specific model to analyse

    top_k_samples = analysis_workflow.get_top_k_max_emergence_samples(model_name=focus_model_name,dataset_name=focus_dataset_name,k=4,plot=True,save_plots=True)


if 1: 

    rankings = analysis_workflow.get_emergence_score_rankings(dataset_name=focus_dataset_name,model_name=focus_model_name)

    out = analysis_workflow.task_comparison(dataset_name=focus_dataset_name,SAVE=True)


#%% appendix 

save_dir = "/Users/iyngkarrankumar/Documents/Edinburgh MScR/writing/thesis/figures/appendix"
os.makedirs(save_dir, exist_ok=True)

if 1:

    analysis_workflow.aggregate_plots(SHOW=True,SAVE=False,model_name=model_name,dataset_name=dataset_name)

    correlation_scores = analysis_workflow.correlation_scores(model_name=model_name,dataset_name=dataset_name,method="2D")

    correlation_results = analysis_workflow.plot_correlation_heatmap(correlation_type="pearson",metric_pair="score_prob",SHOW=True,SAVE=False,vmin=0,vmax=1,method="2D")