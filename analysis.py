#%%
from dotenv import load_dotenv
load_dotenv(".env.vastai")
#os.environ["LD_LIBRARY_PATH"] = "/root/miniconda3/envs/ml_env_vars/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"

import analysis
import numpy as np 
import os 
import importlib
import warnings
import random

warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', message='findfont: Font family')



importlib.reload(analysis.workflow)
importlib.reload(analysis.core)
importlib.reload(analysis.plotting)
importlib.reload(analysis.text_processing)
importlib.reload(analysis.misc) 
importlib.reload(analysis.plot_configs)
importlib.reload(analysis)

np.random.seed(42)
random.seed(42)

#%%main setup 

all_paths_dir = "main_results_data"
scaling_config = ("deepseek","aime25","32B") #config for scaling paths

paths = os.listdir(all_paths_dir)
paths = [os.path.join(all_paths_dir, path) for path in paths]
main_scale="32B"

focus_model_name = "deepseek"
focus_dataset_name = "aime25"

rank_by = "probability_skewness"

#%% 


if 1:
    save_dir = "results_output_main"
    os.makedirs(save_dir, exist_ok=True)

    analysis_workflow = analysis.AnalysisWorkflow(paths=paths, 
    results_save_dir=save_dir,scaling_config=scaling_config)

    SHOW=False
    SAVE=True


    RUN_CONFIG={
        "TOKEN_BUDGET_COUNT":False,
        "AGGREGATE":False,
        "ACROSS_DATASETS":False,
        "ACROSS_MODELS":False,
        "ACROSS_SCALES":False,
        "TOP_K_SAMPLES":False,
        "TASK_COMPARISON":False,
        "SCALING_ANALYSIS":True,

    }


    if RUN_CONFIG["TOKEN_BUDGET_COUNT"]:
        analysis_workflow.token_budget_plot(model_name=focus_model_name,dataset_name=focus_dataset_name,SHOW=SHOW,SAVE=SAVE)


    if RUN_CONFIG["AGGREGATE"]:

        analysis_workflow.aggregate_plots(SHOW=SHOW,SAVE=SAVE,model_name=focus_model_name,dataset_name=focus_dataset_name)

        #correlation_scores = analysis_workflow.correlation_scores(model_name=model_name,dataset_name=dataset_name,method="2D")


        correlation_results = analysis_workflow.plot_correlation_heatmap(correlation_type="pearson",metric_pair="score_prob",SHOW=SHOW,SAVE=SAVE,vmin=0,vmax=1,method="2D")

    if RUN_CONFIG["ACROSS_DATASETS"]:
        #saving not working at the moment

        for model in analysis_workflow.model_names:
            if focus_model_name not in model.lower():
                continue
            emergence_score_dist_across_datasets = analysis_workflow.emergence_score_dist_across_datasets(model_name=model,SHOW=SHOW,SAVE=SAVE)

            datasets_summary_stats = analysis_workflow.emergence_score_summary_stats_across_datasets(model_name=model,SAVE=SAVE,save_dir=save_dir)

    if RUN_CONFIG["ACROSS_MODELS"]:

        for dataset in analysis_workflow.dataset_names:
            if focus_dataset_name.lower() not in dataset.lower():

                continue
            emergence_score_dist_across_models = analysis_workflow.emergence_score_dist_across_models(dataset_name=dataset,SHOW=SHOW, SAVE=SAVE)
        

            models_summary_stats = analysis_workflow.emergence_score_summary_stats_across_models(dataset_name=dataset,SAVE=SAVE,save_dir=save_dir)
        

    if RUN_CONFIG["ACROSS_SCALES"]:

        
        out = analysis_workflow.plot_scaling_curves(SHOW=SHOW,SAVE=SAVE,SINGLE_PLOT=False)

        out = analysis_workflow.emergence_scores_dist_across_scales(SHOW=SHOW,SAVE=SAVE)

        scales_summary_stats = analysis_workflow.emergence_score_summary_stats_across_scales()

    if RUN_CONFIG["TOP_K_SAMPLES"]:
        #specific model to analyse

        task_orderings = analysis_workflow.get_emergence_score_task_orderings(dataset_name=focus_dataset_name,model_name=focus_model_name,rank_by=rank_by)

        top_k_samples = analysis_workflow.plot_top_k_max_emergence_samples(model_name=focus_model_name,dataset_name=focus_dataset_name,k=4,SHOW=SHOW,SAVE=SAVE,save_dir=save_dir,task_orderings=task_orderings)

        #all 
        focus_dataset_name="aime25"
        scaling_curve_data,top_k_tasks_all = analysis_workflow.plot_top_k_tasks_across_models(dataset_name=focus_dataset_name,k=4,SHOW=SHOW,SAVE=SAVE,save_dir=save_dir,rank_by=rank_by)

    if RUN_CONFIG["SCALING_ANALYSIS"]:

        #analysis_workflow.scaling_aggregate_plots(SHOW=SHOW,SAVE=SAVE)

        top_k_indices = [2,3,15,0]
        scaling_curve_data = analysis_workflow.plot_top_k_across_scales(top_k_indices=top_k_indices,SHOW=SHOW,SAVE=SAVE,save_dir=save_dir)

        breakpoint()


if 0: 
    #appendix
    all_paths_dir = "main_results_data"
    scaling_config = ("deepseek","aime25","32B") #config for scaling paths

    paths = os.listdir(all_paths_dir)
    paths = [os.path.join(all_paths_dir, path) for path in paths]
    main_scale="32B"

    save_dir = "results_output_appendix"
    os.makedirs(save_dir, exist_ok=True)
    model_name_placeholder = "deepseek"
    dataset_name_placeholder = "gpqa"

    analysis_workflow = analysis.AnalysisWorkflow(paths=paths, results_save_dir=save_dir,scaling_config=scaling_config)

    SHOW=False
    SAVE=True


    RUN_CONFIG={
        "TOKEN_BUDGET_COUNT":False,
        "AGGREGATE":True,
        "ACROSS_DATASETS":True,
        "ACROSS_MODELS":True,
        "ACROSS_SCALES":True,
        "TOP_K_SAMPLES":True,
        "TASK_COMPARISON":True,

    }
    
    if RUN_CONFIG["TOKEN_BUDGET_COUNT"]:
        for model in analysis_workflow.model_names:
            for dataset in analysis_workflow.dataset_names:
                analysis_workflow.token_budget_plot(model_name=model,dataset_name=dataset,SHOW=SHOW,SAVE=SAVE)

        print("Done - Appendix - Token Budget Counts")

    #aggregate analysis 
    if RUN_CONFIG["AGGREGATE"]:

        analysis_workflow.aggregate_plots(SHOW=SHOW,SAVE=SAVE)

        correlation_results_prob = analysis_workflow.plot_correlation_heatmap(correlation_type="pearson",metric_pair="score_prob",SHOW=SHOW,SAVE=SAVE,vmin=0,vmax=1,method="2D")
        correlation_results_entropy = analysis_workflow.plot_correlation_heatmap(correlation_type="pearson",metric_pair="score_entropy",SHOW=SHOW,SAVE=SAVE,vmin=0,vmax=1,method="2D")

        print("Done - Appendix - Aggregate Plots")

    if RUN_CONFIG["ACROSS_DATASETS"]:
        #saving not working at the moment

        breakthroughness_lims = [0,50]
        skewness_lims = [-4,4]

        for model in analysis_workflow.model_names:
            emergence_score_dist_across_datasets = analysis_workflow.emergence_score_dist_across_datasets(model_name=model,SHOW=SHOW,SAVE=SAVE,breakthroughness_lims=breakthroughness_lims,skewness_lims=skewness_lims)

            datasets_summary_stats = analysis_workflow.emergence_score_summary_stats_across_datasets(model_name=model,SAVE=SAVE,save_dir=save_dir)

        datasets_summary_stats = analysis_workflow.emergence_score_summary_stats_across_datasets(model_name=model_name_placeholder,SAVE=SAVE,save_dir=save_dir)

        print("Done - Appendix - Emergence Scores Dist Across Datasets")

    if RUN_CONFIG["ACROSS_MODELS"]:

        for dataset in analysis_workflow.dataset_names:
            emergence_score_dist_across_models = analysis_workflow.emergence_score_dist_across_models(dataset_name=dataset,SHOW=SHOW, SAVE=SAVE,breakthroughness_lims=breakthroughness_lims,skewness_lims=skewness_lims)

            models_summary_stats = analysis_workflow.emergence_score_summary_stats_across_models(dataset_name=dataset,SAVE=SAVE,save_dir=save_dir)
        

        models_summary_stats = analysis_workflow.emergence_score_summary_stats_across_models(dataset_name=dataset_name_placeholder,SAVE=SAVE,save_dir=save_dir)
        
        print("Done - Appendix - Emergence Scores Dist Across Models")

    if RUN_CONFIG["ACROSS_SCALES"]: #eventually we can broaden for scaling across all datasets (not just GPQA)

        breakthroughness_lims = [0,60]
        skewness_lims = [-4,4]
        
        out = analysis_workflow.plot_scaling_curves(SHOW=SHOW,SAVE=SAVE,SINGLE_PLOT=False)

        out = analysis_workflow.emergence_scores_dist_across_scales(SHOW=SHOW,SAVE=SAVE,breakthroughness_lims=breakthroughness_lims,skewness_lims=skewness_lims)

        scales_summary_stats = analysis_workflow.emergence_score_summary_stats_across_scales()

        print("Done - Appendix - Scaling Curves")

    if RUN_CONFIG["TOP_K_SAMPLES"]:
        #specific model to analyse



        for model in analysis_workflow.model_names:
            for dataset in analysis_workflow.dataset_names:
                top_k_samples = analysis_workflow.plot_top_k_max_emergence_samples(model_name=model,dataset_name=dataset,k=4,SHOW=SHOW,SAVE=SAVE,save_dir=save_dir,rank_by=rank_by)

        print("Done - Appendix - Top-K Samples")


    if RUN_CONFIG["TASK_COMPARISON"]:

        for dataset in analysis_workflow.dataset_names:
            out = analysis_workflow.task_comparison(dataset_name=dataset,SAVE=SAVE,SHOW=SHOW,rank_by=rank_by)

        print("Done - Appendix - Task Comparison")

