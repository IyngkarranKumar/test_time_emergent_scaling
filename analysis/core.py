
import os 
import pickle
import re
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from transformers import AutoTokenizer
import utils.workflow_utils as workflow_utils
from utils.scoring_utils import recompute_metrics


def table_to_3d_matrix(table_data):
    """Convert 2D table of tuples to 3D matrix (n_samples, tuple_length, n_budgets)"""
    n_samples = len(table_data)
    n_budgets = len(table_data.iloc[0])
    tuple_length = len(table_data.iloc[0,0])
    
    matrix = np.zeros((n_samples, tuple_length, n_budgets))
    
    for i in range(n_samples):
        for k in range(n_budgets):
            for j in range(tuple_length):
                matrix[i, j, k] = table_data.iloc[i,k][j]
    
    return matrix

def table_to_2d_matrix(table_data):
    n_samples = len(table_data)
    n_budgets = len(table_data.iloc[0])
    matrix = np.zeros((n_samples, n_budgets))
    for i in range(n_samples):
        for k in range(n_budgets):
            matrix[i, k] = table_data.iloc[i,k]
    return matrix

def extract_token_budget_and_batch(save_file):
        # Extract token_budget (last number before .pkl)
        token_budget_match = re.search(r'_(\d+)\.pkl$', save_file)
        token_budget = int(token_budget_match.group(1)) if token_budget_match else float('inf')
        # Extract first batch number (first number inside [ ... ])
        batch_match = re.search(r'\[([0-9,\s]+)\]', save_file)
        if batch_match:
            batch_numbers = [int(x) for x in batch_match.group(1).split(',')]
            first_batch = min(batch_numbers)
        else:
            first_batch = float('inf')
        return (token_budget, first_batch)


def get_model_size(path):
    """Extract model size in billions from path. Returns float or None."""
    # Get the model name part (before dataset and timestamp)
    model_name = path.split('/')[-1].split('_')[0]
    
    # Look for number followed by 'B' (case insensitive)
    match = re.search(r'(\d+(?:\.\d+)?)B', model_name, re.IGNORECASE)
    
    if match:
        return float(match.group(1))
    return None

def breakthrough_score(x,y,nan_threshold=100,diff_average="median_squared",square_numerator=False,legacy=False,DEBUG=False):

    n_samples = y.shape[0]
    n_diffs = len(x) - 1
    y_min = np.min(y, axis=1)
    y_max = np.max(y, axis=1)
    argmin = np.argmin(y, axis=1)
    argmax = np.argmax(y, axis=1)
    delta_y = np.diff(y, axis=1)
    sign_args = np.where(argmax - argmin > 0, 1, -1) #sign of the difference

    if legacy:
        avg_diff = np.sqrt(np.median(delta_y**2, axis=1)) #root median square
        len_norm_factor = 1
    elif diff_average=="median_squared":
        avg_diff = np.sqrt(np.median(delta_y**2, axis=1))
        len_norm_factor = 1 #median doesn't need a norm factor (?)
    elif diff_average=="mean":
        avg_diff = np.mean(delta_y, axis=1)
        len_norm_factor = 1
    elif diff_average=="mean_squared":
        avg_diff = np.sqrt(np.mean(delta_y**2, axis=1))
        len_norm_factor = 1/(n_diffs**0.5)
    elif diff_average=="mean_sqrt":
        avg_diff = np.power(np.mean(np.abs(delta_y)**0.5, axis=1), 2)
        len_norm_factor = 1/(n_diffs**2)
    else:
        raise ValueError(f"Invalid diff_average: {diff_average}")

    if legacy: 
        I_y = sign_args * (y_max - y_min)
    elif square_numerator:
        I_y = (sign_args * (y_max - y_min))**2
    else:
        I_y = sign_args * (y_max - y_min)


    breakthroughness = (len_norm_factor) * (I_y / avg_diff)
    
    breakthroughness = np.clip(breakthroughness, 0, nan_threshold) #bound breakthroughness in [0,100]
    breakthroughness = np.nan_to_num(breakthroughness, nan=0) #set score to 0 for flatliners

    if not DEBUG:
        return breakthroughness
    else:
        return breakthroughness, I_y, avg_diff

def differences_skew_score(y,skewness_type="adjusted_fisher_pearson",magnitude_weight=None):
        
        diffs = np.diff(y, axis=1)
        n_samples = diffs.shape[0]
        n_diffs = diffs.shape[1]
        assert n_diffs>2, f"Need array of atleast size 3 to comptue diff skewness"


        if skewness_type=="adjusted_fisher_pearson":
            differences_skew = stats.skew(diffs, axis=1,bias=False) #bias=False used adjusment 
        else:
            differences_skew = stats.skew(diffs, axis=1,bias=True) #bias=True used for regular fisher pearson skewness
        
        if magnitude_weight==True:
            max_abs_diff = np.max(np.abs(diffs), axis=1)
            mean_abs_diff = np.mean(np.abs(diffs), axis=1)
            weighting = max_abs_diff/mean_abs_diff
            weighting = max_abs_diff
        else:
            weighting = np.ones(n_samples)
        weighted_differences_skew = differences_skew * weighting
      
        return weighted_differences_skew


def read_save_data(data_dir):
    """
    Loads and merges all save files from data_dir, supporting:
        - token_budget only
        - token_budget + batch
        - token_budget + sample
    Returns loaded config and merged save data with structure:
        merged_save_data[token_budget][sample_or_batch_index][...]
    """
    import os
    import pickle
    import re

    with open(f"{data_dir}/config.pkl", 'rb') as f:
        config = pickle.load(f)

    save_files = [f for f in os.listdir(data_dir) if f.startswith('save_file_') and f.endswith('.pkl')]
    if not save_files:
        raise ValueError(f"No save_file_*.pkl files found in {data_dir}")

    # Helper: Extract token_budget, group_type ('batch' or 'samples'), group_start, group_end from filename
    def parse_budget_file_name(sf):
        # First try the simple pattern: save_file__budget_(\d+).pkl
        m = re.match(r"save_file__budget_(\d+)\.pkl$", sf)
        if m:
            token_budget = int(m.group(1))
            return (token_budget, None, None, None)
        
        # Then try the time-based pattern
        m = re.match(r"save_file__budget_(\d+)(?:_(samples|batch)_(\d+)-(\d+))?\.pkl$", sf)
        if m:
            token_budget = int(m.group(1))
            group_type = m.group(2)
            group_start = int(m.group(3)) if m.group(3) else None
            group_end = int(m.group(4)) if m.group(4) else None
            return (token_budget, group_type, group_start, group_end)
        else:
            return None

    # Sort save_files: by token_budget ASC, then group_type ('samples' < 'batch' < None), then group_start ASC
    def sort_key(sf):
        parsed = parse_budget_file_name(sf)
        if parsed is not None:
            token_budget, group_type, group_start, _ = parsed
            type_order = {'samples': 0, 'batch': 1, None: 2}
            return (token_budget, type_order.get(group_type, 3), group_start if group_start is not None else -1)
        else:
            # fallback if doesn't match: move to end
            return (float('inf'), 9, float('inf'))

    save_files = sorted(save_files, key=sort_key)

    merged_save_data = {}

    # Determine file style: supports both legacy and time-based
    # We treat as "time-based" if files include "_batch_" or "_samples_"
    time_based_saving = any(("_batch_" in f or "_samples_" in f) for f in save_files)

    for save_file in save_files:
        parse_result = parse_budget_file_name(save_file)
        save_path = os.path.join(data_dir, save_file)
        with open(save_path, 'rb') as f:
            save_data = pickle.load(f)

        if time_based_saving and parse_result and parse_result[1] is not None:
            # Parse the file info (only if it has group_type like 'samples' or 'batch')
            token_budget, group_type, group_start, group_end = parse_result
            assert token_budget in save_data, f"{save_file}: token_budget {token_budget} not in file data"

            save_data_for_budget = save_data[token_budget]
            save_data_no_metrics = {k: v for k, v in save_data_for_budget.items() 
                                   if k not in ['metrics', 'recomputed_metrics']}
            metrics_data = save_data_for_budget.get('metrics', {})
            recomputed_metrics_data = save_data_for_budget.get('recomputed_metrics', None)

            if token_budget not in merged_save_data:
                merged_save_data[token_budget] = {}

            # Allow for either "samples" (iteration index) or "batch" (batch index)
            # The index (sample or batch) comes from the keys of save_data_no_metrics
            for idx, dat in save_data_no_metrics.items():
                merged_save_data[token_budget][idx] = dat

            # Merge metrics, appending by metric_name
            if metrics_data:
                if 'metrics' not in merged_save_data[token_budget]:
                    merged_save_data[token_budget]['metrics'] = {}
                for metric_name, metric_data in metrics_data.items():
                    if metric_name not in merged_save_data[token_budget]['metrics']:
                        merged_save_data[token_budget]['metrics'][metric_name] = []
                    merged_save_data[token_budget]['metrics'][metric_name].extend(metric_data)

            # Handle recomputed_metrics (just copy, don't append)
            if recomputed_metrics_data is not None:
                merged_save_data[token_budget]['recomputed_metrics'] = recomputed_metrics_data

            # Ensure inner dict is sorted: int keys (sample/batch index) first, then 'metrics', then 'recomputed_metrics'
            merged = merged_save_data[token_budget]
            metric_val = merged.pop('metrics', None)
            recomputed_val = merged.pop('recomputed_metrics', None)
            merged = dict(sorted([(k, v) for k, v in merged.items() if isinstance(k, int)], key=lambda x: x[0]))
            if metric_val is not None:
                merged['metrics'] = metric_val
            if recomputed_val is not None:
                merged['recomputed_metrics'] = recomputed_val
            merged_save_data[token_budget] = merged

        else:
            # Legacy/nontimebased or simple format: e.g. just token_budget
            # save_data[token_budget]: {sample: {...}, ...}
            for token_budget in save_data:
                if token_budget not in merged_save_data:
                    merged_save_data[token_budget] = save_data[token_budget].copy()
                else:
                    merged_token_data = merged_save_data[token_budget]
                    for k, v in save_data[token_budget].items():
                        if k == 'metrics':
                            # Merge lists by metric name
                            if 'metrics' not in merged_token_data:
                                merged_token_data['metrics'] = v.copy()
                            else:
                                for metric_name, metric_data in v.items():
                                    if metric_name not in merged_token_data['metrics']:
                                        merged_token_data['metrics'][metric_name] = metric_data[:]
                                    else:
                                        merged_token_data['metrics'][metric_name].extend(metric_data)
                        elif k == 'recomputed_metrics':
                            # Just copy recomputed_metrics (don't merge/extend)
                            merged_token_data['recomputed_metrics'] = v
                        else:
                            merged_token_data[k] = v

    # Sort merged_save_data by token_budget, and ensure within each token_budget:
    # int keys sorted, metrics and recomputed_metrics last
    merged_save_data = dict(sorted(merged_save_data.items(), key=lambda x: x[0]))
    for token_budget in merged_save_data:
        d = merged_save_data[token_budget]
        metrics_val = d.pop('metrics', None)
        recomputed_val = d.pop('recomputed_metrics', None)
        d = dict(sorted([(k, v) for k, v in d.items() if isinstance(k, int)], key=lambda x: x[0]))
        if metrics_val is not None:
            d['metrics'] = metrics_val
        if recomputed_val is not None:
            d['recomputed_metrics'] = recomputed_val
        merged_save_data[token_budget] = d

    return config, merged_save_data


def get_budget_sample_completion_metrics(SAVE_DATA,config,average_across_completions=False):

    token_budgets_with_data = list(SAVE_DATA.keys())
    num_samples = config.num_samples
    if num_samples is None: 
        samples_keys = list(SAVE_DATA[token_budgets_with_data[0]].keys())
        if 'metrics' in samples_keys:
            samples_keys.remove('metrics')
        if 'recomputed_metrics' in samples_keys:
            samples_keys.remove('recomputed_metrics')
        num_samples = len(samples_keys)
    num_completions = config.num_return_sequences if hasattr(config, "num_return_sequences") else config.num_completions

    # For each token_budget, collect metrics for all sample-completion pairs
    n_token_budgets = len(token_budgets_with_data)
    n_samples = num_samples
    n_completions = num_completions

     # Initialize arrays: shape (n_token_budgets, n_samples, n_completions)

    scores = np.zeros((n_token_budgets, n_samples, n_completions))
    probs = np.zeros((n_token_budgets, n_samples, n_completions))
    entropy = np.zeros((n_token_budgets, n_samples, n_completions))
    rankings = np.zeros((n_token_budgets, n_samples, n_completions))

    recomputed_metrics_exist = 'recomputed_metrics' in SAVE_DATA[token_budgets_with_data[0]]
    if recomputed_metrics_exist:
        #print(f"Recomputed metrics found. Using recomputed metrics...")
        metric_key='recomputed_metrics'
    else:
        #print(f"No recomputed metrics found. Using metrics...")
        metric_key='metrics'
    for tb_idx, token_budget in enumerate(token_budgets_with_data):
        scores_metric = SAVE_DATA[token_budget][metric_key]['answer_score']
        probs_metric = SAVE_DATA[token_budget][metric_key]['answer_probability']
        entropy_metric = SAVE_DATA[token_budget][metric_key]['answer_entropy']
        rankings_metric = SAVE_DATA[token_budget][metric_key]['answer_ranking']
        
        for sample_idx in range(n_samples):
            for completion_idx in range(n_completions):
                metric_idx = sample_idx * n_completions + completion_idx
                try:
                    scores[tb_idx, sample_idx, completion_idx] = scores_metric[metric_idx][-1]
                    probs[tb_idx, sample_idx, completion_idx] = probs_metric[metric_idx][-1]
                    entropy[tb_idx, sample_idx, completion_idx] = entropy_metric[metric_idx][-1]
                    rankings[tb_idx, sample_idx, completion_idx] = rankings_metric[metric_idx][-1]
                except:
                    print(f"WARNING: Model: {config.model_name}, Dataset: {config.dataset_name}, Metric: {metric_key}, Token Budget: {token_budget}, Sample: {sample_idx}, Completion: {completion_idx} - No data found - set to 0 - WARNING")
                    scores[tb_idx, sample_idx, completion_idx] = 0
                    probs[tb_idx, sample_idx, completion_idx] = 0
                    entropy[tb_idx, sample_idx, completion_idx] = 0
                    rankings[tb_idx, sample_idx, completion_idx] = 0
    
    if average_across_completions:
        scores = np.mean(scores, axis=2)
        probs = np.mean(probs, axis=2)
        entropy = np.mean(entropy, axis=2)
        rankings = np.mean(rankings, axis=2)

    return scores, probs, entropy, rankings



def get_samplewise_breakthrough_and_skew(SAVE_DATA, config, breakthroughness_lims=[0,20],skewness_lims=[-3,3],legacy_scoring=False):
    """
    Computes and plots samplewise breakthroughness and skewness for probability and negative entropy.
    """
     
    token_budgets_with_data = list(SAVE_DATA.keys())
    num_samples = config.num_samples if hasattr(config, "num_samples") else config.num_samples
    if num_samples is None: 
        samples_keys = list(SAVE_DATA[token_budgets_with_data[0]].keys())
        if 'metrics' in samples_keys:
            samples_keys.remove('metrics')
        if 'recomputed_metrics' in samples_keys:
            samples_keys.remove('recomputed_metrics')
        num_samples = len(samples_keys)
    num_completions = config.num_return_sequences if hasattr(config, "num_return_sequences") else config.num_completions

    n_token_budgets = len(token_budgets_with_data)
    n_samples = num_samples
    n_completions = num_completions

    scores, probs, entropy, rankings = get_budget_sample_completion_metrics(SAVE_DATA,config)

    negent = -entropy

    sample_means_probs = np.mean(probs, axis=2).T
    sample_stds_probs = np.std(probs, axis=2).T
    sample_means_entropy = np.mean(entropy, axis=2).T
    sample_stds_entropy = np.std(entropy, axis=2).T

    df_probs = pd.DataFrame(sample_means_probs, index=[f'Sample {i}' for i in range(num_samples)], columns=token_budgets_with_data)
    df_entropy = pd.DataFrame(sample_means_entropy, index=[f'Sample {i}' for i in range(num_samples)], columns=token_budgets_with_data)

    x = np.array(df_probs.columns)
    y_probs = np.array(df_probs.values)
    y_negent = -1*np.array(df_entropy.values)

    if legacy_scoring:
        probs_breakthroughness = breakthrough_score(x, y_probs)
        probs_skewness = differences_skew_score(y_probs,magnitude_weight=False)
        negent_breakthroughness = breakthrough_score(x, y_negent)
        negent_skewness = differences_skew_score(y_negent,magnitude_weight=False)
    else:
        probs_breakthroughness = breakthrough_score(x, y_probs, diff_average="mean_sqrt")
        probs_skewness = differences_skew_score(y_probs,magnitude_weight=True)
        negent_breakthroughness = breakthrough_score(x, y_negent, diff_average="mean_sqrt")
        negent_skewness = differences_skew_score(y_negent,magnitude_weight=True)

    abs_probs_breakthroughness = np.abs(probs_breakthroughness)
    abs_negent_breakthroughness = np.abs(negent_breakthroughness)


    def filter_finite(arr):
        arr = np.asarray(arr)
        return arr[np.isfinite(arr)]

    abs_probs_breakthroughness_finite = filter_finite(abs_probs_breakthroughness)
    abs_negent_breakthroughness_finite = filter_finite(abs_negent_breakthroughness)
    probs_skewness_finite = filter_finite(probs_skewness)
    negent_skewness_finite = filter_finite(negent_skewness)

    n_total = len(abs_probs_breakthroughness)
    n_probs = len(abs_probs_breakthroughness_finite)
    n_negent = len(abs_negent_breakthroughness_finite)
    n_skew_probs = len(probs_skewness_finite)
    n_skew_negent = len(negent_skewness_finite)

   
    return {
        "abs_probs_breakthroughness": abs_probs_breakthroughness,
        "abs_negent_breakthroughness": abs_negent_breakthroughness,
        "probs_skewness": probs_skewness,
        "negent_skewness": negent_skewness,
        "df_probs": df_probs,
        "df_entropy": df_entropy
    }


def get_metric_table(config,SAVE_DATA,mode="token_budget_view",score_average_type="mode",use_recomputed_metrics_bool=True):
    """
    Returns a DataFrame showing metric values for all sample_idx, completion_idx pairs
    for a given token_budget and metric_type.

    Args:
        SAVE_DATA: dict containing the data
        token_budget: the token budget to select
        metric_type: one of 'answer_score', 'answer_probability', 'answer_entropy', 'ranking'

    Returns:
        pd.DataFrame: rows are sample_idx, columns are completion_idx, values are metric_value
    """

    table_dict = {}

    token_budgets = sorted(SAVE_DATA.keys())
    all_sample_idxs = list(SAVE_DATA[token_budgets[0]].keys())
    recomputed_metrics_exist = 'recomputed_metrics' in all_sample_idxs
    all_sample_idxs.remove('metrics')
    if recomputed_metrics_exist:
        all_sample_idxs.remove('recomputed_metrics')

    # List of all metric types to compute tables for
    metric_types = [
        "answer_score",
        "answer_probability",
        "answer_entropy",
        "answer_ranking",
        "solution_set_distribution",
        "ground_truth,model_answer",
    ]

    if use_recomputed_metrics_bool or metric_types == "ground_truth,model_answer":
        if recomputed_metrics_exist:
            #print(f"Recomputed metrics found. Using recomputed metrics...")
            recomputed_metrics = None  # Not needed, we use those already present
        else:
            print(f"Recomputing metrics for all tables...")
            recomputed_metrics = recompute_metrics(SAVE_DATA, config)

    # For each metric type, build a DataFrame
    for metric_type in metric_types:
        df = pd.DataFrame(index=all_sample_idxs, columns=token_budgets)

        for tb in token_budgets:
            if use_recomputed_metrics_bool or metric_type == "ground_truth,model_answer":
                if recomputed_metrics_exist:
                    metric_data = SAVE_DATA[tb]['recomputed_metrics'][metric_type]
                else:
                    metric_data = recomputed_metrics[tb][metric_type]
            else:
                metric_data = SAVE_DATA[tb]['metrics'][metric_type]
            # Build dict: sample_idx -> list of metric_values (across completions)
            sample_to_values = {}
            for sample_idx, _, metric_value in metric_data:
                if sample_idx not in sample_to_values:
                    sample_to_values[sample_idx] = []
                sample_to_values[sample_idx].append(metric_value)
            for sample_idx in all_sample_idxs:
                values = sample_to_values.get(sample_idx, [])
                # Round all values to 2 significant figures
                def round_sf(val, sf=2):
                    try:
                        arr = np.array(val, dtype=float)
                        if arr.size == 0:
                            return tuple()
                        rounded = np.round(arr, decimals=sf-1-int(np.floor(np.log10(np.abs(arr[~np.isnan(arr)]).max()))) if np.any(~np.isnan(arr)) and np.abs(arr[~np.isnan(arr)]).max() != 0 else 0)
                        return tuple(rounded.tolist())
                    except Exception:
                        # fallback for non-numeric or empty
                        return tuple(val)
                if metric_type == "answer_probability" or metric_type == "answer_entropy":
                    df.at[sample_idx, tb] = round_sf(values)
                else:
                    df.at[sample_idx, tb] = tuple(values)
                # Store as tuple for all completions (empty tuple if none)
        table_dict[metric_type] = df

    if mode=="token_budget_view":
        # For all token budgets, for each sample_idx and metric_type,
        # aggregate over completions as specified and create a DataFrame per metric_type
        # (index: sample_idx, columns: token_budgets, value: aggregated value)
        agg_tables = {}
        for metric_type in metric_types:
            df = pd.DataFrame(index=all_sample_idxs, columns=token_budgets)
            for tb in token_budgets:
                for sample_idx in all_sample_idxs:
                    values = table_dict[metric_type].at[sample_idx, tb]
                    if metric_type in ("answer_probability", "answer_entropy", "answer_ranking"):
                        if isinstance(values, (tuple, list)) and len(values) > 0:
                            df.at[sample_idx, tb] = float(np.mean(values))
                        else:
                            df.at[sample_idx, tb] = np.nan
                    elif metric_type == "answer_score":
                        if isinstance(values, (tuple, list)) and len(values) > 0:
                            if score_average_type=="mode":
                                    vals, counts = np.unique(values, return_counts=True)
                                    mode_val = vals[np.argmax(counts)]
                                    df.at[sample_idx, tb] = mode_val
                            elif score_average_type=="mean":
                                df.at[sample_idx, tb] = np.mean(values)
                            else:
                                raise ValueError(f"Invalid score_average_type: {score_average_type}")

                        else:
                            df.at[sample_idx, tb] = np.nan
                    elif metric_type == "solution_set_distribution":
                        # solution_set_distribution: leave as tuple of n completions
                        df.at[sample_idx, tb] = values
                    elif metric_type == "ground_truth,model_answer":
                        df.at[sample_idx, tb] = values
            agg_tables[metric_type] = df

        # Ensure answer_score, answer_probability, answer_entropy are all dtype float
        for k in ['answer_score', 'answer_probability', 'answer_entropy']:
            if k in agg_tables:
                agg_tables[k] = agg_tables[k].astype(float)

        return agg_tables

    return table_dict


def get_token_count_table(config,SAVE_DATA,mode="full_view"):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    token_budgets = sorted(list(SAVE_DATA.keys()))
    token_counts_present = 'token_count' in SAVE_DATA[token_budgets[0]][0]['completions'][0]

    token_budgets = sorted(list(SAVE_DATA.keys()))
    sample_idxs = list(SAVE_DATA[token_budgets[0]].keys())
    sample_idxs = [a for a in sample_idxs if isinstance(a, int)]
    df = pd.DataFrame(index=sorted(sample_idxs), columns=token_budgets)
    for tb in token_budgets:
        for sample_idx in sorted(sample_idxs):
            if sample_idx not in list(SAVE_DATA[tb].keys()):
                continue
            token_counts = []
            for completion_idx in range(config.num_completions):
                if token_counts_present:
                    token_count = SAVE_DATA[tb][sample_idx]['completions'][completion_idx]['token_count']
                else:
                    text = SAVE_DATA[tb][sample_idx]['completions'][completion_idx]['text'][0]
                    encoded_text = tokenizer.encode(text, add_special_tokens=False)
                    end_of_input_idxs = workflow_utils.get_end_of_input_idxs(encoded_text, tokenizer)[0]
                    token_count = workflow_utils.count_generated_tokens(encoded_text, end_of_input_idxs, tokenizer)
                token_counts.append(token_count)
            df.loc[sample_idx, tb] = token_counts
        
    if mode=="full_view":
        return df 
    elif mode=="token_budget_view":
        df_mean = pd.DataFrame(index=df.index, columns=df.columns)
        for row in df.index:
            for col in df.columns:
                cell = df.at[row, col]
                if isinstance(cell, (list, tuple)) and len(cell) > 0:
                    mean_val = int(round(np.mean(cell)))
                    df_mean.at[row, col] = mean_val
                else:
                    df_mean.at[row, col] = np.nan
        return df_mean
    else:
        raise ValueError(f"Unknown mode: {mode}")

        
