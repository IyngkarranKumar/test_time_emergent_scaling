
import pickle, importlib, os, copy, random, pdb
import matplotlib.pyplot as plt, numpy as np, pandas as pd, utils
import seaborn as sns
import conf
from tempfile import NamedTemporaryFile
import webbrowser
import json
import re

importlib.reload(conf)
importlib.reload(utils)
importlib.reload(utils.dataset_utils)
importlib.reload(utils.workflow_utils)

from transformers import AutoTokenizer
from conf import config
from utils import *
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import pearsonr, spearmanr


random.seed(42)
np.random.seed(42)

#plot settings

sns.set_style('whitegrid')
sns.set_context('paper',font_scale=0.8)


def topk_nanmax_arg(x, k=1):
    x = np.asarray(x)
    mask = (~np.isnan(x)) & (~np.isinf(x)) & (x > float('-inf'))
    if not np.any(mask):
        return []
    valid_x = np.where(mask, x, float('-inf'))
    if k == 1:
        return int(np.argmax(valid_x))
    else:
        # Get indices of top-k values
        topk_indices = np.argpartition(valid_x, -k)[-k:]
        # Sort them in descending order of value
        topk_indices = topk_indices[np.argsort(valid_x[topk_indices])[::-1]]
        return topk_indices.tolist()
    

def dict_structure(dict_data):
    df=pd.json_normalize(dict_data)
    return df.info()

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

def get_model_size(path):
    """Extract model size in billions from path. Returns float or None."""
    # Get the model name part (before dataset and timestamp)
    model_name = path.split('/')[-1].split('_')[0]
    
    # Look for number followed by 'B' (case insensitive)
    match = re.search(r'(\d+(?:\.\d+)?)B', model_name, re.IGNORECASE)
    
    if match:
        return float(match.group(1))
    return None

def read_save_data(data_dir,localisation_run=False):

    with open(f"{data_dir}/config.pkl", 'rb') as f:
        config = pickle.load(f)

    # Load and merge all save files
    save_files = [f for f in os.listdir(data_dir) if f.startswith('save_file_') and f.endswith('.pkl')]

    # Sort save_files by token_budget (ascending), then by first batch number (ascending)    
    save_files = sorted(save_files, key=extract_token_budget_and_batch)
    
    merged_save_data = {}

    time_based_saving = True if "batch" in save_files[0] else False
    
    
    for save_file in save_files:

        with open(f"{data_dir}/{save_file}", 'rb') as f:
            save_data = pickle.load(f)


            if time_based_saving:

                # Extract token_budget from filename (assume it's the last number before .pkl)
                match = re.search(r'_(\d+)\.pkl$', save_file)
                if match:
                    token_budget = int(match.group(1))
                else:
                    # fallback: skip if cannot parse
                    raise ValueError(f"Cannot parse token budget from filename: {save_file}")

                save_data_without_metrics = {k: v for k, v in save_data[token_budget].items() if k != 'metrics'}
                metrics_data = save_data[token_budget]['metrics']

                if token_budget not in merged_save_data:
                    merged_save_data[token_budget] = {}

                # save_data is a dict: {sample_idx: {...}, ...}
                # Add each sample (excluding 'metrics') to merged_save_data
                for sample_idx, sample_data in save_data_without_metrics.items():
                    merged_save_data[token_budget][sample_idx] = sample_data

                # Handle metrics: concatenate metrics across save files for this token_budget
                if 'metrics' not in merged_save_data[token_budget]:
                    merged_save_data[token_budget]['metrics'] = {}

                for metric_name, metric_list in metrics_data.items():
                    if metric_name not in merged_save_data[token_budget]['metrics']:
                        merged_save_data[token_budget]['metrics'][metric_name] = []
                    merged_save_data[token_budget]['metrics'][metric_name].extend(metric_list)

            else:   
                # Merge dictionaries
                for token_budget in save_data:
                    if token_budget not in merged_save_data:
                        merged_save_data[token_budget] = save_data[token_budget]
                    else:
                        merged_save_data[token_budget].update(save_data[token_budget])

        # Order SAVE_DATA by key size (token_budget)
    merged_save_data = dict(sorted(merged_save_data.items(), key=lambda x: x[0]))
    # Ensure 'metrics' is the last key in each token_budget dict
    for token_budget in merged_save_data:
        d = merged_save_data[token_budget]
        if 'metrics' in d:
            # Remove and re-insert 'metrics' at the end
            metrics_val = d.pop('metrics')
            d['metrics'] = metrics_val


    if localisation_run:
        # For localisation run, save files are named save_file_{token_budget}_{sample_idx}.pkl
        # We want merged_save_data[token_budget][sample_idx] = ... for all files
        merged_save_data = {}
        for save_file in save_files:
            # Parse token_budget and sample_idx from filename
            # Example: save_file_128_3.pkl
            parts = save_file.replace('.pkl', '').split('_')
            if len(parts) < 4:
                continue  # skip malformed

            sample_idx = int(parts[4])
            token_budget = int(parts[5])
            
            with open(f"{data_dir}/{save_file}", 'rb') as f:
                save_data = pickle.load(f)
            if sample_idx not in merged_save_data:
                merged_save_data[sample_idx] = {}
            merged_save_data[sample_idx][token_budget] = list(save_data.values())[0]
        
        # Order by sample_idx
        merged_save_data = dict(sorted(merged_save_data.items(), key=lambda x: x[0]))

        for sample_idx in merged_save_data:
            merged_save_data[sample_idx] = dict(sorted(merged_save_data[sample_idx].items(), key=lambda x: x[0]))

    return config, merged_save_data

def plot_data(data_dir=None, config=None, SAVE_DATA=None, metrics=["answer_score", "answer_probability", "answer_entropy"], sample_idxs=None, n_samples=None, logy=False, titles=None,fig_title=None,localisation_run=False,save_path=None,name=None,score_average_type="mode",style_dict=None,single_legend=True):

    if style_dict is not None:
        old_params = plt.rcParams.copy() 
        plt.rcParams.update(style_dict)

    # Map legacy metric names to standardized keys if needed
    metric_key_map = {
        "score": "answer_score",
        "probs": "answer_probability",
        "prob": "answer_probability",
        "entropy": "answer_entropy",
        "ranking": "answer_ranking"
    }
    # Standardize metrics list
    metrics = [metric_key_map.get(m, m) for m in metrics]

    if data_dir is None: 
        SAVE_DATA = SAVE_DATA
        config = config
        log2_start_budget, log2_end_budget = np.log2(list(SAVE_DATA.keys())[0]), np.log2(list(SAVE_DATA.keys())[-1])
    else: 
        config, SAVE_DATA = read_save_data(data_dir)
        log2_start_budget, log2_end_budget = config.start_token_budget, config.end_token_budget

    n_sample_data = (len(SAVE_DATA[2**log2_start_budget].keys())) - 1
    n_completions = len(SAVE_DATA[2**log2_start_budget][0]['completions'])

    if sample_idxs is None:
        assert n_sample_data >= n_samples
        np.random.seed(42)
        selected_samples = np.arange(n_samples)
    else:
        selected_samples = sample_idxs
        n_samples = len(selected_samples)

    # Handle axes for single or multiple samples
    if len(selected_samples) == 1:
        n_rows = 1
        fig, axes = plt.subplots(1, 1)
        axes = np.array([axes])
    else:
        n_rows = (len(selected_samples) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2)
        axes = axes.flatten()

    # Variables to collect legend info from first subplot
    legend_handles = []
    legend_labels = []
    legend_collected = False

    for plot_idx, sample_idx in enumerate(selected_samples):
        token_budgets = []
        scores_average = []
        probs_mean = []
        probs_std = []
        entropies_mean = []
        entropies_std = []

        all_token_budgets = sorted([tb for tb in SAVE_DATA.keys()])

        for token_budget in all_token_budgets:
            score_data_all = SAVE_DATA[token_budget]['metrics'].get('answer_score', None)
            probs_data_all = SAVE_DATA[token_budget]['metrics'].get('answer_probability', None)
            entropy_data_all = SAVE_DATA[token_budget]['metrics'].get('answer_entropy', None)

            token_budgets.append(token_budget)
            if 'metrics' in SAVE_DATA[token_budget] and score_data_all is not None and probs_data_all is not None and entropy_data_all is not None:
                sample_scores = []
                sample_probs = []
                sample_entropies = []

                for completion_idx in range(n_completions):
                    if localisation_run:
                        metric_idx = completion_idx
                    else:
                        metric_idx = sample_idx * n_completions + completion_idx

                    sample_scores.append(score_data_all[metric_idx][-1])
                    sample_probs.append(probs_data_all[metric_idx][-1])
                    sample_entropies.append(entropy_data_all[metric_idx][-1])

                if score_average_type=="mode":
                    scores_average.append(stats.mode(sample_scores)[0])
                elif score_average_type=="mean":
                    scores_average.append(np.mean(sample_scores))
                else:
                    raise ValueError(f"Invalid score_average: {score_average}")

                probs_mean.append(np.mean(sample_probs))
                probs_std.append(np.std(sample_probs))
                entropies_mean.append(np.mean(sample_entropies))
                entropies_std.append(np.std(sample_entropies))
            else:
                scores_average.append(np.nan)
                probs_mean.append(np.nan)
                probs_std.append(np.nan)
                entropies_mean.append(np.nan)
                entropies_std.append(np.nan)

        ax2 = axes[plot_idx].twinx() if "answer_entropy" in metrics else None

        if "answer_score" in metrics:
            axes[plot_idx].plot(token_budgets, scores_average, label='Score', color='blue')

        if "answer_probability" in metrics:
            if logy:
                logprobs_mean = np.clip(np.log10(probs_mean), -100, 100)
                axes[plot_idx].errorbar(token_budgets, logprobs_mean, yerr=probs_std, label='log10(p)', color='red')
            else:
                axes[plot_idx].errorbar(token_budgets, probs_mean, yerr=probs_std, label='Probability', color='red')
                axes[plot_idx].set_ylim(-0.1, 1.1)

        if "answer_entropy" in metrics and ax2 is not None:
            ax2.errorbar(token_budgets, -np.array(entropies_mean), yerr=entropies_std, label='Negentropy', color='green')

        axes[plot_idx].set_xscale('log', base=2)
        axes[plot_idx].set_xticks(token_budgets)
        axes[plot_idx].set_xticklabels([str(tb) for tb in token_budgets])
        
        # Determine position in grid for selective axis removal
        row = plot_idx // 2
        col = plot_idx % 2
        
        # Set labels only for appropriate positions
        if row == n_rows - 1:  # Bottom row
            axes[plot_idx].set_xlabel('Token Budget')

        
                # In your plotting function, modify the y-axis labeling:
        if col == 0 and row==1:  # Left column
            if row == 1 and col == 0:  # Center of left column (for 2x2 grid, row 1 is center)
                axes[plot_idx].set_ylabel('Score/Probability', labelpad=0)
                axes[plot_idx].yaxis.set_label_coords(0.0, 0.5, transform=fig.transFigure)
  
        if col==1 and row==1:
            if ax2:
                ax2.set_ylabel('Negentropy')  # Right axis
                ax2.yaxis.set_label_coords(1.0, 0.5, transform=fig.transFigure)

        # Also color the tick labels to match
        axes[plot_idx].tick_params(axis='y', labelcolor='blue')
        if ax2:
            ax2.tick_params(axis='y', labelcolor='green')
            
        axes[plot_idx].set_title(
            titles[plot_idx] if titles is not None else f'Sample {sample_idx}'
        )

        # Remove labels and ticks for non-edge subplots
        if col > 0:  # Not leftmost column
            axes[plot_idx].set_ylabel('')
            axes[plot_idx].tick_params(left=False, labelleft=False)
            
        if row < n_rows - 1:  # Not bottom row
            axes[plot_idx].set_xlabel('')
            axes[plot_idx].tick_params(bottom=False, labelbottom=False)

        # Collect legend handles and labels from first subplot only
        if not legend_collected:
            lines1, labels1 = axes[plot_idx].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
            legend_handles = lines1 + lines2
            legend_labels = labels1 + labels2
            legend_collected = True

    # Add single legend at the top
    if legend_handles and single_legend:
        fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.10), 
                   ncol=len(legend_labels), frameon=False)

    title_y = 1.0
    if fig_title is None:
        fig.suptitle(
            f'{config.dataset_name} — {config.model_name} — {config.num_completions} completions',
            y=title_y
        )
    else:
        fig.suptitle(fig_title,y=title_y)
    
    fig.tight_layout()

    if save_path is not None and name is not None:
        print(save_path)
        fig.savefig(f"{save_path}/{name}.png", bbox_inches='tight')
    else:
        plt.show()

def plot_aggregate_data(SAVE_DATA, config, metrics=['answer_score', 'answer_probability', 'answer_entropy', 'answer_ranking'], logy=False, style_dict=None,title=None,save_path=None):
    """
    Plots aggregate metrics (averaged across all samples and completions) for each metric type.
    """
    if style_dict is not None:
        old_params = plt.rcParams.copy() 
        plt.rcParams.update(style_dict)

    token_budgets = list(SAVE_DATA.keys())
    token_budgets = sorted(token_budgets)
    metric_types = metrics

    fig, axes = plt.subplots(1, 1)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax = axes[0]
    ax2 = ax.twinx() if "answer_entropy" in metric_types else None

    # Prepare storage for each metric type
    metric_means = {m: [] for m in metric_types}
    metric_stds = {m: [] for m in metric_types}

    for token_budget in token_budgets:
        metric_data = SAVE_DATA[token_budget]['metrics']
        for m in metric_types:
            if m not in metric_data:
                metric_means[m].append(np.nan)
                metric_stds[m].append(np.nan)
                continue
            arr = np.array(metric_data[m])  # shape: (N, 3)
            if arr.size == 0:
                metric_means[m].append(np.nan)
                metric_stds[m].append(np.nan)
                continue
            values = arr[:, 2].astype(float)
            metric_means[m].append(np.nanmean(values))
            metric_stds[m].append(np.nanstd(values))

    # Plotting
    colors = {'answer_score': 'blue', 'answer_probability': 'red', 'answer_entropy': 'green', 'answer_ranking': 'purple'}
    labels = {'answer_score': 'Score', 'answer_probability': 'Probability', 'answer_entropy': 'Negentropy', 'answer_ranking': 'Ranking'}

    if "answer_score" in metric_types:
        ax.plot(token_budgets, metric_means['answer_score'], label=labels['answer_score'], color=colors['answer_score'])
    if "answer_probability" in metric_types:
        if logy:
            logprobs_mean = np.clip(np.log10(metric_means['answer_probability']), -100, 100)
            ax.errorbar(token_budgets, logprobs_mean, yerr=metric_stds['answer_probability'], label='log10(p)', color=colors['answer_probability'])
        else:
            ax.errorbar(token_budgets, metric_means['answer_probability'], yerr=metric_stds['answer_probability'], label=labels['answer_probability'], color=colors['answer_probability'])
            ax.set_ylim(-0.1, 1.1)

    if "answer_entropy" in metric_types and ax2 is not None:
        ax2.errorbar(token_budgets, -np.array(metric_means['answer_entropy']), yerr=metric_stds['answer_entropy'], label=labels['answer_entropy'], color=colors['answer_entropy'])
    
    if ax2 is not None and "answer_entropy" in metric_types:
        entropy_vals = -np.array(metric_means['answer_entropy'])
        entropy_stds = np.array(metric_stds['answer_entropy'])
        lower = np.nanmin(entropy_vals - entropy_stds)
        upper = np.nanmax(entropy_vals + entropy_stds)
        ax2.set_ylim(lower, upper)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Token Budget')
    ax.set_ylabel('Score/Probability')
    if ax2:
        ax2.set_ylabel('Negentropy')


    ax.tick_params(axis='both', which='major')
    if ax2:
        ax2.tick_params(axis='y', which='major')
    ax.grid(True)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
    if ax2:
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax.legend(lines1, labels1, loc='best')

    
    if title is None:
        fig.suptitle(f'{config.dataset_name} - {config.model_name} - {config.num_samples} samples - {config.num_completions} completions (Aggregate)')
    else:
        fig.suptitle(title)
    plt.tight_layout()


    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else: 
        plt.show()


#text viewing 
def df_to_html(df,save=True,save_path="results_output"):
    # Create a copy of the dataframe and replace \n with <br> tags
    df_html = df.copy()
    for col in df_html.columns:
        if df_html[col].dtype == 'object':  # Only process string columns
            df_html[col] = df_html[col].astype(str).str.replace('\n', '<br>', regex=False)
    
    # Create HTML table with styling
    html = """
    <html>
    <head>
    <style>
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
        vertical-align: top;  /* Added to align content to top */
    }
    th {
        background-color: #f2f2f2;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    </style>
    </head>
    <body>
    """
    # Use escape=False to allow HTML tags in the dataframe
    html += df_html.to_html(index=False, escape=False)
    html += "</body></html>"

    # Get current df index based on existing files
    df_files = [f for f in os.listdir(save_path) if f.startswith('df_') and f.endswith('.html')]
    df_idx = len(df_files)

    if save:
        with open(f'{save_path}/df_{df_idx}.html', 'w') as f:
            f.write(html)
    else:
        with NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            f.write(html)
        webbrowser.open('file://' + f.name)

def token_budget_text_view(config,SAVE_DATA,completion_idx=0,save=True,sample_idxs=None):
    
    # Get all token budgets
    token_budgets = list(SAVE_DATA.keys())
    
    # Initialize data dictionary
    data = {f'budget_{budget}': [] for budget in token_budgets}
    
    # Fill data
    for budget in token_budgets:
        # Get sample indices
        if sample_idxs is None:
            curr_sample_idxs = [k for k in SAVE_DATA[budget].keys() if isinstance(k, int)]
        else:
            curr_sample_idxs = sample_idxs
            
        # Get text for each sample's first completion
        for sample_idx in curr_sample_idxs:
            text = SAVE_DATA[budget][sample_idx]['completions'][completion_idx]['text'][0]
            data[f'budget_{budget}'].append(text)
            
    # Create dataframe
    df = pd.DataFrame(data, index=curr_sample_idxs).reset_index()

    if "DeepSeek" in config.model_name:
        pad_token = "!"
    elif "QwQ" in config.model_name:
        pad_token = "<|endoftext|>"
    elif "Phi-4" in config.model_name:
        pad_token = "<|dummy_85|>"
    else:
        pad_token = "<pad>"

    df_with_delims = add_delimiters_to_text(df, token_budgets, config,mode="token_budget_view", pad_token=pad_token)
    
    df_to_html(df_with_delims,save=save)

def completions_text_view(config,SAVE_DATA,token_budget,save=True,sample_idxs=None):
    # Get all sample indices
    # Get all sample indices if none provided
    token_budgets = list(SAVE_DATA.keys())


    if sample_idxs is None:
        sample_idxs = [k for k in SAVE_DATA[token_budget].keys() if isinstance(k, (int, np.integer))]
    
    # Get number of completions from first sample
    n_completions = config.num_completions
    
    # Initialize data dictionary
    data = {f'completion_{i}': [] for i in range(n_completions)}
    
    # Fill data
    for sample_idx in sample_idxs:
        for completion_idx in range(n_completions):
            text = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['text'][0]
            data[f'completion_{completion_idx}'].append(text)
    
            
    # Create dataframe
    df = pd.DataFrame(data, index=sample_idxs).reset_index()
    
    if "DeepSeek" in config.model_name:
        pad_token = "!"
    elif "QwQ" in config.model_name.lower():
        pad_token = "<|endoftext|>"
    elif "Phi-4" in config.model_name.lower():
        pad_token = "<|dummy_85|>"
    else:
        pad_token = "<pad>"
    df_with_delims = add_delimiters_to_text(df, token_budgets, config,mode="completions_view", pad_token=pad_token)

    df_to_html(df_with_delims,save=save)

def generate_dummy_completion_text(token_budget, sample_idx, completion_idx):
    """Generate realistic-looking completion text based on the pattern from your real data"""
    
    # Expanded set of sample problems (cycling through if sample_idx > available problems)
    problems = [
        "Sarah has 24 apples. She gives away half of them to her friends. How many apples does Sarah have left?",
        "A store sells pencils for $0.50 each. If Tom buys 8 pencils, how much does he spend?",
        "Maria reads 15 pages of a book each day. If the book has 180 pages, how many days will it take her to finish?",
        "A recipe calls for 3 cups of flour. If you want to make half the recipe, how much flour do you need?",
        "There are 36 students in a class. If they form groups of 4, how many groups will there be?",
        "John runs 3 miles every morning. How many miles does he run in a week?",
        "A pizza is cut into 8 slices. If 5 people each eat 1 slice, how many slices are left?",
        "Emma saves $5 every week. How much money will she have saved after 12 weeks?",
        "A box contains 48 chocolates arranged in 6 rows. How many chocolates are in each row?",
        "Lisa has $50. She spends $18 on a book and $12 on lunch. How much money does she have left?"
    ]
    
    problem = problems[sample_idx % len(problems)]
    
    # Set random seed based on inputs for reproducible but varied outputs
    local_random = random.Random(sample_idx * 1000 + completion_idx * 100 + token_budget)
    answer = local_random.randint(1, 100)
    
    # Generate different completion styles based on completion_idx
    completion_styles = {
        0: "detailed_reasoning",    # Verbose step-by-step approach
        1: "concise_direct",       # Short and to the point
        2: "methodical_working",   # Shows work but moderate length
        3: "exploratory_thinking", # Shows multiple approaches or uncertainty
        4: "confident_assertion",  # Direct but includes confidence statements
        5: "educational_explanation", # Teaching-style with general principles
    }
    
    style = completion_styles.get(completion_idx % len(completion_styles), "methodical_working")
    
    if style == "detailed_reasoning":
        completion = f"<｜begin▁of▁sentence｜>{problem} Let me solve this step by step.\n\n"
        completion += "First, I need to identify what information is given and what I need to find. "
        completion += "Looking at this problem, I can see the key variables and relationships. "
        completion += "Let me work through each step carefully to ensure accuracy. "
        completion += f"After working through the math systematically, I arrive at my answer.\n\nTHE FINAL ANSWER IS: {answer}"
        
    elif style == "concise_direct":
        completion = f"<｜begin▁of▁sentence｜>{problem} "
        completion += f"The answer is {answer}."
        
    elif style == "methodical_working":
        completion = f"<｜begin▁of▁sentence｜>{problem} Let me work through this problem. "
        completion += "I'll set up the calculation and solve it step by step. "
        completion += f"Following the mathematical operations, the final answer is {answer}."
        
    elif style == "exploratory_thinking":
        completion = f"<｜begin▁of▁sentence｜>{problem} Hmm, let me think about this. "
        completion += "There might be different ways to approach this problem. "
        completion += "Let me consider the most straightforward method. "
        completion += f"Actually, working through it this way, I get {answer}. "
        completion += "Let me double-check this makes sense."
        
    elif style == "confident_assertion":
        completion = f"<｜begin▁of▁sentence｜>{problem} "
        completion += "This is a straightforward calculation. "
        completion += f"I'm confident the answer is {answer}. "
        completion += "This type of problem has a clear solution method."
        
    else:  # educational_explanation
        completion = f"<｜begin▁of▁sentence｜>{problem} "
        completion += "This is a classic example of a basic arithmetic word problem. "
        completion += "The key is to identify what operation is needed. "
        completion += f"Applying the appropriate mathematical operation gives us {answer}. "
        completion += "Understanding these fundamentals is important for more complex problems."
    
    # Adjust length based on token budget (roughly 4 chars per token)
    base_length = len(completion)
    target_length = int(token_budget * 4 * 0.8)  # Target 80% of token budget
    
    if base_length < target_length:
        # Add appropriate filler based on style and token budget
        if style == "detailed_reasoning":
            completion += " Let me verify this calculation once more to ensure correctness. "
            completion += "Going through the steps again confirms my reasoning is sound. "
            if token_budget >= 512:
                completion += "This systematic approach helps avoid computational errors. "
                completion += "When solving word problems, methodical thinking is essential. "
            if token_budget >= 1024:
                completion += "Mathematical problem-solving requires careful attention to detail. "
                completion += "Each step builds upon the previous one to reach the final solution. "
                completion += "Developing strong problem-solving skills takes practice and patience. "
                
        elif style == "educational_explanation":
            if token_budget >= 256:
                completion += " Students often struggle with word problems because they require translating between language and mathematics. "
            if token_budget >= 512:
                completion += "The key is to identify the mathematical relationship described in words. "
                completion += "Practice with similar problems helps build pattern recognition skills. "
            if token_budget >= 1024:
                completion += "Word problems appear frequently in mathematics education because they connect abstract concepts to real-world situations. "
                completion += "This connection helps students understand the practical applications of mathematical thinking. "
                
        else:
            # Generic expansion for other styles
            completion += " This approach ensures accuracy in the solution. "
            if token_budget >= 512:
                completion += "Mathematical reasoning involves logical steps and careful calculation. "
            if token_budget >= 1024:
                completion += "Problem-solving skills develop through practice and systematic thinking. "
                completion += "Each problem type has characteristic patterns that become recognizable with experience. "
    
    completion += "<｜end▁of▁sentence｜>"
    
    return completion

def generate_dummy_data(token_budgets, num_samples, num_completions):
    """
    Generate dummy data structure with arbitrary parameters
    
    Args:
        token_budgets (list): List of token budget values (e.g., [128, 256, 512])
        num_samples (int): Number of sample indices (0 to num_samples-1)
        num_completions (int): Number of completion indices (0 to num_completions-1)
    
    Returns:
        dict: Complete data structure matching the experimental format
    """
    
    data = {}
    
    for token_budget in token_budgets:
        data[token_budget] = {}
        
        # Generate completions for each sample
        for sample_idx in range(num_samples):
            data[token_budget][sample_idx] = {'completions': {}}
            
            for completion_idx in range(num_completions):
                completion_text = generate_dummy_completion_text(token_budget, sample_idx, completion_idx)
                
                data[token_budget][sample_idx]['completions'][completion_idx] = {
                    'text': [completion_text],
                    'logits': None  # As requested, skip the logit tensors
                }
        
        # Generate metrics
        answer_scores = []
        answer_probabilities = []
        answer_entropies = []
        
        for sample_idx in range(num_samples):
            for completion_idx in range(num_completions):
                # Set seed for reproducible metrics based on all parameters
                metric_random = random.Random(sample_idx * 10000 + completion_idx * 1000 + token_budget)
                
                # Generate realistic metrics
                # Answer score: binary (0 or 1) with higher budgets having slightly better performance
                # Base success rate between 30-90% depending on token budget
                min_success_rate = 0.3
                max_success_rate = 0.9
                budget_factor = min(token_budget / 2048, 1.0)  # Normalize to max budget
                score_prob = min_success_rate + budget_factor * (max_success_rate - min_success_rate)
                
                # Add some variation based on completion_idx (some styles might be more/less successful)
                completion_modifier = {
                    0: 0.1,   # detailed_reasoning: slightly better
                    1: -0.05, # concise_direct: slightly worse
                    2: 0.0,   # methodical_working: baseline
                    3: -0.1,  # exploratory_thinking: uncertainty hurts performance
                    4: 0.05,  # confident_assertion: confidence helps slightly
                    5: 0.08,  # educational_explanation: teaching approach helps
                }.get(completion_idx % 6, 0.0)
                
                score_prob = max(0.1, min(0.95, score_prob + completion_modifier))
                answer_score = 1 if metric_random.random() < score_prob else 0
                answer_scores.append((sample_idx, completion_idx, answer_score))
                
                # Answer probability: between 0 and 1, correlated with score
                if answer_score == 1:
                    prob = metric_random.uniform(0.6, 1.0)  # Higher probability for correct answers
                else:
                    prob = metric_random.uniform(0.0, 0.4)  # Lower probability for incorrect answers
                answer_probabilities.append((sample_idx, completion_idx, prob))
                
                # Answer entropy: inversely related to probability
                entropy = -prob * np.log2(max(prob, 1e-10)) - (1-prob) * np.log2(max(1-prob, 1e-10))
                # Add some noise and ensure non-negative
                entropy = max(0, entropy + metric_random.uniform(-0.3, 0.3))
                answer_entropies.append((sample_idx, completion_idx, entropy))
        
        data[token_budget]['metrics'] = {
            'answer_score': answer_scores,
            'answer_probability': answer_probabilities,
            'answer_entropy': answer_entropies
        }
    
    return data

def breakthrough_score(x,y,nan_threshold=100):

    n_samples = y.shape[0]
    y_min = np.min(y, axis=1)
    y_max = np.max(y, axis=1)
    argmin = np.argmin(y, axis=1)
    argmax = np.argmax(y, axis=1)
    delta_y = np.diff(y, axis=1)
    root_median_sq = np.sqrt(np.median(delta_y**2, axis=1))
    sign_args = np.where(argmax - argmin > 0, 1, -1)
    I_y = (y_max - y_min) * sign_args
    breakthroughness = I_y / root_median_sq

    breakthroughness = np.clip(breakthroughness, -nan_threshold, nan_threshold)



    return breakthroughness

def differences_skew_score(y,skewness_type="adjusted_fisher_pearson"):
        
        diffs = np.diff(y, axis=1)
        n_samples = diffs.shape[0]
        n_diffs = diffs.shape[1]
        assert n_diffs>2, f"Need array of atleast size 3 to comptue diff skewness"
        skewness = np.zeros(n_samples)
        
        for i in range(n_samples):
            diffs_i = diffs[i]
            correction_factor = (np.sqrt(len(diffs_i)*(len(diffs_i)-1)))/(len(diffs_i)-2)
            std_dev = np.sqrt(np.var(diffs_i))
            mean = np.mean(diffs_i)
            fisher_pearson = np.sum(((diffs_i-mean)**3/len(diffs_i))/std_dev**3).item()
            
            if skewness_type=="fisher_pearson":
                skewness[i] = fisher_pearson
            elif skewness_type=="adjusted_fisher_pearson":
                skewness[i] = correction_factor*fisher_pearson
            else:
                raise ValueError(f"Invalid skewness type: {skewness_type}")
            
            
        return skewness

def differences_clustering(ys,n_clusters=3):
    """
    Clusters curves in difference space
    
    
    Args:
    - ys: array of dependent variables
    - n_clusters: number of clusters
    """

    diffs=np.diff(ys,axis=1)
    diffs_reshaped=diffs.reshape(diffs.shape[0],-1)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters=kmeans.fit_predict(diffs_reshaped)
    return clusters
    
def add_delimiters_to_text(df, token_budgets, config, pad_token="<pad>",mode="token_budget_view"):

    """Add delimiters to text to show question boundaries and token budget cutoffs"""

    dataset_setup = utils.DatasetSetup(config.dataset_name,config=config)
    dataset = dataset_setup.load_dataset()
    if "Qwen2.5" in config.model_name or "simplescaling" in config.model_name:
        end_of_input_ids = utils.qwen_end_of_input_ids
    elif "google/gemma" in config.model_name:
        end_of_input_ids = utils.gemma_end_of_input_ids
    elif "deepseek" in config.model_name:
        end_of_input_ids = utils.deepseek_end_of_input_ids
    elif "QwQ" in config.model_name:
        end_of_input_ids = utils.qwq_end_of_input_ids
    elif "Phi-4" in config.model_name:
        end_of_input_ids = utils.phi_end_of_input_ids
    else:
        raise ValueError(f"No end of input ids found for {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name,local_files_only=True if os.getenv("TRANSFORMERS_OFFLINE") else False)
    end_of_input_str = tokenizer.decode(end_of_input_ids,add_special_tokens=False)

    if pad_token is not None:
        tokenizer.pad_token = pad_token #set toknizer pad token to custom

    # Define ANSI escape codes for red color
    RED_START = '<span style="color: red;"><b>'
    RED_END = '</b></span>'
    ANSWER_START = '<span style="color: green;"><b>'
    ANSWER_END = '</b></span><br><br>'

    question_delim = [f"\n\n{RED_START}--QUESTION_END--{RED_END}\n\n"]
    token_budget_delims = [f"\n\n{RED_START} --BUDGET_{x}-- {RED_END}\n\n" for x in token_budgets]
    list_of_delims = question_delim + token_budget_delims

    # Initialize new dataframe with same structure
    df_with_delims = pd.DataFrame(columns=df.columns)
    
    if mode=="token_budget_view":
        # Process each row
        for idx, row in df.iterrows():
            sample_idx = row['index']
            sample = next(d for d in dataset if d['sample_idx'] == sample_idx)
            question = sample['question']
            answer = sample['answer']
            if config.dataset_name == "Idavidrein/gpqa":
                mcq_answer = sample['MCQ answer']
                answer = mcq_answer
            
            new_row = {'index': sample_idx}
            
            # Process each budget column
            for budget in token_budgets:
                try: 
                    col = f'budget_{budget}'
                    text = row[col]
                except:
                    continue

                
                # Remove pad tokens if present
                if tokenizer.pad_token is not None:
                    text = text.replace(tokenizer.pad_token, '')

                #remove bos tokens 
                if tokenizer.bos_token is not None:
                    text = text.replace(tokenizer.bos_token, '')

                
                
                # Insert question delimiter
                text_parts = text.split(end_of_input_str, 1)
                if len(text_parts) > 1:
                    text = text_parts[0] + end_of_input_str + " " + question_delim[0] + " " + text_parts[1]


                    
                    # Get tokens after question
                    response = text_parts[1]
                    response_tokens = tokenizer.encode(response,add_special_tokens=False)
                    
                    # Get all budgets lower than current budget
                    lower_budgets = sorted([b for b in token_budgets if b <= budget], reverse=True)
                    
                    processed_text = response
                    for lower_budget in lower_budgets:
                        if len(response_tokens) > lower_budget:
                            # Split at the lower budget point
                            budget_text = tokenizer.decode(response_tokens[:int(lower_budget)])
                            remaining_text = tokenizer.decode(response_tokens[int(lower_budget):])
                            
                            # Insert delimiter
                            budget_delim = f" {token_budget_delims[token_budgets.index(lower_budget)]} "
                            processed_text = budget_text + budget_delim + remaining_text
                            
                            # Update response tokens for next iteration
                            response_tokens = tokenizer.encode(processed_text,add_special_tokens=False)
                    
                    # Combine all parts
                    text = text_parts[0] + end_of_input_str + " " + question_delim[0] + " " +  f" {ANSWER_START}ANSWER: {answer} {ANSWER_END} " + processed_text
                        
                new_row[col] = text
                
            df_with_delims = pd.concat([df_with_delims, pd.DataFrame([new_row])], ignore_index=True)

    elif mode=="completions_view":
        for idx, row in df.iterrows():
            sample_idx = row['index']
            sample = next(d for d in dataset if d['sample_idx'] == sample_idx)
            question = sample['question']
            answer = sample['answer']
            if config.dataset_name == "Idavidrein/gpqa":
                mcq_answer = sample['MCQ answer']
                answer = mcq_answer

            new_row = {'index': sample_idx}

            for col in df.columns:
                if col == 'index':
                    continue
                text = row[col]

                # Remove pad tokens if present
                if tokenizer.pad_token is not None:
                    text = text.replace(tokenizer.pad_token, '')

                # Remove bos tokens
                if tokenizer.bos_token is not None:
                    text = text.replace(tokenizer.bos_token, '')

                # Insert question delimiter
                text_parts = text.split(end_of_input_str, 1)
                if len(text_parts) > 1:
                    text = text_parts[0] + end_of_input_str + " " + question_delim[0] + " " + text_parts[1]
                    # Insert answer delimiters
                    text = text_parts[0] + end_of_input_str + " " + question_delim[0] + " " + f" {ANSWER_START}ANSWER: {answer} {ANSWER_END} " + text_parts[1]

                new_row[col] = text

            df_with_delims = pd.concat([df_with_delims, pd.DataFrame([new_row])], ignore_index=True)

    
    else: 
        raise ValueError(f"Invalid mode: {mode}")
    
    return df_with_delims

    # INSERT_YOUR_CODE

def get_budget_sample_completion_metrics(SAVE_DATA,config):

    token_budgets_with_data = list(SAVE_DATA.keys())
    num_samples = config.num_samples if hasattr(config, "num_samples") else config.num_samples
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

    for tb_idx, token_budget in enumerate(token_budgets_with_data):
        scores_metric = SAVE_DATA[token_budget]['metrics']['answer_score']
        probs_metric = SAVE_DATA[token_budget]['metrics']['answer_probability']
        entropy_metric = SAVE_DATA[token_budget]['metrics']['answer_entropy']
        rankings_metric = SAVE_DATA[token_budget]['metrics']['answer_ranking']
        
        for sample_idx in range(n_samples):
            for completion_idx in range(n_completions):
                metric_idx = sample_idx * n_completions + completion_idx
                scores[tb_idx, sample_idx, completion_idx] = scores_metric[metric_idx][-1]
                probs[tb_idx, sample_idx, completion_idx] = probs_metric[metric_idx][-1]
                entropy[tb_idx, sample_idx, completion_idx] = entropy_metric[metric_idx][-1]
                rankings[tb_idx, sample_idx, completion_idx] = rankings_metric[metric_idx][-1]

    return scores, probs, entropy, rankings

def get_samplewise_breakthrough_and_skew(SAVE_DATA, config, plot_log_breakthroughness=False, plot_log_skewness=False, breakthroughness_lims=[0,20],skewness_lims=[-3,3],style_dict=None,plot=True):
    """
    Computes and plots samplewise breakthroughness and skewness for probability and negative entropy.
    Returns the data used for plotting if return_data=True.
    """

    if style_dict is not None:
        old_params = plt.rcParams.copy() 
        plt.rcParams.update(style_dict) 

    token_budgets_with_data = list(SAVE_DATA.keys())
    num_samples = config.num_samples if hasattr(config, "num_samples") else config.num_samples
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

    probs_breakthroughness = utils.breakthrough_score(x, y_probs)
    abs_probs_breakthroughness = np.abs(probs_breakthroughness)
    probs_skewness = utils.differences_skew_score(y_probs)

    negent_breakthroughness = utils.breakthrough_score(x, y_negent)
    abs_negent_breakthroughness = np.abs(negent_breakthroughness)
    negent_skewness = utils.differences_skew_score(y_negent)

    if plot_log_breakthroughness:
        abs_probs_breakthroughness = np.log10(abs_probs_breakthroughness)
        abs_negent_breakthroughness = np.log10(abs_negent_breakthroughness)
    if plot_log_skewness:
        probs_skewness = np.log10(probs_skewness)
        negent_skewness = np.log10(negent_skewness)

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

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(f"{config.model_name} on {config.dataset_name} - num_completions={config.num_completions}")

    if breakthroughness_lims is not None:
        breakthrough_min = breakthroughness_lims[0]
        breakthrough_max = breakthroughness_lims[1]
    else:
        breakthrough_min = min(
            np.min(abs_probs_breakthroughness_finite) if len(abs_probs_breakthroughness_finite) > 0 else 0,
            np.min(abs_negent_breakthroughness_finite) if len(abs_negent_breakthroughness_finite) > 0 else 0,
        )
        breakthrough_max = max(
            np.max(abs_probs_breakthroughness_finite) if len(abs_probs_breakthroughness_finite) > 0 else 1,
            np.max(abs_negent_breakthroughness_finite) if len(abs_negent_breakthroughness_finite) > 0 else 1,
        )

    breakthrough_diff = breakthrough_max - breakthrough_min
    breakthrough_range = (breakthrough_min-breakthrough_diff*0.1, breakthrough_max+breakthrough_diff*0.1)
    breakthrough_bins = np.linspace(breakthrough_min, breakthrough_max, 21)

    if skewness_lims is not None:
        skew_min = skewness_lims[0]
        skew_max = skewness_lims[1]
    else:
        skew_min = min(
        np.min(probs_skewness_finite) if len(probs_skewness_finite) > 0 else 0,
        np.min(negent_skewness_finite) if len(negent_skewness_finite) > 0 else 0,
        )
        skew_max = max(
            np.max(probs_skewness_finite) if len(probs_skewness_finite) > 0 else 1,
            np.max(negent_skewness_finite) if len(negent_skewness_finite) > 0 else 1,
        )

    skew_diff = skew_max - skew_min
    skew_range = (skew_min-skew_diff*0.1, skew_max+skew_diff*0.1)
    skew_bins = np.linspace(skew_min, skew_max, 21)

    ax = axes[0, 0]
    ax.set_title("Probability Breakthroughness")
    ax.hist(
        abs_probs_breakthroughness_finite,
        density=True,
        alpha=0.7,
        color='tab:blue',
        label=f'{"LOG10" if plot_log_breakthroughness else ""} Probability Breakthroughness\n({n_probs}/{n_total} samples)',
        bins=breakthrough_bins,
        range=breakthrough_range,
        edgecolor='black',
    )
    ax.set_ylabel('Density')
    ax.set_xlim(breakthrough_range)
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.set_title("Negative Entropy Breakthroughness")
    ax.hist(
        abs_negent_breakthroughness_finite,
        density=True,
        alpha=0.7,
        color='tab:orange',
        label=f'{"LOG10" if plot_log_breakthroughness else ""} Negative Entropy Breakthroughness\n({n_negent}/{n_total} samples)',
        bins=breakthrough_bins,
        range=breakthrough_range,
        edgecolor='black',
    )
    ax.set_ylabel('Density')
    ax.set_xlim(breakthrough_range)
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.set_title("Probability Skewness")
    ax.hist(
        probs_skewness_finite,
        density=True,
        alpha=0.7,
        color='tab:green',
        label=f'Probability Skewness\n({n_skew_probs}/{n_total} samples)',
        bins=skew_bins,
        range=skew_range,
        edgecolor='black',
    )
    ax.set_ylabel('Density')
    ax.set_xlim(skew_range)
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.set_title("Negative Entropy Skewness")
    ax.hist(
        negent_skewness_finite,
        density=True,
        alpha=0.7,
        color='tab:red',
        label=f'Negative Entropy Skewness\n({n_skew_negent}/{n_total} samples)',
        bins=skew_bins,
        range=skew_range,
        edgecolor='black',
    )
    ax.set_ylabel('Density')
    ax.set_xlim(skew_range)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if plot:
        plt.show()
    else: 
        plt.close()

    return {
        "abs_probs_breakthroughness": abs_probs_breakthroughness,
        "abs_negent_breakthroughness": abs_negent_breakthroughness,
        "probs_skewness": probs_skewness,
        "negent_skewness": negent_skewness,
        "df_probs": df_probs,
        "df_entropy": df_entropy
    }

def get_metric_table(SAVE_DATA, metric_type,mode="full_view", token_budget=None,score_average_type="mode"):
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
    if mode == "completions_view":
        metric_data = SAVE_DATA[token_budget]['metrics'][metric_type]
        # Find all sample_idx and completion_idx
        sample_idxs = set()
        completion_idxs = set()
        for sample_idx, completion_idx, _ in metric_data:
            sample_idxs.add(sample_idx)
            completion_idxs.add(completion_idx)
        sample_idxs = sorted(sample_idxs)
        completion_idxs = sorted(completion_idxs)
        # Create DataFrame
        df = pd.DataFrame(index=sample_idxs, columns=completion_idxs)
        for sample_idx, completion_idx, metric_value in metric_data:
            df.at[sample_idx, completion_idx] = metric_value
        return df

    elif mode == "token_budget_view":
        # For all token_budgets, get mean across completions for each sample
        token_budgets = sorted(SAVE_DATA.keys())
        # Find all sample_idxs
        all_sample_idxs = set()
        for tb in token_budgets:
            metric_data = SAVE_DATA[tb]['metrics'][metric_type]
            for sample_idx, _, _ in metric_data:
                all_sample_idxs.add(sample_idx)
        all_sample_idxs = sorted(all_sample_idxs)
        df = pd.DataFrame(index=all_sample_idxs, columns=token_budgets)
        for tb in token_budgets:
            metric_data = SAVE_DATA[tb]['metrics'][metric_type]
            # Build dict: sample_idx -> list of metric_values (across completions)
            sample_to_values = {}
            for sample_idx, _, metric_value in metric_data:
                if sample_idx not in sample_to_values:
                    sample_to_values[sample_idx] = []
                sample_to_values[sample_idx].append(metric_value)
            for sample_idx in all_sample_idxs:
                values = sample_to_values.get(sample_idx, [])
                
                if metric_type == "answer_ranking" or metric_type == "answer_score":
                    if metric_type == "answer_score":
                        if score_average_type=="mode":
                            average_value = stats.mode(values).mode if values else np.nan
                        elif score_average_type=="mean":
                            average_value = np.mean(values) if values else np.nan
                    else:
                        average_value = stats.mode(values).mode if values else np.nan
                else: 
                    average_value = np.mean(values) if values else np.nan

                df.at[sample_idx, tb] = average_value
        return df

    elif mode == "full_view":
        token_budgets = sorted(SAVE_DATA.keys())
        # Find all sample_idxs
        all_sample_idxs = set()
        for tb in token_budgets:
            metric_data = SAVE_DATA[tb]['metrics'][metric_type]
            for sample_idx, _, _ in metric_data:
                all_sample_idxs.add(sample_idx)
        all_sample_idxs = sorted(all_sample_idxs)
        df = pd.DataFrame(index=all_sample_idxs, columns=token_budgets)
        for tb in token_budgets:
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
                #df.at[sample_idx, tb] = tuple(values)
        return df
    else:
        raise ValueError(f"Unknown mode: {mode}")



def get_token_count_table(config,SAVE_DATA,mode="full_view"):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if mode=="full_view":
        token_budgets = list(SAVE_DATA.keys())
        sample_idxs = list(SAVE_DATA[token_budgets[0]].keys())
        df = pd.DataFrame(index=sample_idxs, columns=token_budgets)
        for tb in token_budgets:
            for sample_idx in SAVE_DATA[tb].keys():
                if sample_idx == 'metrics':
                    continue
                token_counts = []
                for completion_idx in range(config.num_completions):
                    text = SAVE_DATA[tb][sample_idx]['completions'][completion_idx]['text'][0]
                    encoded_text = tokenizer.encode(text, add_special_tokens=False)
                    end_of_input_idxs = utils.workflow_utils.get_end_of_input_idxs(encoded_text, tokenizer)[0]
                    token_count = utils.workflow_utils.count_generated_tokens(encoded_text, end_of_input_idxs, tokenizer)
                    token_counts.append(token_count)
                df.loc[sample_idx, tb] = token_counts
        return df
    else:
        raise ValueError(f"Unknown mode: {mode}")

def get_text_completion(SAVE_DATA,config,token_budgets=[],sample_idxs=[],completion_idxs=[],remove_padding_token=True,padding_token="!"):

    """
    Use when debugging at terminal
    """

    text_completions = [] 

    if len(token_budgets)==0:
        token_budgets = list(list(SAVE_DATA.keys())[0])

    if len(sample_idxs)==0:
        sample_idxs = [0]
    if len(completion_idxs)==0:
        completion_idxs = [0]

    for token_budget in token_budgets:
        for sample_idx in sample_idxs:
            for completion_idx in completion_idxs:
                try:
                    text = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['text'][0]
                except KeyError:
                    text = None
                if remove_padding_token:
                    text = text.replace(padding_token, "")
                text_completions.append(text)


    return text_completions




approx_monotonic_increase_condition = lambda x: (np.all(np.diff(x) >= 0)) or (np.sum(np.diff(x) >= -1e-11) >= len(x) - 2)
approx_monotonic_decrease_condition = lambda x: (np.all(np.diff(x) <= 0)) or (np.sum(np.diff(x) <= 1e-11) >= len(x) - 2)
approx_flatlined_condition = lambda x: (np.all(np.diff(x) == 0)) or (np.sum(np.diff(x) == 0) >= len(x) - 1)

def alternating_condition(x, entropy_threshold=0.5):
    """
    Returns True if the entropy of the distribution over 0s and 1s in x is sufficiently close to 1.0,
    indicating strong alternation, otherwise False.
    """
    x = np.asarray(x)
    if len(x) == 0:
        return False
    x = np.asarray(x, dtype=int)
    counts = np.bincount(x, minlength=2)
    probs = counts / np.sum(counts)


    #entropy condition
    from scipy.stats import entropy
    ent = entropy(probs, base=2)
    entropy_condition = np.isclose(ent, 1.0, atol=1.0 - entropy_threshold)

    #diff condition
    diffs = np.diff(x)
    has_positive = np.any(diffs > 0)
    has_negative = np.any(diffs < 0)
    diff_condition = has_positive and has_negative

    return entropy_condition and diff_condition



def classify_scaling_curves(
    SAVE_DATA,
    config,
    curve_metric_type_map={
        "monotonic_increase": "answer_probability",
        "monotonic_decrease": "answer_probability",
        "flatlined": "answer_score",
        "alternating": "answer_score"
    },
    alternating_entropy_threshold=0.5,
    PLOT=True,
    style_dict=None
):
    """
    Classifies each sample (row) in metric_table into one of four categories:
    - monotonic_increase
    - monotonic_decrease
    - flatlined
    - alternating

    Returns:
        dict with keys as category names and values as lists of sample_idxs
    """
    import numpy as np
    from scipy.stats import entropy

    # Prepare output dict
    clusters = {
        "monotonic_increase": [],
        "monotonic_decrease": [],
        "flatlined": [],
        "alternating": []
    }

    score_table = utils.get_metric_table(SAVE_DATA=SAVE_DATA, mode="token_budget_view",metric_type="answer_score")
    probs_table = utils.get_metric_table(SAVE_DATA=SAVE_DATA, mode="token_budget_view",metric_type="answer_probability")
    negent_table = utils.get_metric_table(SAVE_DATA=SAVE_DATA, mode="token_budget_view",metric_type="answer_entropy")


    # Iterate over all sample indices first, using the tables above
    all_sample_idxs = score_table.index

    for sample_idx in all_sample_idxs:
        score_values = np.array(score_table.loc[sample_idx].values, dtype=float)
        probs_values = np.array(probs_table.loc[sample_idx].values, dtype=float)
        negent_values = np.array(negent_table.loc[sample_idx].values, dtype=float)
        probs_values[np.abs(probs_values) < 1e-8] = 0

        flatlined_condition_check = approx_flatlined_condition(score_values) or np.all(probs_values == 0)
        approx_monotonic_increase_condition_check = approx_monotonic_increase_condition(probs_values)
        approx_monotonic_decrease_condition_check = approx_monotonic_decrease_condition(probs_values)
        alternating_condition_check = alternating_condition(score_values)

        # Skip if any value is nan in any metric table
        if (
            np.any(pd.isnull(score_values)) or
            np.any(pd.isnull(probs_values)) or
            np.any(pd.isnull(negent_values))
        ):
            continue

        # Flatlined
        if flatlined_condition_check:
            clusters['flatlined'].append(sample_idx)

        # Monotonic Increase
        if approx_monotonic_increase_condition_check:
            if sample_idx not in clusters["flatlined"]:
                clusters["monotonic_increase"].append(sample_idx)
            continue

        # Monotonic Decrease
        if approx_monotonic_decrease_condition_check:
            if sample_idx not in clusters["flatlined"]:
                clusters["monotonic_decrease"].append(sample_idx)
            continue

        # Alternating
        if alternating_condition_check:
            clusters['alternating'].append(sample_idx)
        


    #remove duplicates from clusters
    for k in clusters:
        clusters[k] = list(set(clusters[k]))



    if PLOT:
        if len(clusters["monotonic_increase"]) > 0:
            utils.plot_data(
                SAVE_DATA=SAVE_DATA,
                config=config,
                metrics=["answer_score", "answer_probability", "answer_entropy"],
                sample_idxs=clusters["monotonic_increase"],
                fig_title=f"Monotonic Increase - {curve_metric_type_map['monotonic_increase']}",
                style_dict=style_dict
            )
        if len(clusters["monotonic_decrease"]) > 0:
            utils.plot_data(
                SAVE_DATA=SAVE_DATA,
                config=config,
                metrics=["answer_score", "answer_probability", "answer_entropy"],
                sample_idxs=clusters["monotonic_decrease"],
                fig_title=f"Monotonic Decrease - {curve_metric_type_map['monotonic_decrease']}",
                style_dict=style_dict
            )
        if len(clusters["flatlined"]) > 0:
            utils.plot_data(
                SAVE_DATA=SAVE_DATA,
                config=config,
                metrics=["answer_score", "answer_probability", "answer_entropy"],
                sample_idxs=clusters["flatlined"],
                fig_title=f"Flatlined - {curve_metric_type_map['flatlined']}",
                style_dict=style_dict
            )
        if len(clusters["alternating"]) > 0:
            utils.plot_data(
                SAVE_DATA=SAVE_DATA,
                config=config,
                metrics=["answer_score", "answer_probability", "answer_entropy"],
                sample_idxs=clusters["alternating"],
                fig_title=f"Alternating - {curve_metric_type_map['alternating']}",
                style_dict=style_dict
            )

    return (
        clusters["monotonic_increase"],
        clusters["monotonic_decrease"],
        clusters["flatlined"],
        clusters["alternating"]
    )


def plot_metric_correlations(SAVE_DATA, metric_maps, sample_idxs=None, SHOW_PLOTS=True,style_dict=None):

    

    if style_dict is not None:
        old_params = plt.rcParams.copy() 
        plt.rcParams.update(style_dict)

    x_metric = metric_maps["x"]
    y_metric = metric_maps["y"]

    x_metric_table = utils.get_metric_table(SAVE_DATA, metric_type=x_metric, mode="token_budget_view", score_average_type="mean")
    if x_metric == "answer_entropy":
        x_metric_table = -1*x_metric_table #we're interested in negativty entropy
    y_metric_table = utils.get_metric_table(SAVE_DATA, metric_type=y_metric, mode="token_budget_view", score_average_type="mean")
    if y_metric == "answer_entropy":
        y_metric_table = -1*y_metric_table #we're interested in negativty entropy

    if sample_idxs is None:
        sample_idxs = x_metric_table.index

    pearson_corrs = {}
    spearman_corrs = {}

    fig, axes = plt.subplots(len(sample_idxs), 1, figsize=(10, 4 * len(sample_idxs)), squeeze=False)
    for i, sample_idx in enumerate(sample_idxs):
        ax = axes[i, 0]
        x = x_metric_table.loc[sample_idx].values.astype(float)
        y = y_metric_table.loc[sample_idx].values.astype(float)
        ax.scatter(x, y, marker='o', color='blue')
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f"{x_metric} vs {y_metric} for Sample {sample_idx}")
        ax.grid(True)

        # Calculate and store Pearson and Spearman correlations
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) > 1:
            pearson_corr, _ = pearsonr(x[mask], y[mask])
            spearman_corr, _ = spearmanr(x[mask], y[mask])
        else:
            pearson_corr = np.nan
            spearman_corr = np.nan
        pearson_corrs[sample_idx] = pearson_corr
        spearman_corrs[sample_idx] = spearman_corr

        # Set axis limits based on metric type
        if x_metric in ["answer_score", "answer_probability"]:
            ax.set_xlim(-0.1, 1.1)
        elif x_metric == "answer_entropy":
            x_min, x_max = np.nanmin(x), np.nanmax(x)
            margin = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
            ax.set_xlim(x_min - margin, x_max + margin)
        if y_metric in ["answer_score", "answer_probability"]:
            ax.set_ylim(-0.1, 1.1)
        elif y_metric == "answer_entropy":
            y_min, y_max = np.nanmin(y), np.nanmax(y)
            margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
            ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    if SHOW_PLOTS:
        print(SHOW_PLOTS)
        plt.show()

    return pearson_corrs, spearman_corrs


def find_repeating_sequences(SAVE_DATA,config,end_of_input_string,tokenizer,pad_token=None, k=5):
    """
    Find repeating sequences based on frequency of config.answer_indicator_token and config.force_continue.
    Returns:
        - List of dicts with keys: sample_idx, completion_idx, num_answer_indicators, num_force_continues
        - List of top-k dicts with keys: sample_idx, completion_idx, text, num_answer_indicators, num_force_continues
    """
    results = {}
    texts_with_counts = []

    token_budgets = list(SAVE_DATA.keys())

    if pad_token is None:
        pad_token = config.pad_token

    for token_budget in token_budgets:
        results[token_budget] = []
        sample_idxs = list(SAVE_DATA[token_budget].keys())
        for sample_idx in sample_idxs:
            if sample_idx == 'metrics':
                continue
            for completion_idx in range(config.num_completions):
                text = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['text']
                if isinstance(text, list):
                    text = text[0]
                text_without_pad = text.replace(pad_token,"")
                generated_text_only = text_without_pad.split(end_of_input_string)[1]
                # If text is a list, get the first element
                number_of_generated_tokens = len(tokenizer.encode(generated_text_only, add_special_tokens=False))
                
                num_answer_indicators = generated_text_only.count(config.answer_indicator_token)
                num_force_continues = generated_text_only.count(config.force_continue)
                results[token_budget].append({
                    'sample_idx': sample_idx, 
                    'completion_idx': completion_idx, 
                    'num_answer_indicators': num_answer_indicators, 
                    'num_force_continues': num_force_continues,
                    'n_generated_tokens': number_of_generated_tokens
                })
                texts_with_counts.append({
                    'sample_idx': sample_idx, 
                    'completion_idx': completion_idx, 
                    'text': generated_text_only, 
                    'num_answer_indicators': num_answer_indicators,
                    'num_force_continues': num_force_continues,
                    'n_generated_tokens': number_of_generated_tokens
                })

    # Sort by num_answer_indicators descending
    top_k_by_answer_indicators = sorted(texts_with_counts, key=lambda x: x['num_answer_indicators'], reverse=True)[:k]
    top_k_by_force_continues = sorted(texts_with_counts, key=lambda x: x['num_force_continues'], reverse=True)[:k]


    return results, top_k_by_answer_indicators, top_k_by_force_continues

class AnalysisWorkflow:

    def __init__(self,paths, results_save_dir,scaling_config=None):

        self.paths = paths

        self.model_names = [] 
        self.dataset_names = []

        self.gpqa_paths = []
        self.aime24_paths = []
        self.aime25_paths = []
        self.deepseek_paths = []
        self.qwen_paths = []
        self.gemma_paths = []
        self.qwq_paths = []
        self.phi_paths = []
        
        if scaling_config is not None:
            self.scaling_config = scaling_config
        else:
            self.scaling_config = ("deepseek","gpqa","32B")

        for path in self.paths:
            lower_path = path.lower()

            # Avoid repetitions in model_names and dataset_names
            if "deepseek" in lower_path and "deepseek" not in self.model_names:
                self.model_names.append("deepseek")
            elif "qwen2.5" in lower_path and "qwen2.5" not in self.model_names:
                self.model_names.append("qwen2.5")
            elif "gemma" in lower_path and "gemma" not in self.model_names:
                self.model_names.append("gemma")
            elif "qwq" in lower_path and "qwq" not in self.model_names:
                self.model_names.append("qwq")
            elif "phi" in lower_path and "phi" not in self.model_names:
                self.model_names.append("phi")

            if "gpqa" in lower_path and "gpqa" not in self.dataset_names:
                self.dataset_names.append("gpqa")
            elif "aime_2024" in lower_path and "aime_2024" not in self.dataset_names:
                self.dataset_names.append("aime_2024")
            elif "aime25" in lower_path and "aime25" not in self.dataset_names:
                self.dataset_names.append("aime25")

            if "gpqa" in lower_path:
                self.gpqa_paths.append(path)
            elif "aime_2024" in lower_path:
                self.aime24_paths.append(path)
            elif "aime25" in lower_path:
                self.aime25_paths.append(path)

            if "qwen2.5" in lower_path:
                self.qwen_paths.append(path)
            elif "gemma" in lower_path:
                self.gemma_paths.append(path)
            elif "qwq" in lower_path:
                self.qwq_paths.append(path)
            elif "phi" in path:
                self.phi_paths.append(path)

        self.scaling_paths = self._find_scaling_paths()
        self.scaling_paths = sorted(self.scaling_paths, key=lambda path: utils.get_model_size(path))

        self.non_scaling_paths = []
        for path in self.paths:
            if path not in self.scaling_paths:
                self.non_scaling_paths.append(path)
        for path in self.paths:
            if all ([part.lower() in path.lower() for part in self.scaling_config]):
                self.non_scaling_paths.append(path)

        self.results_save_dir = results_save_dir
        os.makedirs(self.results_save_dir, exist_ok=True)
        
        self.STYLE_DICT_MAPPINGS = {} #can figure this out later. should be quite simple (?)

        self.kde_bw_adjust = 1.0

    def _find_scaling_paths(self):
        self.scaling_paths = []
        for path in self.paths:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            if self.scaling_config[0] in model_name.lower() and self.scaling_config[1] in dataset_name.lower():
                self.scaling_paths.append(path)
        return self.scaling_paths

    def get_selected_path(self,model_name,dataset_name,model_size=None):

        selected_paths = []
        for path in self.paths:
            path_lower = path.lower()
            match = True
            if model_name is not None:
                if model_name.lower() not in path_lower:
                    match = False
            if dataset_name is not None:
                if dataset_name.lower() not in path_lower:
                    match = False
            if model_size is not None:
                if str(model_size).lower() not in path_lower:
                    match = False
            if match:
                selected_paths.append(path)
        if not selected_paths:
            raise ValueError("No paths found matching the provided criteria (model_name, dataset_name, model_size).")
        
        # handle case for scaling config
        if self.scaling_config[0] in model_name.lower() and self.scaling_config[1] in dataset_name.lower():
            main_scale = self.scaling_config[2]
            selected_paths = [path for path in selected_paths if main_scale in path]
            selected_path = selected_paths[0]
        else:
            selected_path = selected_paths[0]


        return selected_path

    def get_path_model_and_dataset(self,path):

        leaf = os.path.basename(os.path.normpath(path))
        model_snip,dataset_snip = leaf.split("_")[0].lower(),"_".join(leaf.split("_")[1:]).lower()

        if "aime25" in dataset_snip: 
            dataset_name = "aime25"
        elif "aime_2024" in dataset_snip: 
            dataset_name = "aime_2024"
        elif "gpqa" in dataset_snip: 
            dataset_name = "gpqa"
        else: 
            dataset_name = dataset_snip

        if "deepseek" in model_snip:
            model_name = "deepseek"
        elif "qwen2.5" in model_snip:
            model_name = "qwen2.5"
        elif "gemma" in model_snip:
            model_name = "gemma"
        elif "qwq" in model_snip:
            model_name = "qwq"
        elif "phi" in model_snip:
            model_name = "phi"
        else:
            model_name = model_snip
        
        return model_name, dataset_name

    def aggregate_plots(self, SHOW=True,SAVE=False,log_probs_aime=False,model_name=None,dataset_name=None):
        
        plt.rcParams.update(utils.top_k_style)


        for path in self.non_scaling_paths:

            if model_name is not None:
                if model_name.lower() not in path.lower():
                    continue
            if dataset_name is not None:
                if dataset_name.lower() not in path.lower():
                    continue


            config, SAVE_DATA = utils.read_save_data(path)
            fig_title = f"{config.dataset_name} - {config.model_name} - {config.num_samples} samples - {config.num_completions} completions (Aggregate)"
            plot_model_name = config.model_name.split("/")[-1]
            plot_dataset_name = config.dataset_name.split("/")[-1]
            fig_title = f" {plot_dataset_name} - {plot_model_name} - {config.num_samples} samples"

            if "aime" in plot_dataset_name.lower() and log_probs_aime:
                logy=True
            else:
                logy=False

            if SHOW:
                plot_aggregate_data(SAVE_DATA, config, title=fig_title,logy=logy)
            if SAVE:
                os.makedirs(os.path.join(self.results_save_dir, "aggregate_plots"), exist_ok=True)
                save_name = os.path.basename(path).replace(".pkl", "") + "_aggregate.png"
                save_path = os.path.join(self.results_save_dir, "aggregate_plots", save_name)
                plot_aggregate_data(SAVE_DATA, config, title=fig_title,save_path=save_path,logy=logy)


    def plot_correlation_heatmap(self, correlation_type="pearson", metric_pair="score_prob",method="1D" ,agg_func=np.nanmean, figsize=(8,6), cmap="coolwarm", annot=True, vmin=-1, vmax=1, title=None, SHOW=True,SAVE=False):
        """
        Plots a heatmap of correlations (pearson or spearman) between models (x-axis) and datasets (y-axis).
        Args:
            correlation_type: "pearson" or "spearman"
            metric_pair: "score_prob" or "score_entropy"
            agg_func: function to aggregate samplewise correlations (default: np.nanmean)
            figsize: tuple for figure size
            cmap: colormap for heatmap
            annot: whether to annotate cells
            vmin, vmax: value limits for colormap
            title: plot title
            SHOW: whether to show the plot
            SAVE: whether to save the plot
        """

        assert metric_pair in ["score_prob", "score_entropy"]
        assert method in ["1D", "2D", "3D"]
        
        plt.rcParams.update(utils.heatmap_config)

        # Collect all unique model and dataset names
        model_names = []
        dataset_names = []
        heatmap_data = {}

        for path in self.non_scaling_paths:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            if model_name not in model_names:
                model_names.append(model_name)
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            correlation_results = self.correlation_scores(model_name=model_name, dataset_name=dataset_name, method=method)
            
            
            if method not in ["2D","3D"]:
                corr_arr = correlation_results[metric_pair][correlation_type]
                agg_corr = corr_arr

            else:  
                # Aggregate correlation for the metric_pair and correlation_type
                corr_arr = correlation_results[metric_pair][correlation_type]
                agg_corr = agg_func(corr_arr)

            if np.round(agg_corr, 2) == 1.0:
                agg_corr = 0.99
            
            heatmap_data[(dataset_name, model_name)] = agg_corr



        # Sort for consistent axis order
        model_names = sorted(model_names)
        dataset_names = sorted(dataset_names)
        remapped_dataset_name_map = {
            "aime25": "AIME25",
            "gpqa": "GPQA",
            "aime_2024": "AIME24",
        }
        remapped_model_name_map = {
            "deepseek": "Deepseek-R1-Distill",
            "qwen2.5": "Qwen-2.5",
            "gemma": "Gemma-2",
            "qwq": "QwQ-32B",
            "phi": "Phi-4-Reasoning-Plus",
        }

        # Build heatmap matrix
        heatmap_matrix = np.full((len(dataset_names), len(model_names)), np.nan)
        for i, dataset in enumerate(dataset_names):
            for j, model in enumerate(model_names):
                if (dataset, model) in heatmap_data:
                    heatmap_matrix[i, j] = heatmap_data[(dataset, model)]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(heatmap_matrix, cmap=cmap, vmin=vmin, vmax=vmax)


        # Set axis labels
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(len(dataset_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_yticklabels(dataset_names)

        # Relabel x/y axis labels with remapped namesxw
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(len(dataset_names)))
        ax.set_xticklabels([remapped_model_name_map.get(m.lower(), m) for m in model_names], rotation=45, ha="right")
        ax.set_yticklabels([remapped_dataset_name_map.get(d.lower(), d) for d in dataset_names])


        # Annotate cells
        if annot:
            for i in range(len(dataset_names)):
                for j in range(len(model_names)):
                    val = heatmap_matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if title is None:
            title = f"{metric_pair.replace('_', ' vs. ').title()} ({correlation_type.capitalize()})"
        ax.set_title(title)
        plt.tight_layout()

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "correlation_heatmap"), exist_ok=True)
            save_name = "correlation_heatmap.png"
            save_path = os.path.join(self.results_save_dir, "correlation_heatmap", save_name)
            plt.savefig(save_path,bbox_inches='tight')
        if SHOW:
            plt.show()

        return heatmap_matrix

    def emergence_scores(self,path):


        config, SAVE_DATA = utils.read_save_data(path)

        emergence_metric_keys = ["abs_probs_breakthroughness","abs_negent_breakthroughness","probs_skewness","negent_skewness"]

        emergence_metric_data  = get_samplewise_breakthrough_and_skew(SAVE_DATA,config,plot=False)

        emergence_metric_data = {key: emergence_metric_data[key] for key in emergence_metric_keys}


        return emergence_metric_data

    def correlation_scores(self, model_name, dataset_name,method="1D"):

        if method == "1D":

            tables = self.get_metric_tables(model_name=model_name,dataset_name=dataset_name,mode="token_budget_view")
            score_table, prob_table, entropy_table, ranking_table = tables["answer_score"], tables["answer_probability"], tables["answer_entropy"], tables["answer_ranking"]
            score_matrix, prob_matrix, entropy_matrix, ranking_matrix = utils.table_to_2d_matrix(score_table), utils.table_to_2d_matrix(prob_table), utils.table_to_2d_matrix(entropy_table), utils.table_to_2d_matrix(ranking_table)
            negative_entropy_matrix = -1*entropy_matrix

            score_1d_arr, prob_1d_arr, entropy_1d_arr = score_matrix.mean(axis=0), prob_matrix.mean(axis=0), entropy_matrix.mean(axis=0)
            negentropy_1d_arr = -1*entropy_1d_arr



            correlation_results = {
                "score_prob": {
                    "pearson": pearsonr(score_1d_arr, prob_1d_arr)[0],
                    "spearman": spearmanr(score_1d_arr, prob_1d_arr)[0]
                },
                "score_entropy": {
                    "pearson": pearsonr(score_1d_arr, negentropy_1d_arr)[0],
                    "spearman": spearmanr(score_1d_arr, negentropy_1d_arr)[0]
                },
            }

            

        if method == "2D":
            tables = self.get_metric_tables(model_name=model_name,dataset_name=dataset_name,mode="token_budget_view")
            score_table, prob_table, entropy_table, ranking_table = tables["answer_score"], tables["answer_probability"], tables["answer_entropy"], tables["answer_ranking"]
            score_matrix, prob_matrix, entropy_matrix, ranking_matrix = utils.table_to_2d_matrix(score_table), utils.table_to_2d_matrix(prob_table), utils.table_to_2d_matrix(entropy_table), utils.table_to_2d_matrix(ranking_table)
            negative_entropy_matrix = -1*entropy_matrix
            n_samples, n_budgets = score_matrix.shape

            correlation_results = {
            "score_prob": {
                "pearson": np.full((n_samples, ), np.nan),
                "spearman": np.full((n_samples, ), np.nan)
            },
            "score_entropy": {
                "pearson": np.full((n_samples, ), np.nan),
                "spearman": np.full((n_samples, ), np.nan)
                    },
            }

             # Score vs Probability
            for i in range(n_samples):
                x = score_matrix[i, :]
                y_prob = prob_matrix[i, :]
                mask_prob = ~np.isnan(x) & ~np.isnan(y_prob)
                if np.sum(mask_prob) > 1:
                    correlation_results["score_prob"]["pearson"][i], _ = pearsonr(x[mask_prob], y_prob[mask_prob])
                    correlation_results["score_prob"]["spearman"][i], _ = spearmanr(x[mask_prob], y_prob[mask_prob])

            # Score vs negative entropy
            for i in range(n_samples):
                x = score_matrix[i, :]
                y_entropy = negative_entropy_matrix[i, :]
                mask_entropy = ~np.isnan(x) & ~np.isnan(y_entropy)
                if np.sum(mask_entropy) > 1:
                    correlation_results["score_entropy"]["pearson"][i], _ = pearsonr(x[mask_entropy], y_entropy[mask_entropy])
                    correlation_results["score_entropy"]["spearman"][i], _ = spearmanr(x[mask_entropy], y_entropy[mask_entropy])


        elif method == "3D":
            tables = self.get_metric_tables(model_name=model_name,dataset_name=dataset_name,mode="full_view")
            score_table, prob_table, entropy_table, ranking_table = tables["answer_score"], tables["answer_probability"], tables["answer_entropy"], tables["answer_ranking"]
            score_matrix, prob_matrix, entropy_matrix, ranking_matrix = utils.table_to_3d_matrix(score_table), utils.table_to_3d_matrix(prob_table), utils.table_to_3d_matrix(entropy_table), utils.table_to_3d_matrix(ranking_table)
            negative_entropy_matrix = -1*entropy_matrix
            n_samples, n_completions, n_budgets = score_matrix.shape

            correlation_results = {
            "score_prob": {
                "pearson": np.full((n_samples, n_completions), np.nan),
                "spearman": np.full((n_samples, n_completions), np.nan)
            },
            "score_entropy": {
                "pearson": np.full((n_samples, n_completions), np.nan),
                "spearman": np.full((n_samples, n_completions), np.nan)
                    },
            }

                # Score vs Probability
            for i in range(n_samples):
                for j in range(n_completions):
                    x = score_matrix[i, j, :]
                    y_prob = prob_matrix[i, j, :]
                    mask_prob = ~np.isnan(x) & ~np.isnan(y_prob)
                    if np.sum(mask_prob) > 1:
                        correlation_results["score_prob"]["pearson"][i, j], _ = pearsonr(x[mask_prob], y_prob[mask_prob])
                        correlation_results["score_prob"]["spearman"][i, j], _ = spearmanr(x[mask_prob], y_prob[mask_prob])

            # Score vs Entropy
            for i in range(n_samples):
                for j in range(n_completions):
                    x = score_matrix[i, j, :]
                    y_entropy = negative_entropy_matrix[i, j, :]
                    mask_entropy = ~np.isnan(x) & ~np.isnan(y_entropy)
                    if np.sum(mask_entropy) > 1:
                        correlation_results["score_entropy"]["pearson"][i, j], _ = pearsonr(x[mask_entropy], y_entropy[mask_entropy])
                        correlation_results["score_entropy"]["spearman"][i, j], _ = spearmanr(x[mask_entropy], y_entropy[mask_entropy])



        return correlation_results

    def emergence_score_summary_stats_across_datasets(self,model_name):

        selected_paths = []
        for dataset in self.dataset_names:
            selected_paths.append(self.get_selected_path(model_name=model_name, dataset_name=dataset))


        EMERGENCE_METRIC_DATA = {}
        emergence_score_types = ["abs_probs_breakthroughness","abs_negent_breakthroughness","probs_skewness","negent_skewness"]
        metric_names = ["Breakthroughness (p)","Breakthroughness (-H)","Skewness (p)","Skewness (-H)"]
        stats_dict = {}
        for path in selected_paths:
            config, SAVE_DATA = utils.read_save_data(path)
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            emergence_metric_data = get_samplewise_breakthrough_and_skew(SAVE_DATA, config, plot=False)
            EMERGENCE_METRIC_DATA[dataset_name] = emergence_metric_data
            stats_dict[dataset_name] = {
                metric_type: (
                    float(f"{np.nanmean(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanstd(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanmedian(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanpercentile(emergence_metric_data[metric_type], 75) - np.nanpercentile(emergence_metric_data[metric_type], 25):.3g}")
                )
                for metric_type in emergence_score_types
            }
            
        # Create a DataFrame: rows = dataset names, columns = metric_names, cells = tuple (median, IQR) or (mean, stddev)
        data = {}
        for idx, metric_type in enumerate(emergence_score_types):
            metric_name = metric_names[idx]
            col = []
            for dataset in stats_dict.keys():
                vals = stats_dict[dataset][metric_type]
                if "breakthroughness" in metric_name.lower():
                    col.append((vals[2], vals[3]))  # (median, IQR)
                elif "skewness" in metric_name.lower():
                    col.append((vals[0], vals[1]))  # (mean, stddev)
                else:
                    col.append((np.nan, np.nan))
            data[metric_name] = col
        summary_stats_df = pd.DataFrame(data, index=list(stats_dict.keys()))

        return summary_stats_df

    def emergence_score_summary_stats_across_models(self,dataset_name):

        selected_paths = []
        for model in self.model_names:
            selected_paths.append(self.get_selected_path(model_name=model, dataset_name=dataset_name))

        EMERGENCE_METRIC_DATA = {}
        emergence_score_types = ["abs_probs_breakthroughness","abs_negent_breakthroughness","probs_skewness","negent_skewness"]
        metric_names = ["Breakthroughness (p)","Breakthroughness (-H)","Skewness (p)","Skewness (-H)"]
        stats_dict = {}
        for path in selected_paths:
            config, SAVE_DATA = utils.read_save_data(path)
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            emergence_metric_data = get_samplewise_breakthrough_and_skew(SAVE_DATA, config, plot=False)
            EMERGENCE_METRIC_DATA[model_name] = emergence_metric_data
            stats_dict[model_name] = {
                metric_type: (
                    float(f"{np.nanmean(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanstd(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanmedian(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanpercentile(emergence_metric_data[metric_type], 75) - np.nanpercentile(emergence_metric_data[metric_type], 25):.3g}")
                )
                for metric_type in emergence_score_types
            }
        
        # Create a DataFrame: rows = model names, columns = metric_names, cells = tuple (median, IQR) or (mean, stddev)
        data = {}
        for idx, metric_type in enumerate(emergence_score_types):
            metric_name = metric_names[idx]
            col = []
            for model in stats_dict.keys():
                vals = stats_dict[model][metric_type]
                if "breakthroughness" in metric_name.lower():
                    col.append((vals[2], vals[3]))  # (median, IQR)
                elif "skewness" in metric_name.lower():
                    col.append((vals[0], vals[1]))  # (mean, stddev)
                else:
                    col.append((np.nan, np.nan))
            data[metric_name] = col
        summary_stats_df = pd.DataFrame(data, index=list(stats_dict.keys()))

        return summary_stats_df

    def emergence_score_dist_across_datasets(self,model_name,SHOW=True,SAVE=False,breakthroughness_lims=None,skewness_lims=None):
        
        plt.rcParams.update(utils.aggregate_plot_config)

        PLOT_DATA = {}
        for dataset in self.dataset_names:
            selected_path = self.get_selected_path(model_name=model_name, dataset_name=dataset)
            emergence_scores = self.emergence_scores(path=selected_path)
            PLOT_DATA[dataset] = emergence_scores

        sns.set_palette("husl")

        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        axs = axs.flatten()
        fig.suptitle(f'Emergence score distributions - {model_name}', y=0.98)

        metric_types = ['abs_probs_breakthroughness', 'abs_negent_breakthroughness', 'probs_skewness', 'negent_skewness']
        datasets = ['aime25', 'gpqa', 'aime_2024']
        metric_name_map = {
            'abs_probs_breakthroughness': 'Breakthroughness (p)',
            'abs_negent_breakthroughness': 'Breakthroughness (-H)',
            'probs_skewness': 'Skewness (p)',
            'negent_skewness': 'Skewness (-H)'
        }
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        line_styles = ['-', '--', '-.']

        for i, metric in enumerate(metric_types):
            ax = axs[i]
            ax.set_facecolor('#fafafa')

            if "breakthroughness" in metric: 
                clip = (0,None)
            elif "skewness" in metric:
                clip = (None,None)
            else:
                clip = (None,None)

            for j, dataset in enumerate(datasets):
                try:
                    values = np.array(PLOT_DATA[dataset][metric])
                    values = values[np.isfinite(values)]
                    if len(values) > 1:
                        sns.kdeplot(
                            values, ax=ax, label=dataset,
                            color=colors[j], alpha=0.15, linewidth=3,
                            linestyle=line_styles[j], fill=True, bw_adjust=self.kde_bw_adjust,clip=clip)
                except Exception as e:
                    print(f"Error plotting {dataset}, {metric}: {e}")

            title = metric_name_map[metric]
            ax.set_title(title, pad=15)
            ax.set_xlabel('Value')
            if i == 0:
                ax.set_ylabel('Density')
            
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#cccccc')
            if "breakthroughness" in metric:
                ax.set_xlim(breakthroughness_lims)
            elif "skewness" in metric:
                ax.set_xlim(skewness_lims)

        # Get legend handles and labels from the first subplot after all plotting is done
        handles, labels = axs[0].get_legend_handles_labels()

        # Remove legends from all subplots
        for ax in axs:
            legend = ax.get_legend()
            if legend:
                legend.remove()

        # Add single legend at the top
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.10), 
                ncol=len(datasets), frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "emergence_score_dist_across_datasets"), exist_ok=True)
            save_name = f"{model_name}_emergence_score_dist_across_datasets.png"
            save_path = os.path.join(self.results_save_dir, "emergence_score_dist_across_datasets", save_name)
            plt.draw()
            plt.savefig(save_path,bbox_inches='tight')
        if SHOW:
            plt.show()

        return PLOT_DATA

    def emergence_score_dist_across_models(self, dataset_name, SHOW=True, SAVE=True,breakthroughness_lims=None,skewness_lims=None):

        """
        Plot KDE distributions of emergence scores across different models for a given dataset.
        """

        plt.rcParams.update(utils.aggregate_plot_config)

        PLOT_DATA = {}
        for model_name in self.model_names:
            path = self.get_selected_path(model_name=model_name, dataset_name=dataset_name)
            emergence_scores = self.emergence_scores(path=path)
            PLOT_DATA[model_name] = emergence_scores

        #plt.style.use('default')
        sns.set_palette("husl")

        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        axs = axs.flatten()
        fig.suptitle(f'Emergence score distributions - {dataset_name}', y=0.98)

        metric_types = ['abs_probs_breakthroughness', 'abs_negent_breakthroughness', 'probs_skewness', 'negent_skewness']
        models = self.model_names
        metric_name_map = {
            'abs_probs_breakthroughness': 'Breakthroughness (p)',
            'abs_negent_breakthroughness': 'Breakthroughness (-H)',
            'probs_skewness': 'Skewness (p)',
            'negent_skewness': 'Skewness (-H)'
        }

        # Better color palette with more contrast
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e', '#1abc9c']
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

        for i, metric in enumerate(metric_types):
            ax = axs[i]
            ax.set_facecolor('#fafafa')

            if "breakthroughness" in metric: 
                clip = (0,None)
            elif "skewness" in metric:
                clip = (None,None)
            else:
                clip = (None,None)

            for j, model in enumerate(models):
                try:
                    values = np.array(PLOT_DATA[model][metric])
                    values = values[np.isfinite(values)]
                    if len(values) > 1:
                        sns.kdeplot(
                            values, ax=ax, label=model,
                            color=colors[j % len(colors)], alpha=0.15, linewidth=3,
                            linestyle=line_styles[j % len(line_styles)], fill=True, bw_adjust=self.kde_bw_adjust,clip=clip)
                except Exception as e:
                    print(f"Error plotting {model}, {metric}: {e}")

            title = metric.replace('_', ' ').replace('abs ', 'Absolute ').title()
            title = metric_name_map[metric]
            ax.set_title(title, pad=15)
            ax.set_xlabel('Value')
            if i == 0:
                ax.set_ylabel('Density')
            
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#cccccc')

            #enforce limits 
            if "breakthroughness" in metric:
                ax.set_xlim(breakthroughness_lims)
            elif "skewness" in metric:
                ax.set_xlim(skewness_lims)

        # Get legend handles and labels from the first subplot after all plotting is done
        handles, labels = axs[0].get_legend_handles_labels()

        # Remove legends from all subplots
        for ax in axs:
            legend = ax.get_legend()
            if legend:
                legend.remove()

        # Add single legend at the bottom center
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.10), 
                   ncol=len(models), frameon=True, fancybox=True, shadow=True, title='')

        plt.tight_layout()

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "emergence_score_dist_across_models"), exist_ok=True)
            save_name = f"{dataset_name}_emergence_score_dist_across_models.png"
            save_path = os.path.join(self.results_save_dir, "emergence_score_dist_across_models", save_name)
            plt.draw()
            plt.savefig(save_path,bbox_inches='tight')
        if SHOW:
            plt.show()

        return PLOT_DATA
        
    def get_top_k_max_emergence_samples(self,model_name,dataset_name,k=4,plot=False,save_plots=False,aggregate_with="probability"):

        """
        Get the top k samples with the highest emergence scores for a given model and dataset.

        TO DO: check this. I've just copied in code from Claude.

        """

        plt.rcParams.update(utils.aggregate_plot_config)

        path = self.get_selected_path(model_name=model_name,dataset_name=dataset_name)
        config,SAVE_DATA = utils.read_save_data(path)




        emergence_metric_data = self.emergence_scores(path=path)
        metric_names = ['abs_probs_breakthroughness', 'abs_negent_breakthroughness', 'probs_skewness', 'negent_skewness']
        metric_matrix = np.vstack([emergence_metric_data[name] for name in metric_names])

        rankings = self.get_emergence_score_rankings(model_name=model_name,dataset_name=dataset_name,aggregate_with=aggregate_with)

        # Find the sample(s) with the lowest  rank (highest emergence)
        top_k_sample_indices = rankings[:k]


        if plot:
            plot_data(SAVE_DATA=SAVE_DATA,config=config,sample_idxs=top_k_sample_indices,fig_title=f"Top {k} Samples with Highest Emergence Scores",style_dict=utils.top_k_style)
        if save_plots:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            os.makedirs(os.path.join(self.results_save_dir, f"top_{k}_samples_{model_name}_{dataset_name}"), exist_ok=True)
            save_name = os.path.basename(path).replace(".pkl", "") + f"_top_{k}_samples.png"
            save_path = os.path.join(self.results_save_dir, f"top_{k}_samples_{model_name}_{dataset_name}", save_name)
            print(save_path)

            plot_data(SAVE_DATA=SAVE_DATA,config=config,sample_idxs=top_k_sample_indices,fig_title=f"Top {k} Samples with Highest Emergence Scores",save_path=save_path,style_dict=utils.top_k_style)


        return top_k_sample_indices

    def emergence_score_summary_stats_across_scales(self):

        plt.rcParams.update(utils.aggregate_plot_config)

        selected_paths = self.scaling_paths
        EMERGENCE_METRIC_DATA = {}
        emergence_score_types = ["abs_probs_breakthroughness","abs_negent_breakthroughness","probs_skewness","negent_skewness"]
        metric_names = ["Breakthroughness (p)","Breakthroughness (-H)","Skewness (p)","Skewness (-H)"]
        stats_dict = {}
        for path in selected_paths:
            config, SAVE_DATA = utils.read_save_data(path)
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            model_size = utils.get_model_size(path)
            emergence_metric_data = get_samplewise_breakthrough_and_skew(SAVE_DATA, config, plot=False)
            EMERGENCE_METRIC_DATA[model_size] = emergence_metric_data
            stats_dict[model_size] = {
                metric_type: (
                    float(f"{np.nanmean(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanstd(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanmedian(emergence_metric_data[metric_type]):.3g}"),
                    float(f"{np.nanpercentile(emergence_metric_data[metric_type], 75) - np.nanpercentile(emergence_metric_data[metric_type], 25):.3g}")
                )
                for metric_type in emergence_score_types
            }
        # Create a DataFrame: rows = model sizes, columns = metric_names, cells = tuple (median, IQR) or (mean, stddev)
        data = {}
        for idx, metric_type in enumerate(emergence_score_types):
            metric_name = metric_names[idx]
            col = []
            for model_size in stats_dict.keys():
                vals = stats_dict[model_size][metric_type]
                if "breakthroughness" in metric_name.lower():
                    col.append((vals[2], vals[3]))  # (median, IQR)
                elif "skewness" in metric_name.lower():
                    col.append((vals[0], vals[1]))  # (mean, stddev)
                else:
                    col.append((np.nan, np.nan))
            data[metric_name] = col
        summary_stats_df = pd.DataFrame(data, index=list(stats_dict.keys()))


        return summary_stats_df

    def emergence_scores_dist_across_scales(self, SHOW=True, SAVE=False, breakthroughness_lims=[-20, 50], skewness_lims=[-4, 4]):
        """
        Plot KDE distributions of emergence scores across different model sizes for a given scaling config.
        Each axes shows a metric, and each curve is a model size.
        """

        plt.rcParams.update(utils.aggregate_plot_config)

        PLOT_DATA = {}
        model_sizes = []
        for path in self.scaling_paths:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            model_size = utils.get_model_size(path)
            model_sizes.append(model_size)
            emergence_scores = self.emergence_scores(path=path)
            PLOT_DATA[model_size] = emergence_scores

        # Sort model sizes for consistent plotting
        model_sizes = sorted(list(set(model_sizes)))
        metric_types = ['abs_probs_breakthroughness', 'abs_negent_breakthroughness', 'probs_skewness', 'negent_skewness']
        metric_name_map = {
            'abs_probs_breakthroughness': 'Breakthroughness (p)',
            'abs_negent_breakthroughness': 'Breakthroughness (-H)',
            'probs_skewness': 'Skewness (p)',
            'negent_skewness': 'Skewness (-H)'
        }

        # Better color palette with more contrast
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e', '#1abc9c']
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

        sns.set_palette("husl")

        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        axs = axs.flatten()
        fig.suptitle('Emergence score distributions - Deepseek-R1-Distill', y=0.98)


        for i, metric in enumerate(metric_types):
            ax = axs[i]
            ax.set_facecolor('#fafafa')

            if "breakthroughness" in metric: 
                clip = (0,None)
            elif "skewness" in metric:
                clip = (None,None)
            else:
                clip = (None,None)

            for j, model_size in enumerate(model_sizes):
                try:
                    values = np.array(PLOT_DATA[model_size][metric])
                    values = values[np.isfinite(values)]
                    if len(values) > 1:
                        sns.kdeplot(
                            values, ax=ax, label=f"{model_size}B",
                            color=colors[j % len(colors)], alpha=0.15, linewidth=3,
                            linestyle=line_styles[j % len(line_styles)], fill=True, bw_adjust=self.kde_bw_adjust,clip=clip
                        )
                except Exception as e:
                    print(f"Error plotting {model_size}, {metric}: {e}")

            title = metric.replace('_', ' ').replace('abs ', 'Absolute ').title()
            title = metric_name_map[metric]
            ax.set_title(title, pad=15)
            #ax.set_xlabel('Value')
            if i == 0:
                ax.set_ylabel('Density')
            
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#cccccc')

            #enforce limits 
            if "breakthroughness" in metric:
                ax.set_xlim(breakthroughness_lims)
            elif "skewness" in metric:
                ax.set_xlim(skewness_lims)

        # Get legend handles and labels from the first subplot after all plotting is done
        handles, labels = axs[0].get_legend_handles_labels()

        # Remove legends from all subplots
        for ax in axs:
            legend = ax.get_legend()
            if legend:
                legend.remove()

        # Add single legend at the bottom center
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.10), 
                   ncol=len(model_sizes), frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "emergence_score_dist_across_scales"), exist_ok=True)
            save_name = f"emergence_score_dist_across_scales.png"
            save_path = os.path.join(self.results_save_dir, "emergence_score_dist_across_scales", save_name)
            plt.draw()
            plt.savefig(save_path,bbox_inches='tight')
        if SHOW:
            plt.show()

        return PLOT_DATA

    def plot_scaling_curves(self,SHOW=True,SAVE=False,SINGLE_PLOT=False):

        PLOT_DATA = {}

        if not SINGLE_PLOT:
            plt.rcParams.update(utils.scaling_curves_2x2_style)
        else:
            plt.rcParams.update(utils.scaling_curves_style)

        for path in self.scaling_paths:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            model_size = get_model_size(path)
            config, SAVE_DATA = utils.read_save_data(path)
            emergence_metric_data = self.emergence_scores(path=path)
            scale_summary_stats = {}
            for metric_name, arr in emergence_metric_data.items():
                arr = np.array(arr)
                minval, maxval = np.nanmin(arr), np.nanmax(arr)

                mean = np.nanmean(arr)
                std = np.nanstd(arr)
                median = np.nanmedian(arr)
                q75, q25 = np.nanpercentile(arr, [75, 25])
                iqr = q75 - q25

                scale_summary_stats[metric_name] = {
                    'mean': mean,
                    'std': std,
                    'median': median,
                    'iqr': iqr,
                }
            PLOT_DATA[model_size] = scale_summary_stats

        # Order PLOT_DATA by key size (assuming keys are model sizes, e.g., int or float)
        PLOT_DATA = dict(sorted(PLOT_DATA.items(), key=lambda x: x[0]))
        metric_name_mapping = {
            'abs_probs_breakthroughness': 'Breakthroughness (p)',
            'abs_negent_breakthroughness': 'Breakthroughness (-H)',
            'probs_skewness': 'Skewness (p)',
            'negent_skewness': 'Skewness (-H)'
        }

        if SINGLE_PLOT:
            
            plt.rcParams.update(utils.scaling_curves_style)
            fig,ax=plt.subplots()
            # Plot z means with z stds as error bars, all on the same plot, colored with legend
            if len(PLOT_DATA) == 0:
                return  # nothing to plot

            metric_names = list(next(iter(PLOT_DATA.values())).keys())
            model_sizes = sorted(PLOT_DATA.keys())
            colors = plt.cm.tab10.colors if len(metric_names) <= 10 else plt.cm.tab20.colors

            fig, ax = plt.subplots()
            for idx, metric in enumerate(metric_names):
                z_means = []
                z_stds = []
                for size in model_sizes:
                    z_means.append(PLOT_DATA[size][metric]['normed_mean'])
                    z_stds.append(PLOT_DATA[size][metric]['normed_std'])
                z_means = np.array(z_means)
                z_stds = np.array(z_stds)
                ax.errorbar(
                    model_sizes, z_means, yerr=z_stds, fmt='-o',
                    color=colors[idx % len(colors)], capsize=5, label=metric_name_mapping[metric]
                )
            ax.set_xlabel("Model Size (B)")
            ax.set_ylabel("Normalized Mean (mean ± std)")
            ax.set_title("Emergence Metrics (Normalized Mean) vs Model Size - Deepseek-R1-Distill")
            ax.set_xticks(model_sizes)
            ax.grid(True, alpha=0.3)
            ax.legend(title="Metric")
            fig.tight_layout()


        else:
            # Collect all metric names from the first entry
            if len(PLOT_DATA) == 0:
                return  # nothing to plot

            plt.rcParams.update(utils.scaling_curves_2x2_style)
            
            metric_names = list(next(iter(PLOT_DATA.values())).keys())
            model_sizes = sorted(PLOT_DATA.keys())

            n_metrics = len(metric_names)
            
            # Use a color palette for different metrics
            if n_metrics <= 10:
                colors = plt.cm.tab10.colors
            else:
                colors = plt.cm.tab20.colors

            # Create 2x2 grid layout for single column format
            fig, axs = plt.subplots(nrows=2, ncols=2)
            axs = axs.flatten()  # Make it easy to iterate
            
            for idx, metric in enumerate(metric_names):
                centres = []
                spreads = []

                if "breakthroughness" in metric.lower():
                    centre_y_label = "Median"
                    spread_y_label = "IQR"
                    medians = [PLOT_DATA[size][metric]['median'] for size in model_sizes]
                    iqrs = [PLOT_DATA[size][metric]['iqr'] for size in model_sizes]
                    centres = medians
                    spreads = iqrs
                elif "skewness" in metric.lower():
                    centre_y_label = "Mean"
                    spread_y_label = "Std"
                    means = [PLOT_DATA[size][metric]['mean'] for size in model_sizes]
                    stds = [PLOT_DATA[size][metric]['std'] for size in model_sizes]
                    centres = means
                    spreads = stds
                else:
                    centres = []
                    spreads = []

                centres = np.array(centres)
                spreads = np.array(spreads)
                
                axs[idx].errorbar(model_sizes, centres, yerr=spreads, fmt='x-', capsize=3, color=colors[idx % len(colors)])
                axs[idx].set_xlabel("Model Size (B)")
                axs[idx].set_ylabel(f"{centre_y_label} ± {spread_y_label}")
                axs[idx].set_title(metric_name_mapping[metric])
                axs[idx].set_xticks(model_sizes)
                axs[idx].grid(True, alpha=0.3)
            
            fig.suptitle("Emergence Metrics vs Model Size - Deepseek-R1-Distill", y=0.98)

        plt.tight_layout()

        

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "scaling_curves"), exist_ok=True)
            save_name = "scaling_curves.png"
            save_path = os.path.join(self.results_save_dir, "scaling_curves", save_name)
            fig.savefig(save_path,bbox_inches='tight')
        if SHOW:
            plt.show()

        return PLOT_DATA

    def get_emergence_score_rankings(self,dataset_name,model_name,aggregate_with="probability"):

        """
        Returns task idxs ranked by emergence score rankings 

        e.g: [10,3,5,7,8,...] means that task 10 has first ranking, task 3 has second ranking, etc.

        Lower ranking means higher total emergence score
        
        """

        assert aggregate_with in ["probability", "entropy"], "aggregate_with must be either 'probability' or 'entropy'"

        path = self.get_selected_path(model_name=model_name,dataset_name=dataset_name)
        emergence_scores = self.emergence_scores(path=path)

        # Compute rankings for each metric (largest = rank 1, nans get lowest ranking)
        rankings = {}
        for key, arr in emergence_scores.items():
            arr_for_sort = np.copy(arr)
            nan_mask = np.isnan(arr_for_sort)
            # Temporarily set nans to -inf so they sort last
            arr_for_sort[nan_mask] = -np.inf
            order = np.argsort(-arr_for_sort) # order[0] is index with ihighest value, order[i] is the index with ith highest value. Each element is a task index. 
            ranks = np.empty_like(order, dtype=float)
            # Assign ranks: 1 is best, len(arr) is worst
            ranks[order] = np.arange(1, len(arr_for_sort)+1) #each element gets a ranking
            # Set nans to lowest possible rank (i.e., worst)
            ranks[nan_mask] = len(arr_for_sort)
            rankings[key] = ranks

        # Aggregate ranking by summing individual rankings (nans get lowest ranking)

        if aggregate_with == "probability":
            prob_rankings = np.stack([rankings[k] for k in rankings if "probs" in k])
            aggregate_ranking = np.sum(prob_rankings, axis=0)
        elif aggregate_with == "entropy":
            entropy_rankings = np.stack([rankings[k] for k in rankings if "entropy" in k])
            aggregate_ranking = np.sum(entropy_rankings, axis=0)
        else: 
            raise ValueError(f"Invalid aggregate_with: {aggregate_with}")

        return np.argsort(aggregate_ranking) # lower ranking is better (high emergence scores get first rankings)

    def task_comparison(self,dataset_name,SAVE=False):

        import itertools

        plt.rcParams.update(utils.aggregate_plot_config)


        PLOT_DATA = {}
        for model in self.model_names:
            emergence_score_rankings = self.get_emergence_score_rankings(dataset_name=dataset_name, model_name=model,aggregate_with="probability")
            PLOT_DATA[model] = emergence_score_rankings

        model_names = list(PLOT_DATA.keys())
        pairwise = list(itertools.combinations(model_names, 2))
        n_pairs = len(pairwise)

        fig, axs = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 7), squeeze=False)
        axs = axs[0]

        for idx, (model_a, model_b) in enumerate(pairwise):
            ranking_a = PLOT_DATA[model_a]
            ranking_b = PLOT_DATA[model_b]


            # Build task -> rank mapping for each model
            task_to_rank_a = {task: rank for rank, task in enumerate(ranking_a)}
            task_to_rank_b = {task: rank for rank, task in enumerate(ranking_b)}

            tasks = sorted(task_to_rank_a.keys())
            x = [task_to_rank_a[task] for task in tasks] # x[i] is the rank of task i in model a (lower is better)
            y = [task_to_rank_b[task] for task in tasks] # y[i] is the rank of task i in model b (lower is better)
            x_1 = [task_to_rank_a[task] + 1 for task in tasks] # ONE INDEXED
            y_1 = [task_to_rank_b[task] + 1 for task in tasks] # ONE INDEXED



            ax = axs[idx]
            ax.scatter(x_1, y_1, s=250, c='C0', zorder=2)
            # for task, x_val, y_val in zip(tasks, x, y):
            #     ax.text(
            #         x_val, y_val, str(task),
            #         fontsize=18, ha='center', va='center',
            #         color='white',
            #         bbox=dict(facecolor='C0', alpha=0.85, boxstyle='circle,pad=0.25'),
            #         zorder=3
            #     )


            ax.plot([0, len(tasks) - 1], [0, len(tasks) - 1], 'k--', alpha=0.5, zorder=1)
            ax.set_xlabel(f"{model_a} task ranking (lower is better)", fontsize=18, fontweight='bold')
            ax.set_ylabel(f"{model_b} task ranking (lower is better)", fontsize=18, fontweight='bold')
            ax.set_title(f"Task ranking comparison:\n{model_a} vs {model_b}", fontsize=20, fontweight='bold', pad=20)
            ax.set_xlim(-1, len(tasks) + 1)
            ax.set_ylim(-1, len(tasks) + 1)
            #ax.set_xticks(range(len(x)))
            #ax.set_yticks(range(len(y)))
            ax.invert_xaxis()
            ax.invert_yaxis()
            # INSERT_YOUR_CODE
            n_ticks = min(6, len(tasks))
            tick_locs = np.linspace(1, len(tasks), n_ticks, dtype=int)
            ax.set_xticks(tick_locs)
            ax.set_yticks(tick_locs)
            
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=16)

            # Pearson r^2
            r, _ = pearsonr(x_1, y_1)
            r2 = r ** 2
            ax.text(
                0.98, 0.02,
                f"$R^2$ = {r2:.2f}",
                transform=ax.transAxes,
                fontsize=18,
                ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='C0', boxstyle='round,pad=0.3'),
                zorder=10
            )

        plt.tight_layout()
        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "task_comparison"), exist_ok=True)
            save_name = "task_comparison.png"
            save_path = os.path.join(self.results_save_dir, "task_comparison", save_name)
            plt.savefig(save_path,bbox_inches='tight')
        plt.show()

        return PLOT_DATA
        
    def get_metric_tables(self,model_name,dataset_name,mode="full_view",score_average_type="mean"):
        
        selected_paths = self.get_selected_path(model_name=model_name,dataset_name=dataset_name)

        config, SAVE_DATA = utils.read_save_data(selected_paths)


        tables = {}
        metric_types = ["answer_score", "answer_probability", "answer_entropy", "answer_ranking"]
        for metric_type in metric_types:
            tables[metric_type] = utils.get_metric_table(SAVE_DATA, metric_type=metric_type, mode=mode, score_average_type=score_average_type)
        
        return tables

  


def apply_style_dict(style_dict):
    """Apply the LaTeX academic style to matplotlib."""
    plt.rcParams.update(style_dict)
    return style_dict

def force_continue_analysis(texts,config,tokenizer,end_of_input_ids, pad_token="<pad>",):

    force_continue_str = config.force_continue
    end_of_input_str = tokenizer.decode(end_of_input_ids,skip_special_tokens=True)

    texts_without_pad = [text.replace(pad_token,"") for text in texts]
    generated_text_only = [text.split(end_of_input_str)[1] for text in texts_without_pad]

    force_continue_counts = []
    token_counts_after_force_continue = []
    
    for text in generated_text_only:
        # Count total force_continue instances
        text=text[0]
        force_continue_count = text.count(force_continue_str)
        force_continue_counts.append(force_continue_count)
        
        # Split by force_continue to get segments
        segments = text.split(force_continue_str)
        
        # Count tokens in each segment after force_continue (excluding the first segment which is before any force_continue)
        segment_token_counts = []
        for i, segment in enumerate(segments[1:], 1):  # Skip first segment, start from index 1
            # Tokenize the segment and count tokens
            tokens = tokenizer.encode(segment, add_special_tokens=False)
            segment_token_counts.append(len(tokens))
        
        token_counts_after_force_continue.append(segment_token_counts)
    
    return force_continue_counts, token_counts_after_force_continue
    