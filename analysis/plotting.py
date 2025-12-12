import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os 

from .core import read_save_data
from .plot_configs import *

def get_model_plot_name(model_name):
    if "deepseek" in model_name.lower():
        return "DeepSeek-R1"
    elif "qwen2.5" in model_name.lower():
        return "Qwen2.5"
    elif "gemma" in model_name.lower():
        return "Gemma"
    elif "qwq" in model_name.lower():
        return "QwQ"
    elif "phi" in model_name.lower():
        return "Phi-4-Reasoning-Plus"
    else:
        return model_name

def get_dataset_plot_name(dataset_name):
    if "gpqa" in dataset_name.lower():
        return "GPQA"
    elif "aime_2024" in dataset_name.lower():
        return "AIME24"
    elif "aime25" in dataset_name.lower():
        return "AIME25"
    else:
        return dataset_name


def plot_aggregate_data(SAVE_DATA, config, metrics=['answer_score', 'answer_probability', 'answer_entropy', 'answer_ranking'], logy=False, style_dict=None,title=None,save_path=None,negent_lims=None):
    """
    Plots aggregate metrics (averaged across all samples and completions) for each metric type.
    """
    if style_dict is not None:
        old_params = plt.rcParams.copy() 
        plt.rcParams.update(style_dict)
    else:
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(aggregate_plot_style)

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

    recomputed_metrics_exist = 'recomputed_metrics' in SAVE_DATA[token_budgets[0]]
    if recomputed_metrics_exist:
        #print(f"Recomputed metrics found. Using recomputed metrics...")
        metric_key='recomputed_metrics'
    else:
        #print(f"No recomputed metrics found. Using metrics...")
        metric_key='metrics'

    for token_budget in token_budgets:
        metric_data = SAVE_DATA[token_budget][metric_key]
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
        ax.plot(token_budgets, metric_means['answer_score'], label=labels['answer_score'], color=colors['answer_score'],marker='o')
    if "answer_probability" in metric_types:
        if logy:
            logprobs_mean = np.clip(np.log10(metric_means['answer_probability']), -100, 100)
            ax.errorbar(token_budgets, logprobs_mean, yerr=metric_stds['answer_probability'], label='log10(p)', color=colors['answer_probability'],marker='s')
        else:
            ax.errorbar(token_budgets, metric_means['answer_probability'], yerr=metric_stds['answer_probability'], label=labels['answer_probability'], color=colors['answer_probability'],marker='s')
            ax.set_ylim(-0.1, 1.1)

    if "answer_entropy" in metric_types and ax2 is not None:
        ax2.errorbar(token_budgets, -np.array(metric_means['answer_entropy']), yerr=metric_stds['answer_entropy'], label=labels['answer_entropy'], color=colors['answer_entropy'],marker='^')
    
    if ax2 is not None and "answer_entropy" in metric_types:
        entropy_vals = -np.array(metric_means['answer_entropy'])
        entropy_stds = np.array(metric_stds['answer_entropy'])
        if negent_lims is None:
            lower = np.nanmin(entropy_vals - entropy_stds)
            upper = np.nanmax(entropy_vals + entropy_stds)
        else:
            lower = negent_lims[0]
            upper = negent_lims[1]
        ax2.set_ylim(lower, upper)
    if negent_lims is not None:
        ax2.set_ylim(negent_lims[0], negent_lims[1])

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Token Budget')
    ax.set_ylabel('Score/Probability',color='blue')
    if ax2:
        ax2.set_ylabel('Negentropy',color='green')


    ax.tick_params(axis='both', which='major')
    if ax2:
        ax2.tick_params(axis='y', which='major')
    ax.grid(True)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
    if ax2:
        ax2.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left',  # CHANGED - placed in whitespace area
              frameon=True,
              framealpha=0.95,  # CHANGED - slightly more opaque
              edgecolor='black',
              fancybox=False)
    else:
        ax.legend(lines1, labels1, 
                loc='upper left',  # CHANGED
                frameon=True,
                framealpha=0.95,  # CHANGED
                edgecolor='black',
                fancybox=False)

    
    if title is None:
        fig.suptitle(f'{config.dataset_name} - {config.model_name} - {config.num_samples} samples - {config.num_completions} completions (Aggregate)')
    else:
        fig.suptitle(title)
    plt.tight_layout()


    if save_path is not None:
        plt.savefig(save_path, dpi=300,bbox_inches='tight')
        plt.close()
    else: 
        plt.show()

def plot_data(data_dir=None, config=None, SAVE_DATA=None, metrics=["answer_score", "answer_probability", "answer_entropy"], sample_idxs=None, n_samples=None, logy=False, titles=None,fig_title=None,localisation_run=False,score_average_type="mode",style_dict=None,single_legend=True,SHOW=True,SAVE=False,save_dir=None,save_name=None,negent_lims=None):

    if style_dict is not None:
        plt.rcParams.update(plt.rcParamsDefault)
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
        if sample_idxs.ndim > 1:
            selected_samples = np.ravel(selected_samples)
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

        recomputed_metrics_exist = 'recomputed_metrics' in SAVE_DATA[all_token_budgets[0]]
        if recomputed_metrics_exist:
            #print(f"Recomputed metrics found. Using recomputed metrics...")
            metric_key='recomputed_metrics'
        else:
            #print(f"No recomputed metrics found. Using metrics...")
            metric_key='metrics'

        for token_budget in all_token_budgets:
            score_data_all = SAVE_DATA[token_budget][metric_key].get('answer_score', None)
            probs_data_all = SAVE_DATA[token_budget][metric_key].get('answer_probability', None)
            entropy_data_all = SAVE_DATA[token_budget][metric_key].get('answer_entropy', None)

            token_budgets.append(token_budget)
            if metric_key in SAVE_DATA[token_budget] and score_data_all is not None and probs_data_all is not None and entropy_data_all is not None:
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
                    raise ValueError(f"Invalid score_average: {score_average_type}")

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
            axes[plot_idx].plot(token_budgets, scores_average, label='Score', color='blue',marker='o')

        if "answer_probability" in metrics:
            if logy:
                logprobs_mean = np.clip(np.log10(probs_mean), -100, 100)
                axes[plot_idx].errorbar(token_budgets, logprobs_mean, yerr=probs_std, label='log10(p)', color='red',marker='s')
            else:
                axes[plot_idx].errorbar(token_budgets, probs_mean, yerr=probs_std, label='Probability', color='red',marker='s')
                axes[plot_idx].set_ylim(-0.1, 1.1)

        if "answer_entropy" in metrics and ax2 is not None:
            ax2.errorbar(token_budgets, -np.array(entropies_mean), yerr=entropies_std, label='Negentropy', color='green',marker='^')
        if negent_lims is not None:
            ax2.set_ylim(negent_lims[0], negent_lims[1])

        axes[plot_idx].set_xscale('log', base=2)
        axes[plot_idx].set_xticks(token_budgets)
        axes[plot_idx].set_xticklabels([str(tb) for tb in token_budgets])
        
        # Edge tick label coloring (keep, but axes ylabels are global now)
        axes[plot_idx].tick_params(axis='y', labelcolor='blue')
        if ax2:
            ax2.tick_params(axis='y', labelcolor='green')
            
        axes[plot_idx].set_title(
            titles[plot_idx] if titles is not None else f'Sample {sample_idx}'
        )

        # Remove labels and ticks for non-edge subplots
        row = plot_idx // 2
        col = plot_idx % 2
        if col > 0:  # Not leftmost column
            axes[plot_idx].set_ylabel('')
            axes[plot_idx].tick_params(left=False, labelleft=False)
        if row < n_rows - 1:  # Not bottom row
            axes[plot_idx].set_xlabel('')
            axes[plot_idx].tick_params(bottom=False, labelbottom=False)
        else:
            axes[plot_idx].set_xlabel('Token Budget')


        # Don't set axis-level left/ylabels anywhere

        # Collect legend handles and labels from first subplot only
        if not legend_collected:
            lines1, labels1 = axes[plot_idx].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
            legend_handles = lines1 + lines2
            legend_labels = labels1 + labels2
            legend_collected = True

    # Add big axis labels for y and y2 (global side labels, not per subplot)
    # For left (Score/Probability)
    fig.text(0.04, 0.5, 'Score/Probability', va='center', rotation='vertical', fontsize=plt.rcParams.get('axes.labelsize', 18), color='blue', fontweight=plt.rcParams.get('axes.labelweight','bold'))
    # For right (Negentropy), if present in metrics
    if "answer_entropy" in metrics:
        fig.text(0.965, 0.5, 'Negentropy', va='center', rotation='vertical', fontsize=plt.rcParams.get('axes.labelsize', 18), color='green', fontweight=plt.rcParams.get('axes.labelweight','bold'))

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

    if SAVE:
        save_dir = save_dir if save_dir is not None else os.path.join("tmp_dir")
        save_path = os.path.join(save_dir, "plot_data.png")
        fig.savefig(save_path, bbox_inches='tight',dpi=300)
        plt.close()
