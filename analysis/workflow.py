import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from scipy.stats import pearsonr, spearmanr

from .core import *
from .plotting import *
from .plot_configs import *
from .text_processing import *
from .misc import *
from itertools import combinations
from scipy.stats import mannwhitneyu

probability_breakthroughness_lims = [0,2]
negative_entropy_breakthroughness_lims = [0,3]
probability_skewness_lims = [-2,4]
negative_entropy_skewness_lims = [-6,12]



probability_breakthroughness_lims = [-5,50]
negative_entropy_breakthroughness_lims = [-5,100]
probability_skewness_lims = [-3,3]
negative_entropy_skewness_lims = [-10,10]

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

        self.void_paths = [] 
        for path in self.paths:
            if ("1.5B" in path or "7B" in path or "14B" in path) and self.scaling_config[1].lower() not in path.lower():
                self.void_paths.append(path)

        self.scaling_paths = self._find_scaling_paths()
        self.scaling_paths = sorted(self.scaling_paths, key=lambda path: get_model_size(path))

        self.non_scaling_paths = []
        for path in self.paths:
            if path not in self.scaling_paths:
                if path not in self.void_paths:
                    self.non_scaling_paths.append(path)
        for path in self.paths:
            if all ([part.lower() in path.lower() for part in self.scaling_config]):
                if path not in self.void_paths:
                    self.non_scaling_paths.append(path)


        self.results_save_dir = results_save_dir
        os.makedirs(self.results_save_dir, exist_ok=True)
        
        self.STYLE_DICT_MAPPINGS = {} #can figure this out later. should be quite simple (?)

        self.kde_bw_adjust = 1.0

    def _find_scaling_paths(self):
        self.scaling_paths = []
        for path in self.paths:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            if self.scaling_config[0].lower() in model_name.lower() and self.scaling_config[1].lower() in dataset_name.lower():
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
        if self.scaling_config[0].lower() in model_name.lower() and self.scaling_config[1].lower() in dataset_name.lower():
            main_scale = self.scaling_config[2]
            selected_paths = [path for path in selected_paths if main_scale in path]
            selected_path = selected_paths[0]
        else:
            selected_path = selected_paths[0]


        return selected_path

    def get_selected_path_from_non_scaling_paths(self,model_name,dataset_name):
        for path in self.non_scaling_paths:
            if model_name.lower() in path.lower() and dataset_name.lower() in path.lower():
                return path
        raise ValueError(f"No path found for model {model_name} and dataset {dataset_name} in non-scaling paths")

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

    def token_budget_plot(self,model_name=None,dataset_name=None,SHOW=True,SAVE=False):

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(boxplot_style)

        path = self.get_selected_path(model_name=model_name,dataset_name=dataset_name)
        config, SAVE_DATA = read_save_data(path)
        counts = get_token_count_table(config,SAVE_DATA)


        boxplot_data = []
        token_budgets = sorted(counts.columns)


        for tb in token_budgets:
            # For each cell in this column, which is a tuple of ints
            vals = []
            for cell in counts[tb]:
                if isinstance(cell, (tuple, list, np.ndarray)):
                        vals.extend(cell)
            boxplot_data.append(vals)

                        
        color_palette = matplotlib.colormaps["tab10"].colors if hasattr(matplotlib.colormaps["tab10"], "colors") \
            else plt.cm.tab10.colors
        colors = [color_palette[i % len(color_palette)] for i in range(len(token_budgets))]

        fig, ax = plt.subplots()


        bp = ax.boxplot(
            boxplot_data,
            positions=range(len(token_budgets)),
            patch_artist=True,  # enables facecolor by patches
            showfliers=False,
        )

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            # Edge/colors/linewidth/etc otherwise governed by rcParams
        title = f"Token counts for {model_name.capitalize()} on {dataset_name.capitalize()} "

        ax.set_xticks(range(len(token_budgets)))
        ax.set_xticklabels(token_budgets, rotation=45)
        ax.set_xlabel("Token Budgets")
        ax.set_ylabel("Token Counts per Completion")
        ax.set_title("Distribution of Token Counts per Completion across Token Budgets")
        ax.grid(True, axis='y')  # style handled by rcParams
        fig.tight_layout()

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "token_budget_plots"), exist_ok=True)
            save_name = f"{model_name.capitalize()}_{dataset_name.capitalize()}_token_budget_plot.png"
            save_path = os.path.join(self.results_save_dir, "token_budget_plots", save_name)
            plt.savefig(save_path,bbox_inches='tight')
            plt.close()
        if SHOW:
            plt.show()


        return counts  

    def aggregate_plots(self, SHOW=True,SAVE=False,log_probs_aime=False,model_name=None,dataset_name=None):

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(aggregate_plot_style)


        for path in self.non_scaling_paths:

            if model_name is not None:
                if model_name.lower() not in path.lower():
                    continue
            if dataset_name is not None:
                if dataset_name.lower() not in path.lower():
                    continue


            config, SAVE_DATA = read_save_data(path)
            plot_model_name = get_model_plot_name(config.model_name)
            plot_dataset_name = get_dataset_plot_name(config.dataset_name)
            fig_title = f"{plot_dataset_name} - {plot_model_name}"

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
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(heatmap_config)

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
            "deepseek": "Deepseek",
            "qwen2.5": "Qwen-2.5",
            "gemma": "Gemma-2",
            "qwq": "QwQ",
            "phi": "Phi",
        }

        # Build heatmap matrix
        heatmap_matrix = np.full((len(dataset_names), len(model_names)), np.nan)
        for i, dataset in enumerate(dataset_names):
            for j, model in enumerate(model_names):
                if (dataset, model) in heatmap_data:
                    heatmap_matrix[i, j] = heatmap_data[(dataset, model)]

        fig, ax = plt.subplots()
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
            #title = f"{metric_pair.replace('_', ' vs. ').title()} ({correlation_type.capitalize()})"
            title = f"{metric_pair.replace('_', ' vs. ').title()}"

        ax.set_title(title)
        plt.tight_layout()

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "correlation_heatmap"), exist_ok=True)
            save_name = f"correlation_heatmap_{metric_pair}_{correlation_type}.png"
            save_path = os.path.join(self.results_save_dir, "correlation_heatmap", save_name)
            plt.savefig(save_path,bbox_inches='tight')
        if SHOW:
            plt.show()

        return heatmap_matrix

    def emergence_scores(self,path):


        config, SAVE_DATA = read_save_data(path)

        emergence_metric_keys = ["abs_probs_breakthroughness","abs_negent_breakthroughness","probs_skewness","negent_skewness"]

        emergence_metric_data  = get_samplewise_breakthrough_and_skew(SAVE_DATA,config)

        emergence_metric_data = {key: emergence_metric_data[key] for key in emergence_metric_keys}

        return emergence_metric_data

    def get_all_non_scaling_emergence_scores(self):

        emergence_scores = {model_name: {dataset_name: None for dataset_name in self.dataset_names} for model_name in self.model_names}
        for path in self.non_scaling_paths:
            config, SAVE_DATA = read_save_data(path)
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            emergence_scores[model_name][dataset_name] = self.emergence_scores(path=path)
        return emergence_scores

    def correlation_scores(self, model_name, dataset_name,method="1D"):

        if method == "1D":

            tables = self.get_metric_tables(model_name=model_name,dataset_name=dataset_name,mode="token_budget_view")
            score_table, prob_table, entropy_table, ranking_table = tables["answer_score"], tables["answer_probability"], tables["answer_entropy"], tables["answer_ranking"]
            score_matrix, prob_matrix, entropy_matrix, ranking_matrix =  table_to_2d_matrix(score_table),  table_to_2d_matrix(prob_table),  table_to_2d_matrix(entropy_table),  table_to_2d_matrix(ranking_table)
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
            score_matrix, prob_matrix, entropy_matrix, ranking_matrix =  table_to_2d_matrix(score_table),  table_to_2d_matrix(prob_table),  table_to_2d_matrix(entropy_table),  table_to_2d_matrix(ranking_table)
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
            score_matrix, prob_matrix, entropy_matrix, ranking_matrix =  table_to_3d_matrix(score_table),  table_to_3d_matrix(prob_table),  table_to_3d_matrix(entropy_table),  table_to_3d_matrix(ranking_table)
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

    def emergence_score_summary_stats_across_datasets(self,model_name,SAVE=False,save_dir=None):

        selected_paths = []
        for dataset in self.dataset_names:
            selected_paths.append(self.get_selected_path(model_name=model_name, dataset_name=dataset))


        EMERGENCE_METRIC_DATA = {}
        emergence_score_types = ["abs_probs_breakthroughness","abs_negent_breakthroughness","probs_skewness","negent_skewness"]
        metric_names = ["Breakthroughness (p)","Breakthroughness (-H)","Skewness (p)","Skewness (-H)"]
        stats_dict = {}
        for path in selected_paths:
            config, SAVE_DATA =  read_save_data(path)
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            emergence_metric_data = get_samplewise_breakthrough_and_skew(SAVE_DATA, config)
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

        if SAVE:
            os.makedirs(os.path.join(save_dir, "emergence_score_summary_stats_across_datasets"), exist_ok=True)
            save_name = f"{model_name}_emergence_score_summary_stats_across_datasets.csv"
            save_path = os.path.join(save_dir, "emergence_score_summary_stats_across_datasets", save_name)
            summary_stats_df.to_csv(save_path)

        return summary_stats_df

    def emergence_score_summary_stats_across_models(self,dataset_name,SAVE=False,save_dir=None):

        selected_paths = []
        for model in self.model_names:
            selected_paths.append(self.get_selected_path(model_name=model, dataset_name=dataset_name))

        EMERGENCE_METRIC_DATA = {}
        emergence_score_types = ["abs_probs_breakthroughness","abs_negent_breakthroughness","probs_skewness","negent_skewness"]
        metric_names = ["Breakthroughness (p)","Breakthroughness (-H)","Skewness (p)","Skewness (-H)"]
        stats_dict = {}
        for path in selected_paths:
            config, SAVE_DATA =  read_save_data(path)
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            emergence_metric_data = get_samplewise_breakthrough_and_skew(SAVE_DATA, config)
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

        if SAVE:
            os.makedirs(os.path.join(save_dir, "emergence_score_summary_stats_across_models"), exist_ok=True)
            save_name = f"{dataset_name}_emergence_score_summary_stats_across_models.csv"
            save_path = os.path.join(save_dir, "emergence_score_summary_stats_across_models", save_name)
            summary_stats_df.to_csv(save_path)

        return summary_stats_df

    def emergence_score_dist_across_datasets(self,model_name,SHOW=True,SAVE=False,breakthroughness_lims=probability_breakthroughness_lims,skewness_lims=probability_skewness_lims):
        
        plt.rcParams.update(plt.rcParamsDefault)

        if LEGACY:
            plt.rcParams.update(emergence_score_dist_style_legacy)
        else:
            plt.rcParams.update(emergence_score_dist_style)

        PLOT_DATA = {}
        for dataset in self.dataset_names:
            selected_path = self.get_selected_path(model_name=model_name, dataset_name=dataset)
            emergence_scores = self.emergence_scores(path=selected_path)
            PLOT_DATA[dataset] = emergence_scores

        sns.set_palette("husl")


        metric_types = ['abs_probs_breakthroughness', 'probs_skewness', 'abs_negent_breakthroughness', 'negent_skewness']
        datasets = ['aime25', 'gpqa', 'aime_2024']
        metric_name_map = {
            'abs_probs_breakthroughness': 'Breakthroughness (p)',
            'abs_negent_breakthroughness': 'Breakthroughness (-H)',
            'probs_skewness': 'Skewness (p)',
            'negent_skewness': 'Skewness (-H)'
        }
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        line_styles = ['-', '--', '-.']

        if LEGACY:

            fig, axs = plt.subplots(1, 4)
            axs = axs.flatten()
            fig.suptitle(f'Emergence score distributions - {get_model_plot_name(model_name)}', y=0.98)

            for i, metric in enumerate(metric_types):
                ax = axs[i]
                if "probs" in metric:
                    ax.set_facecolor(PROB_PLOTS_BACKGROUND_COLOR)
                elif "negent" in metric:
                    ax.set_facecolor(NEGENT_PLOTS_BACKGROUND_COLOR)

                if "breakthroughness" in metric: 
                    clip = (0,None)
                elif "skewness" in metric:
                    clip = (None,None)
                else:
                    clip = (None,None)

                for j, dataset in enumerate(datasets):

                    values = np.array(PLOT_DATA[dataset][metric])
                    values = values[np.isfinite(values)]
                    if len(values) > 1:
                        if "breakthroughness" in metric: #we renormalize over positive values
                            from scipy.stats import gaussian_kde
                            kde = gaussian_kde(values,bw_method="scott")
                            kde.set_bandwidth(kde.factor * self.kde_bw_adjust)

                            # Evaluate on positive domain
                            x_max = values.max() + 3 * kde.factor * values.std()
                            x_eval = np.linspace(0, x_max, 500)
                            density = kde(x_eval)

                            # Renormalize
                            integral = np.trapezoid(density, x_eval) # this is < 1 (some probability mass falls into negative values)
                            density_normalized = density / integral #renormalize positive density to 1
                            
                            # Plot manually
                            ax.plot(x_eval, density_normalized, color=colors[j], 
                                linewidth=3, linestyle=line_styles[j], label=dataset, alpha=0.8)
                            ax.fill_between(x_eval, density_normalized, alpha=0.15, color=colors[j])

                        else: 
                            sns.kdeplot(
                                values, ax=ax, label=dataset,
                                color=colors[j], alpha=0.15, linewidth=3,
                                linestyle=line_styles[j], fill=True, bw_adjust=self.kde_bw_adjust,clip=clip)

                except Exception as e:
                    print(f"Error plotting {dataset}, {metric}: {e}")

                            # Evaluate including some negative domain
                            x_max = values.max() + 3 * kde.factor * values.std()
                            x_min = -1  # Start from negative to show it touching 0
                            x_eval = np.linspace(x_min, x_max, 500)
                            density = kde(x_eval)
                            
                            # Set density to 0 for negative values (since breakthroughness is non-negative)
                            density[x_eval < 0] = 0  # FORCE negative domain to 0

                            # Renormalize over full domain
                            integral = np.trapezoid(density, x_eval)
                            density_normalized = density / integral
                            
                            # Plot manually
                            ax.plot(x_eval, density_normalized, color=colors[j], 
                                linewidth=3, linestyle=line_styles[j], label=get_dataset_plot_name(dataset), alpha=0.8)
                            ax.fill_between(x_eval, density_normalized, alpha=0.15, color=colors[j])

                        else: 
                            sns.kdeplot(
                                values, ax=ax, label=dataset,
                                color=colors[j], alpha=0.15, linewidth=3,
                                linestyle=line_styles[j], fill=True, bw_adjust=self.kde_bw_adjust,clip=clip)


                if "breakthroughness" in metric:
                    ax.set_ylim(0)
                #title = metric_name_map[metric]
                #ax.set_title(title, pad=15)
                ax.set_xlabel(metric_name_map[metric])
                if i == 0:
                    ax.set_ylabel('Density')
                else:
                    ax.set_ylabel('')
                
                ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                ax.set_axisbelow(True)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)
                    spine.set_color('#cccccc')
                if "breakthroughness" in metric:
                    if "prob" in metric:
                        probability_breakthroughness_lims = [0,1.5] #custom set for this one
                        ax.set_xlim(probability_breakthroughness_lims)
                    elif "negent" in metric:
                        ax.set_xlim(negative_entropy_breakthroughness_lims)
                elif "skewness" in metric:
                    if "probs" in metric:
                        ax.set_xlim(probability_skewness_lims)
                    elif "negent" in metric:
                        ax.set_xlim(negative_entropy_skewness_lims)
                ax.autoscale(False) 

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

        
        if not LEGACY:

            fig, axs = plt.subplots(2, 2)
            fig.suptitle(f'Emergence score distributions - {get_model_plot_name(model_name)}', y=0.98)

            # Define layout: probability metrics on left, negentropy on right
            metric_layout = [
                ['abs_probs_breakthroughness', 'abs_negent_breakthroughness'],  # Row 0
                ['probs_skewness', 'negent_skewness']  # Row 1
            ]
            
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#cccccc')
            if "breakthroughness" in metric:
                if "probability" in metric:
                    ax.set_xlim(probability_breakthroughness_lims)
                elif "negative_entropy" in metric:
                    ax.set_xlim(negative_entropy_breakthroughness_lims)
            elif "skewness" in metric:
                if "probability" in metric:
                    ax.set_xlim(probability_skewness_lims)
                elif "negative_entropy" in metric:
                    ax.set_xlim(negative_entropy_skewness_lims)

            for row in range(2):
                for col in range(2):
                    ax = axs[row, col]
                    metric = metric_layout[row][col]
                    
                    # Set background color
                    if "probs" in metric:
                        ax.set_facecolor(PROB_PLOTS_BACKGROUND_COLOR)
                    elif "negent" in metric:
                        ax.set_facecolor(NEGENT_PLOTS_BACKGROUND_COLOR)

                    # Determine clipping
                    if "breakthroughness" in metric:
                        clip = (0, None)
                    else:
                        clip = (None, None)

                    legend_labels = []
                    
                    for j, dataset in enumerate(datasets):

                        bootstrapped_emergence_samples = self.bootstrap_emergence_score_mean_ci(model_name=model_name,dataset_name=dataset,n_bootstrap=n_bootstrap,ci_level=0.95)

                        values = np.array(PLOT_DATA[dataset][metric])
                        values = values[np.isfinite(values)]
                        
                        if len(values) > 1:
                            # Plot KDE
                            if "breakthroughness" in metric:
                                from scipy.stats import gaussian_kde
                                kde = gaussian_kde(values, bw_method="scott")
                                kde.set_bandwidth(kde.factor * self.kde_bw_adjust)

                                x_max = values.max() + 3 * kde.factor * values.std()
                                x_min = -1
                                x_eval = np.linspace(x_min, x_max, 500)
                                density = kde(x_eval)
                                
                                density[x_eval < 0] = 0
                                integral = np.trapezoid(density, x_eval)
                                density_normalized = density / integral
                                
                                ax.plot(x_eval, density_normalized, color=colors[j], 
                                        linewidth=3, linestyle=line_styles[j], alpha=0.8)
                                ax.fill_between(x_eval, density_normalized, alpha=0.15, color=colors[j])
                            else:
                                sns.kdeplot(values, ax=ax, color=colors[j], alpha=0.15, 
                                            linewidth=3, linestyle=line_styles[j], fill=True, 
                                            bw_adjust=self.kde_bw_adjust, clip=clip)
                            
                            # Get bootstrap statistics
                            bootstrap_stats = bootstrapped_emergence_samples[metric]
                            mean = bootstrap_stats['mean']
                            ci_lower = bootstrap_stats['ci_low']
                            ci_upper = bootstrap_stats['ci_high']
                            
                            # Create legend label with bootstrap statistics
                            legend_label = f"{get_dataset_plot_name(dataset)}: μ={mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
                            legend_labels.append((colors[j], line_styles[j], legend_label))

                    
                    # Set axis properties
                    if "breakthroughness" in metric:
                        ax.set_ylim(0)
                    
                    ax.set_xlabel(metric_name_map[metric])
                    if col == 0:
                        ax.set_ylabel('Density')
                    else:
                        ax.set_ylabel('')
                    
                    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                    ax.set_axisbelow(True)
                    
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.2)
                        spine.set_color('#cccccc')
                    
                    # Set x-limits
                    if "breakthroughness" in metric:
                        if "prob" in metric:
                            ax.set_xlim([0, 1.5])
                        elif "negent" in metric:
                            ax.set_xlim(negative_entropy_breakthroughness_lims)
                    elif "skewness" in metric:
                        if "probs" in metric:
                            ax.set_xlim(probability_skewness_lims)
                        elif "negent" in metric:
                            ax.set_xlim(negative_entropy_skewness_lims)
                    
                    ax.autoscale(False)
                    
                    # Add legend to each panel (top right)
                    if legend_labels:
                        from matplotlib.lines import Line2D
                        legend_handles = [Line2D([0], [0], color=c, linestyle=ls, linewidth=2) 
                                        for c, ls, _ in legend_labels]
                        legend_text = [label for _, _, label in legend_labels]
                        ax.legend(legend_handles, legend_text, loc='upper right')

            plt.tight_layout()



        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "emergence_score_dist_across_datasets"), exist_ok=True)
            save_name = f"{model_name}_emergence_score_dist_across_datasets.png"
            save_path = os.path.join(self.results_save_dir, "emergence_score_dist_across_datasets", save_name)
            plt.draw()
            plt.savefig(save_path,bbox_inches='tight')
            plt.close()

        if SHOW:
            plt.show()
        else:
            plt.close()

        return PLOT_DATA

    def emergence_score_dist_across_models(self, dataset_name, LEGACY=True, n_bootstrap=1000, SHOW=True, SAVE=True, breakthroughness_lims=None, skewness_lims=None):
        """
        Plot KDE distributions of emergence scores across different models for a given dataset.
        """

        plt.rcParams.update(plt.rcParamsDefault)

        if LEGACY:
            plt.rcParams.update(emergence_score_dist_style_legacy)
        else:
            plt.rcParams.update(emergence_score_dist_style)

        PLOT_DATA = {}
        for model_name in self.model_names:
            path = self.get_selected_path(model_name=model_name, dataset_name=dataset_name)
            emergence_scores = self.emergence_scores(path=path)
            PLOT_DATA[model_name] = emergence_scores

        sns.set_palette("husl")

        metric_types = ['abs_probs_breakthroughness', 'probs_skewness', 'abs_negent_breakthroughness', 'negent_skewness']
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

        if LEGACY:
            
            fig, axs = plt.subplots(1, 4)
            axs = axs.flatten()
            fig.suptitle(f'Emergence score distributions - {get_dataset_plot_name(dataset_name)}', y=0.98)

            #enforce limits 
            if "breakthroughness" in metric:
                if "probability" in metric:
                    ax.set_xlim(probability_breakthroughness_lims)
                elif "negative_entropy" in metric:
                    ax.set_xlim(negative_entropy_breakthroughness_lims)
            elif "skewness" in metric:
                if "probability" in metric:
                    ax.set_xlim(probability_skewness_lims)
                elif "negative_entropy" in metric:
                    ax.set_xlim(negative_entropy_skewness_lims)

                if "breakthroughness" in metric: 
                    clip = (0, None)
                elif "skewness" in metric:
                    clip = (None, None)
                else:
                    clip = (None, None)

                for j, model in enumerate(models):
                    try:
                        values = np.array(PLOT_DATA[model][metric])
                        values = values[np.isfinite(values)]
                        if len(values) > 1:
                            if "breakthroughness" in metric:  # we renormalize over positive values
                                from scipy.stats import gaussian_kde
                                kde = gaussian_kde(values, bw_method="scott")
                                kde.set_bandwidth(kde.factor * self.kde_bw_adjust)

                                # Evaluate including some negative domain
                                x_max = values.max() + 3 * kde.factor * values.std()
                                x_min = -1  # Start from negative to show it touching 0
                                x_eval = np.linspace(x_min, x_max, 500)
                                density = kde(x_eval)
                                
                                # Set density to 0 for negative values (since breakthroughness is non-negative)
                                density[x_eval < 0] = 0  # FORCE negative domain to 0

                                # Renormalize over full domain
                                integral = np.trapezoid(density, x_eval)
                                density_normalized = density / integral
                                
                                # Plot manually
                                ax.plot(x_eval, density_normalized, color=colors[j % len(colors)], 
                                    linewidth=3, linestyle=line_styles[j % len(line_styles)], 
                                    label=model.capitalize(), alpha=0.8)
                                ax.fill_between(x_eval, density_normalized, alpha=0.15, 
                                            color=colors[j % len(colors)])

                            else: 
                                sns.kdeplot(
                                    values, ax=ax, label=model.capitalize(),
                                    color=colors[j % len(colors)], alpha=0.15, linewidth=3,
                                    linestyle=line_styles[j % len(line_styles)], fill=True, 
                                    bw_adjust=self.kde_bw_adjust, clip=clip)

                    except Exception as e:
                        print(f"Error plotting {model}, {metric}: {e}")

                if "breakthroughness" in metric:
                    ax.set_ylim(0)

                ax.set_xlabel(metric_name_map[metric])
                if i == 0:
                    ax.set_ylabel('Density')
                else:
                    ax.set_ylabel('')
                
                ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                ax.set_axisbelow(True)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)
                    spine.set_color('#cccccc')

                # Enforce limits 
                if "breakthroughness" in metric:
                    if "probs" in metric:
                        ax.set_xlim(probability_breakthroughness_lims)
                    elif "negent" in metric:
                        ax.set_xlim(negative_entropy_breakthroughness_lims)
                elif "skewness" in metric:
                    if "probs" in metric:
                        ax.set_xlim(probability_skewness_lims)
                    elif "negent" in metric:
                        ax.set_xlim(negative_entropy_skewness_lims)
                
                ax.autoscale(False)

            # Get legend handles and labels from the first subplot after all plotting is done
            handles, labels = axs[0].get_legend_handles_labels()

            # Remove legends from all subplots
            for ax in axs:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

            # Add single legend at the bottom
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.10), 
                    ncol=len(models), frameon=True, fancybox=True, shadow=True)

            plt.tight_layout()

        if not LEGACY:

            fig, axs = plt.subplots(2, 2)
            fig.suptitle(f'Emergence score distributions - {get_dataset_plot_name(dataset_name)}', y=0.98)

            # Define layout: probability metrics on left, negentropy on right
            metric_layout = [
                ['abs_probs_breakthroughness', 'abs_negent_breakthroughness'],  # Row 0
                ['probs_skewness', 'negent_skewness']  # Row 1
            ]
            
            for row in range(2):
                for col in range(2):
                    ax = axs[row, col]
                    metric = metric_layout[row][col]
                    
                    # Set background color
                    if "probs" in metric:
                        ax.set_facecolor(PROB_PLOTS_BACKGROUND_COLOR)
                    elif "negent" in metric:
                        ax.set_facecolor(NEGENT_PLOTS_BACKGROUND_COLOR)

                    # Determine clipping
                    if "breakthroughness" in metric:
                        clip = (0, None)
                    else:
                        clip = (None, None)

                    legend_labels = []
                    
                    for j, model in enumerate(models):
                        try:
                            bootstrapped_emergence_samples = self.bootstrap_emergence_score_mean_ci(
                                model_name=model, dataset_name=dataset_name, 
                                n_bootstrap=n_bootstrap, ci_level=0.95
                            )

                            values = np.array(PLOT_DATA[model][metric])
                            values = values[np.isfinite(values)]
                            
                            if len(values) > 1:
                                # Plot KDE
                                if "breakthroughness" in metric:
                                    from scipy.stats import gaussian_kde
                                    kde = gaussian_kde(values, bw_method="scott")
                                    kde.set_bandwidth(kde.factor * self.kde_bw_adjust)

                                    x_max = values.max() + 3 * kde.factor * values.std()
                                    x_min = -1
                                    x_eval = np.linspace(x_min, x_max, 500)
                                    density = kde(x_eval)
                                    
                                    density[x_eval < 0] = 0
                                    integral = np.trapezoid(density, x_eval)
                                    density_normalized = density / integral
                                    
                                    ax.plot(x_eval, density_normalized, 
                                        color=colors[j % len(colors)], 
                                        linewidth=3, linestyle=line_styles[j % len(line_styles)], 
                                        alpha=0.8)
                                    ax.fill_between(x_eval, density_normalized, alpha=0.15, 
                                                color=colors[j % len(colors)])
                                else:
                                    sns.kdeplot(values, ax=ax, 
                                            color=colors[j % len(colors)], alpha=0.15, 
                                            linewidth=3, linestyle=line_styles[j % len(line_styles)], 
                                            fill=True, bw_adjust=self.kde_bw_adjust, clip=clip)
                                
                                # Get bootstrap statistics
                                bootstrap_stats = bootstrapped_emergence_samples[metric]
                                mean = bootstrap_stats['mean']
                                ci_lower = bootstrap_stats['ci_low']
                                ci_upper = bootstrap_stats['ci_high']
                                
                                # Create legend label with bootstrap statistics
                                legend_label = f"{model.capitalize()}: μ={mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
                                legend_labels.append((colors[j % len(colors)], 
                                                    line_styles[j % len(line_styles)], 
                                                    legend_label))
                        
                        except Exception as e:
                            print(f"Error plotting {model}, {metric}: {e}")
                    
                    # Set axis properties
                    if "breakthroughness" in metric:
                        ax.set_ylim(0)
                    
                    ax.set_xlabel(metric_name_map[metric])
                    if col == 0:
                        ax.set_ylabel('Density')
                    else:
                        ax.set_ylabel('')
                    
                    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                    ax.set_axisbelow(True)
                    
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.2)
                        spine.set_color('#cccccc')
                    
                    # Set x-limits
                    if "breakthroughness" in metric:
                        if "probs" in metric:
                            ax.set_xlim(probability_breakthroughness_lims)
                        elif "negent" in metric:
                            ax.set_xlim(negative_entropy_breakthroughness_lims)
                    elif "skewness" in metric:
                        if "probs" in metric:
                            ax.set_xlim(probability_skewness_lims)
                        elif "negent" in metric:
                            ax.set_xlim(negative_entropy_skewness_lims)
                    
                    ax.autoscale(False)
                    
                    # Add legend to each panel (top right)
                    if legend_labels:
                        from matplotlib.lines import Line2D
                        legend_handles = [Line2D([0], [0], color=c, linestyle=ls, linewidth=2) 
                                        for c, ls, _ in legend_labels]
                        legend_text = [label for _, _, label in legend_labels]
                        ax.legend(legend_handles, legend_text, loc='upper right')

            plt.tight_layout()

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "emergence_score_dist_across_models"), exist_ok=True)
            save_name = f"{dataset_name}_emergence_score_dist_across_models.png"
            save_path = os.path.join(self.results_save_dir, "emergence_score_dist_across_models", save_name)
            plt.draw()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        if SHOW:
            plt.show()
        else:
            plt.close()

        return PLOT_DATA
        
    def plot_top_k_max_emergence_samples(self,model_name,dataset_name,k=4,SHOW=True,SAVE=False,save_dir=None,rank_by="probability_skewness"):

        """
        Get the top k samples with the highest emergence scores for a given model and dataset.


        """

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(top_k_style)

        path = self.get_selected_path(model_name=model_name,dataset_name=dataset_name)
        config,SAVE_DATA =  read_save_data(path)




        emergence_metric_data = self.emergence_scores(path=path)
        metric_names = ['abs_probs_breakthroughness', 'abs_negent_breakthroughness', 'probs_skewness', 'negent_skewness']
        metric_matrix = np.vstack([emergence_metric_data[name] for name in metric_names])

        rankings = self.get_emergence_score_rankings(model_name=model_name,dataset_name=dataset_name,rank_by=rank_by)

        # Find the sample(s) with the lowest  rank (highest emergence)
        top_k_sample_indices = rankings[:k]

        plot_data(SAVE_DATA=SAVE_DATA,config=config,sample_idxs=top_k_sample_indices,fig_title=f"Top {k} Samples with Highest Emergence Scores",style_dict= top_k_style, save_dir=self.results_save_dir, SHOW=SHOW,SAVE=SAVE,save_name=f"{model_name}_{dataset_name}_top_{k}_samples.png")

        return top_k_sample_indices

    def emergence_score_summary_stats_across_scales(self):

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(emergence_score_dist_style)

        selected_paths = self.scaling_paths
        EMERGENCE_METRIC_DATA = {}
        emergence_score_types = ["abs_probs_breakthroughness","abs_negent_breakthroughness","probs_skewness","negent_skewness"]
        metric_names = ["Breakthroughness (p)","Breakthroughness (-H)","Skewness (p)","Skewness (-H)"]
        stats_dict = {}
        for path in selected_paths:
            config, SAVE_DATA =  read_save_data(path)
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            model_size =  get_model_size(path)
            emergence_metric_data = get_samplewise_breakthrough_and_skew(SAVE_DATA, config)
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

    def emergence_scores_dist_across_scales(self, LEGACY=True, n_bootstrap=1000, SHOW=True, SAVE=False, breakthroughness_lims=[-20, 50], skewness_lims=[-4, 4]):
        """
        Plot KDE distributions of emergence scores across different model sizes for a given scaling config.
        Each axes shows a metric, and each curve is a model size.
        """

        plt.rcParams.update(plt.rcParamsDefault)

        if LEGACY:
            plt.rcParams.update(emergence_score_dist_style_legacy)
        else:
            plt.rcParams.update(emergence_score_dist_style)

        PLOT_DATA = {}
        model_sizes = []
        for path in self.scaling_paths:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            model_size = get_model_size(path)
            model_sizes.append(model_size)
            emergence_scores = self.emergence_scores(path=path)
            PLOT_DATA[model_size] = emergence_scores

        # Sort model sizes for consistent plotting
        model_sizes = sorted(list(set(model_sizes)))
        
        metric_types = ['abs_probs_breakthroughness', 'probs_skewness', 'abs_negent_breakthroughness', 'negent_skewness']
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

        if LEGACY:

            fig, axs = plt.subplots(1, 4)
            axs = axs.flatten()
            fig.suptitle('Emergence score distributions - Deepseek-R1-Distill', y=0.98)

            for i, metric in enumerate(metric_types):
                ax = axs[i]
                if "probs" in metric:
                    ax.set_facecolor(PROB_PLOTS_BACKGROUND_COLOR)
                elif "negent" in metric:
                    ax.set_facecolor(NEGENT_PLOTS_BACKGROUND_COLOR)

                if "breakthroughness" in metric: 
                    clip = (0, None)
                elif "skewness" in metric:
                    clip = (None, None)
                else:
                    clip = (None, None)

                for j, model_size in enumerate(model_sizes):
                    try:
                        values = np.array(PLOT_DATA[model_size][metric])
                        values = values[np.isfinite(values)]
                        if len(values) > 1:
                            if "breakthroughness" in metric:  # we renormalize over positive values
                                from scipy.stats import gaussian_kde
                                kde = gaussian_kde(values, bw_method="scott")
                                kde.set_bandwidth(kde.factor * self.kde_bw_adjust)

                                # Evaluate including some negative domain
                                x_max = values.max() + 3 * kde.factor * values.std()
                                x_min = -1  # Start from negative to show it touching 0
                                x_eval = np.linspace(x_min, x_max, 500)
                                density = kde(x_eval)
                                
                                # Set density to 0 for negative values (since breakthroughness is non-negative)
                                density[x_eval < 0] = 0  # FORCE negative domain to 0

                                # Renormalize over full domain
                                integral = np.trapezoid(density, x_eval)
                                density_normalized = density / integral
                                
                                # Plot manually
                                ax.plot(x_eval, density_normalized, color=colors[j % len(colors)], 
                                    linewidth=3, linestyle=line_styles[j % len(line_styles)], 
                                    label=f"{model_size}B", alpha=0.8)
                                ax.fill_between(x_eval, density_normalized, alpha=0.15, 
                                            color=colors[j % len(colors)])

                            else:
                                sns.kdeplot(
                                    values, ax=ax, label=f"{model_size}B",
                                    color=colors[j % len(colors)], alpha=0.15, linewidth=3,
                                    linestyle=line_styles[j % len(line_styles)], fill=True, 
                                    bw_adjust=self.kde_bw_adjust, clip=clip
                                )
                    except Exception as e:
                        print(f"Error plotting {model_size}, {metric}: {e}")

                if "breakthroughness" in metric:
                    ax.set_ylim(0)
                
                ax.set_xlabel(metric_name_map[metric])

                if i == 0:
                    ax.set_ylabel('Density')
                else:
                    ax.set_ylabel('')
                
                ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                ax.set_axisbelow(True)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)
                    spine.set_color('#cccccc')

                # Enforce limits 
                if "breakthroughness" in metric:
                    if "probs" in metric:
                        probability_breakthroughness_lims = [0, 1.5]  # custom set for this one
                        ax.set_xlim(probability_breakthroughness_lims)
                    elif "negent" in metric:
                        ax.set_xlim(negative_entropy_breakthroughness_lims)
                elif "skewness" in metric:
                    if "probs" in metric:
                        ax.set_xlim(probability_skewness_lims)
                    elif "negent" in metric:
                        ax.set_xlim(negative_entropy_skewness_lims)
                
                ax.autoscale(False)

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

        if not LEGACY:

            fig, axs = plt.subplots(2, 2)
            fig.suptitle('Emergence score distributions - Deepseek-R1-Distill', y=0.98)

            # Define layout: probability metrics on left, negentropy on right
            metric_layout = [
                ['abs_probs_breakthroughness', 'abs_negent_breakthroughness'],  # Row 0
                ['probs_skewness', 'negent_skewness']  # Row 1
            ]

            for row in range(2):
                for col in range(2):
                    ax = axs[row, col]
                    metric = metric_layout[row][col]
                    
                    # Set background color
                    if "probs" in metric:
                        ax.set_facecolor(PROB_PLOTS_BACKGROUND_COLOR)
                    elif "negent" in metric:
                        ax.set_facecolor(NEGENT_PLOTS_BACKGROUND_COLOR)

                    # Determine clipping
                    if "breakthroughness" in metric:
                        clip = (0, None)
                    else:
                        clip = (None, None)

                    legend_labels = []
                    
                    for j, model_size in enumerate(model_sizes):

                        model_size_str = str(int(model_size) if model_size.is_integer() else model_size)


                        bootstrapped_emergence_samples = self.bootstrap_emergence_score_mean_ci(model_name=self.scaling_config[0],dataset_name=self.scaling_config[1],model_size=model_size_str,n_bootstrap=n_bootstrap,ci_level=0.95)

                            
                        values = np.array(PLOT_DATA[model_size][metric])
                        values = values[np.isfinite(values)]
                        
                        if len(values) > 1:
                            # Plot KDE
                            if "breakthroughness" in metric:
                                from scipy.stats import gaussian_kde
                                kde = gaussian_kde(values, bw_method="scott")
                                kde.set_bandwidth(kde.factor * self.kde_bw_adjust)

                                x_max = values.max() + 3 * kde.factor * values.std()
                                x_min = -1
                                x_eval = np.linspace(x_min, x_max, 500)
                                density = kde(x_eval)
                                
                                density[x_eval < 0] = 0
                                integral = np.trapezoid(density, x_eval)
                                density_normalized = density / integral
                                
                                ax.plot(x_eval, density_normalized, 
                                    color=colors[j % len(colors)], 
                                    linewidth=3, linestyle=line_styles[j % len(line_styles)], 
                                    alpha=0.8)
                                ax.fill_between(x_eval, density_normalized, alpha=0.15, 
                                            color=colors[j % len(colors)])
                            else:
                                sns.kdeplot(values, ax=ax, 
                                        color=colors[j % len(colors)], alpha=0.15, 
                                        linewidth=3, linestyle=line_styles[j % len(line_styles)], 
                                        fill=True, bw_adjust=self.kde_bw_adjust, clip=clip)
                                
                            # Calculate bootstrap statistics
                            # Note: You'll need to adapt bootstrap_emergence_score_mean_ci to work with scaling paths
                            # This is a placeholder showing the expected structure
                            bootstrap_stats = bootstrapped_emergence_samples[metric]
                            mean = bootstrap_stats['mean']
                            ci_lower = bootstrap_stats['ci_low']
                            ci_upper = bootstrap_stats['ci_high']
                            
                            # Create legend label with statistics
                            legend_label = f"{model_size}B: μ={mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
                            legend_labels.append((colors[j % len(colors)], 
                                                line_styles[j % len(line_styles)], 
                                                legend_label))

                    
                    # Set axis properties
                    if "breakthroughness" in metric:
                        ax.set_ylim(0)
                    
                    ax.set_xlabel(metric_name_map[metric])
                    if col == 0:
                        ax.set_ylabel('Density')
                    else:
                        ax.set_ylabel('')
                    
                    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                    ax.set_axisbelow(True)
                    
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.2)
                        spine.set_color('#cccccc')
                    
                    # Set x-limits
                    if "breakthroughness" in metric:
                        if "probs" in metric:
                            ax.set_xlim([0, 1.5])
                        elif "negent" in metric:
                            ax.set_xlim(negative_entropy_breakthroughness_lims)
                    elif "skewness" in metric:
                        if "probs" in metric:
                            ax.set_xlim(probability_skewness_lims)
                        elif "negent" in metric:
                            ax.set_xlim(negative_entropy_skewness_lims)
                    
                    ax.autoscale(False)
                    
                    # Add legend to each panel (upper right)
                    if legend_labels:
                        from matplotlib.lines import Line2D
                        legend_handles = [Line2D([0], [0], color=c, linestyle=ls, linewidth=2) 
                                        for c, ls, _ in legend_labels]
                        legend_text = [label for _, _, label in legend_labels]
                        ax.legend(legend_handles, legend_text, loc='upper right')

            plt.tight_layout()

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "emergence_score_dist_across_scales"), exist_ok=True)
            save_name = f"emergence_score_dist_across_scales_{self.scaling_config[1]}.png"
            save_path = os.path.join(self.results_save_dir, "emergence_score_dist_across_scales", save_name)
            plt.draw()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        if SHOW:
            plt.show()
        else:
            plt.close()

        return PLOT_DATA

    def emergence_scores_dist_across_scales_old(self, SHOW=True, SAVE=False, breakthroughness_lims=[-20, 50], skewness_lims=[-4, 4]):
        """
        Plot KDE distributions of emergence scores across different model sizes for a given scaling config.
        Each axes shows a metric, and each curve is a model size.
        """

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(emergence_score_dist_style)

        PLOT_DATA = {}
        model_sizes = []
        for path in self.scaling_paths:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            model_size =  get_model_size(path)
            model_sizes.append(model_size)
            emergence_scores = self.emergence_scores(path=path)
            PLOT_DATA[model_size] = emergence_scores

        # Sort model sizes for consistent plotting
        model_sizes = sorted(list(set(model_sizes)))
        metric_types = ['abs_probs_breakthroughness', 'probs_skewness', 'abs_negent_breakthroughness', 'negent_skewness']
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

        fig, axs = plt.subplots(1, 4)
        axs = axs.flatten()
        fig.suptitle('Emergence score distributions - Deepseek-R1-Distill', y=0.98)


        for i, metric in enumerate(metric_types):
            ax = axs[i]
            if "probs" in metric:
                ax.set_facecolor(PROB_PLOTS_BACKGROUND_COLOR)
            elif "negent" in metric:
                ax.set_facecolor(NEGENT_PLOTS_BACKGROUND_COLOR)

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

            if "breakthroughness" in metric:
                ax.set_ylim(0)
            
            title = metric.replace('_', ' ').replace('abs ', 'Absolute ').title()
            title = metric_name_map[metric]
            #ax.set_title(title, pad=15)
            ax.set_xlabel(metric_name_map[metric])

            if i == 0:
                ax.set_ylabel('Density')
            else:
                ax.set_ylabel('')
            
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#cccccc')

            #enforce limits 
            if "breakthroughness" in metric:
                if "probability" in metric:
                    ax.set_xlim(probability_breakthroughness_lims)
                elif "negative_entropy" in metric:
                    ax.set_xlim(negative_entropy_breakthroughness_lims)
            elif "skewness" in metric:
                if "probability" in metric:
                    ax.set_xlim(probability_skewness_lims)
                elif "negative_entropy" in metric:
                    ax.set_xlim(negative_entropy_skewness_lims)

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
            save_name = f"emergence_score_dist_across_scales_{self.scaling_config[1]}.png"
            save_path = os.path.join(self.results_save_dir, "emergence_score_dist_across_scales", save_name)
            plt.draw()
            plt.savefig(save_path,bbox_inches='tight')
            plt.close()

        if SHOW:
            plt.show()

        return PLOT_DATA

    def plot_scaling_curves(self,SHOW=True,SAVE=False,SINGLE_PLOT=False):

        PLOT_DATA = {}

        if not SINGLE_PLOT:
            plt.rcParams.update(plt.rcParamsDefault)
            plt.rcParams.update( scaling_curves_2x2_style)
        else:
            plt.rcParams.update(plt.rcParamsDefault)
            plt.rcParams.update( scaling_curves_style)

        for path in self.scaling_paths:
            model_name, dataset_name = self.get_path_model_and_dataset(path)
            model_size = get_model_size(path)
            config, SAVE_DATA =  read_save_data(path)
            emergence_metric_data = self.emergence_scores(path=path)
            scale_summary_stats = {}
            for metric_name, arr in emergence_metric_data.items():
                arr = np.array(arr)
                minval, maxval = np.nanmin(arr), np.nanmax(arr)

                nan_len = len(arr) - np.sum(np.isnan(arr))
                mean = np.nanmean(arr)
                std = np.nanstd(arr)
                sem = std / np.sqrt(nan_len)
                median = np.nanmedian(arr)
                q75, q25 = np.nanpercentile(arr, [75, 25])
                iqr = q75 - q25

                scale_summary_stats[metric_name] = {
                    'mean': mean,
                    'std': std,
                    'sem': sem,
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

            plt.rcParams.update(plt.rcParamsDefault)
            plt.rcParams.update(scaling_curves_style)
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

            plt.rcParams.update(plt.rcParamsDefault)
            plt.rcParams.update( scaling_curves_2x2_style)
            
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
                    centre_y_label = "Mean"
                    spread_y_label = "SEM"
                    means = [PLOT_DATA[size][metric]['mean'] for size in model_sizes]
                    sems = [PLOT_DATA[size][metric]['sem'] for size in model_sizes]
                    centres = means
                    spreads = sems
                elif "skewness" in metric.lower():
                    centre_y_label = "Mean"
                    spread_y_label = "SEM"
                    means = [PLOT_DATA[size][metric]['mean'] for size in model_sizes]
                    sems = [PLOT_DATA[size][metric]['sem'] for size in model_sizes]
                    centres = means
                    spreads = sems
                else:
                    centres = []
                    spreads = []

                centres = np.array(centres)
                spreads = np.array(spreads)

                if "prob" in metric:
                    axs[idx].set_facecolor(PROB_PLOTS_BACKGROUND_COLOR)
                elif "negent" in metric:
                    axs[idx].set_facecolor(NEGENT_PLOTS_BACKGROUND_COLOR)

                
                axs[idx].errorbar(model_sizes, centres, yerr=spreads, fmt='x-', capsize=3, color=colors[idx % len(colors)])
                axs[idx].set_xlabel("Model Size (B)")
                axs[idx].set_xscale("log")
                axs[idx].set_ylabel(f"{centre_y_label} ± {spread_y_label}")
                axs[idx].set_title(metric_name_mapping[metric])
                axs[idx].set_xticks(model_sizes)
                axs[idx].set_xticklabels(["1.5", "7", "14", "32"])
                axs[idx].grid(True, alpha=0.3)
            
            fig.suptitle("Emergence Metrics vs Model Size - Deepseek-R1-Distill", y=0.98)

        plt.tight_layout()

        

        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "scaling_curves"), exist_ok=True)
            save_name = f"scaling_curve_{config.dataset_name.split('/')[-1]}.png"
            save_path = os.path.join(self.results_save_dir, "scaling_curves", save_name)
            fig.savefig(save_path,bbox_inches='tight')
            plt.close()
        if SHOW:
            plt.show()
        else:
            plt.close()

        return PLOT_DATA

    def get_emergence_score_rankings(self,dataset_name,model_name,rank_by="probability_skewness",legacy_ranking=False):

        """
        Returns task idxs ranked by emergence score rankings 

        e.g: [10,3,5,7,8,...] means that task 1 comes 10th, task 2 comes 3rd, task 3 comes 5th, etc. 1st rankign is highest emergence score.

        Lower ranking means higher total emergence score
        
        """

        assert rank_by in ["probability_breakthroughness", "entropy_breakthroughness", "probability_skewness", "entropy_skewness"], "rank_by must be either 'probability_breakthroughness' or 'entropy_breakthroughness' or 'probability_skewness' or 'entropy_skewness'"
        dict_key_mapping = {
            "probability_breakthroughness": "abs_probs_breakthroughness",
            "entropy_breakthroughness": "abs_negent_breakthroughness",
            "probability_skewness": "probs_skewness",
            "entropy_skewness": "negent_skewness",
        }

        path = self.get_selected_path(model_name=model_name,dataset_name=dataset_name)
        emergence_scores = self.emergence_scores(path=path)

        # Compute rankings for each metric (largest = rank 1, nans get lowest ranking)
        if legacy_ranking:
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

            if rank_by == "probability_breakthroughness":
                ranking = np.stack([rankings[k] for k in rankings if "abs_probs_breakthroughness" in k])
            elif rank_by == "entropy_breakthroughness":
                ranking = np.stack([rankings[k] for k in rankings if "abs_negent_breakthroughness" in k])
            elif rank_by == "probability_skewness":
                ranking = np.stack([rankings[k] for k in rankings if "probs_skewness" in k])
            elif rank_by == "entropy_skewness":
                ranking = np.stack([rankings[k] for k in rankings if "negent_skewness" in k])
            else: 
                raise ValueError(f"Invalid rank_by: {rank_by}")
            ranking = ranking[0]

            #task_ordering = np.argsort(ranking)

            return ranking # lower ranking is better (high emergence scores get first rankings)

        else:
            scores = emergence_scores[dict_key_mapping[rank_by]]
            ranking = np.argsort(np.argsort(-scores))+1 #1-index. -1 term means we rank descending. Still not sur about logic behind this.
            #task_ordering = np.argsort(ranking)
            
            return ranking

    def task_comparison(self,dataset_name,SAVE=False,SHOW=True,rank_by="probability_skewness"):

        import itertools

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(task_comparison_style)


        PLOT_DATA = {}
        for model in self.model_names:
            emergence_score_rankings = self.get_emergence_score_rankings(dataset_name=dataset_name, model_name=model,rank_by=rank_by)
            PLOT_DATA[model] = emergence_score_rankings

        model_names = list(PLOT_DATA.keys())
        pairwise = list(itertools.combinations(model_names, 2))
        n_pairs = len(pairwise)

        fig, axs = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 7), squeeze=False)
        fig.suptitle(f"Task ranking correlation (rank 1 is best)")
        axs = axs[0]

        for idx, (model_a, model_b) in enumerate(pairwise):

            #compute rankings correlation (via pearson on ranks or via spearmn on scores)
            ranking_a = PLOT_DATA[model_a]
            ranking_b = PLOT_DATA[model_b]




            tasks = range(len(ranking_a))
            r, _ = pearsonr(ranking_a, ranking_b)
            r2 = r ** 2


            ax = axs[idx]
            if "gpqa" in dataset_name.lower() and "aggregate" in rank_by.lower():
                #small fix for this special case to prevent ties in rankings
                ranking_a_original = ranking_a
                ranking_b_original = ranking_b
                ranking_a = break_ties_randomly(ranking_a)
                ranking_b = break_ties_randomly(ranking_b)

            ax.scatter(ranking_a, ranking_b, s=250, c='C0', zorder=2)
            # for task, x_val, y_val in zip(tasks, x, y):
            #     ax.text(
            #         x_val, y_val, str(task),
            #         fontsize=18, ha='center', va='center',
            #         color='white',
            #         bbox=dict(facecolor='C0', alpha=0.85, boxstyle='circle,pad=0.25'),
            #         zorder=3
            #     )

            model_a_plot = get_model_plot_name(model_a)
            model_b_plot = get_model_plot_name(model_b)

            if "phi" in model_a_plot.lower():
                model_a_plot = "Phi"
            if "phi" in model_b_plot.lower():
                model_b_plot = "Phi"


            ax.plot([0, len(tasks) - 1], [0, len(tasks) - 1], 'k--', alpha=0.5, zorder=1,marker='x')
            ax.set_xlabel(f"{model_a_plot} ranking", fontweight='bold')
            ax.set_ylabel(f"{model_b_plot} ranking ", fontweight='bold')
            ax.set_title(f"{model_a_plot} vs {model_b_plot}", fontweight='bold', pad=20)
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




            ax.text(
                0.98, 0.02,
                f"$\\rho$ = {r:.2f}",
                transform=ax.transAxes,
                fontsize=18,
                ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='C0', boxstyle='round,pad=0.3'),
                zorder=10
            )


        plt.tight_layout()
        if SAVE:
            os.makedirs(os.path.join(self.results_save_dir, "task_comparison"), exist_ok=True)
            save_name = f"{dataset_name}_task_comparison_{rank_by}.png"
            save_path = os.path.join(self.results_save_dir, "task_comparison", save_name)
            plt.savefig(save_path,bbox_inches='tight')

        if SHOW:
            plt.show()
        else:
            plt.close()

        return PLOT_DATA

    def plot_top_k_tasks_across_models(self,dataset_name,k=4,SHOW=True,SAVE=False,save_dir=None,rank_by="probability_skewness"):
        
        """
        Plot the scaling curves for top k tasks (by mean model ranking) across all models for a dataset.

        Args:
            dataset_name (str): Name of the dataset (e.g., "aime25")
            k (int): Number of top tasks to plot
            SHOW (bool): Whether to display the plot
            SAVE (bool): Whether to save the plot
            save_dir (str or None): Directory to save the figure
            rank_by (str): Which emergence ranking to use
        Returns:
            (None)
        """

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(top_k_style)

        # Collect emergence rankings for each model
        model_emergence_score_rankings = {model_name:None for model_name in self.model_names}
        for model_name in self.model_names:
            rankings = self.get_emergence_score_rankings(
                dataset_name=dataset_name,
                model_name=model_name,
                rank_by=rank_by,
                legacy_ranking=False
            )
            model_emergence_score_rankings[model_name] = rankings

        # Aggregate and determine mean ranking across models
        stacked_rankings = np.stack(
            [model_emergence_score_rankings[model_name] for model_name in self.model_names]
        )  # shape: (n_models, n_tasks)
        mean_ranking = np.mean(stacked_rankings, axis=0)
        topk_tasks = np.argsort(mean_ranking)[:k]

        # For each top-k task, collect metrics for all models
        scaling_curve_data = {
            task_idx.item(): {model_name: None for model_name in self.model_names}
            for task_idx in topk_tasks
        }

        for task_idx in topk_tasks:
            for model_name in self.model_names:
                # Locate save path
                path = [p for p in self.non_scaling_paths if model_name.lower() in p.lower() and dataset_name.lower() in p.lower()]
                if not path:
                    continue
                path = path[0]
                config, SAVE_DATA = read_save_data(path)
                # Get the averaged data over completions
                from .core import get_budget_sample_completion_metrics
                scores, probabilities, negentropies, rankings = get_budget_sample_completion_metrics(
                    SAVE_DATA, config, average_across_completions=True
                )
                # scores.shape: (n_token_budgets, n_samples)
                scores_task = scores[:, task_idx]
                probabilities_task = probabilities[:, task_idx]
                negentropies_task = negentropies[:, task_idx]
                scaling_curve_data[task_idx][model_name] = {
                    "scores": scores_task,
                    "probabilities": probabilities_task,
                    "negentropies": negentropies_task,
                    "token_budgets": sorted(list(SAVE_DATA.keys()))
                }

        # --- Plotting ---
        n_models = len(self.model_names)
        fig, axes = plt.subplots(k, n_models, figsize=(4*n_models, 2.8*k), squeeze=False)
        cmap = plt.cm.get_cmap("tab10")

        # Set font sizes for labels/titles
        title_fontsize = 18
        label_fontsize = 16
        tick_fontsize = 13
        suptitle_fontsize = 22

        for i, model_name in enumerate(self.model_names):
            for j, task_idx in enumerate(topk_tasks):
                ax = axes[j, i]
                ax.grid(alpha=0.5,linestyle='--')
                metric_dict = scaling_curve_data[task_idx][model_name]
                if metric_dict is None:
                    ax.set_visible(False)
                    continue
                scores = metric_dict["scores"]
                probabilities = metric_dict["probabilities"]
                negentropies = metric_dict["negentropies"]
                token_budgets = metric_dict["token_budgets"]
                ln1 = ax.plot(token_budgets, scores, label="Score", color="blue", marker='o')
                ln2 = ax.plot(token_budgets, probabilities, label="Probability", color="red", marker='o')
                ax2 = ax.twinx()
                ln3 = ax2.plot(token_budgets, -negentropies, label="Negentropy", color="green", marker='o', alpha=0.7)
                
                # Set bigger ticklabels
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                ax2.tick_params(axis='y', labelsize=tick_fontsize, labelcolor='green')

                # Column titles: Model name (big)
                if j == 0:
                    ax.set_title(f"{model_name.capitalize()}", fontsize=title_fontsize, pad=16)

                # Row labels: Task index (big, left, vertically centered)
                if i == 0:
                    ax.annotate(
                        f"Task\n{task_idx}",
                        xy=(0, 0.5), xycoords='axes fraction',
                        fontsize=title_fontsize, ha='center', va='center', rotation=90,
                        fontweight='bold', color='black',
                        bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=0.7)
                    )
                # X axis: log2
                ax.set_xscale("log", base=2)
                ax.set_xticks(token_budgets)
                ax.set_xticklabels([str(tb) for tb in token_budgets], rotation=45, fontsize=tick_fontsize)
                ax.set_ylim(-0.1, 1.1)
                # Style right axis for negentropy
                ax2.set_ylabel("Negentropy", color="green", fontsize=label_fontsize)
                ax2.tick_params(axis='y', labelcolor='green')
                # Left axis label only for leftmost column
                if i == 0:
                    ax.set_ylabel("Score / Probability", color="blue", fontsize=label_fontsize)
                    ax.tick_params(axis='y', labelcolor='blue')
                if j == k-1:
                    ax.set_xlabel("Token Budget", fontsize=label_fontsize)
                # Only add a legend for the top-left subfigure
                if i == 0 and j == 0:
                    lines = ln1 + ln2 + ln3
                    labels = ["Score", "Probability", "Negentropy"]
                    fig.legend(
                        lines, labels, 
                        loc='lower center', bbox_to_anchor=(0.5, -0.07), 
                        ncol=3, frameon=False, fontsize=label_fontsize
                    )

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.suptitle(
            f"Scaling Curves for Top-{k} Tasks (by Mean Model Ranking)\nDataset: {dataset_name}",
            y=1.03, fontsize=suptitle_fontsize
        )
        if SAVE:
            save_dir = os.path.join(self.results_save_dir, "top_k_samples") if save_dir is not None else os.path.join("tmp_dir")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"scaling_curves_top_{k}_tasks_ALL_{dataset_name}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
        if SHOW:
            plt.show()
        else:
            plt.close()

        return scaling_curve_data,topk_tasks

    def scaling_aggregate_plots(self,SHOW=True,SAVE=False,save_dir=None):

        negent_lims = [-5,0] #hardcoded for now

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(aggregate_plot_config)

        for path in self.scaling_paths:

            config, SAVE_DATA = read_save_data(path)
            plot_model_name = config.model_name.split("/")[-1]
            plot_dataset_name = config.dataset_name.split("/")[-1]
            fig_title = f"{plot_dataset_name.capitalize()} - {plot_model_name.capitalize()} - (n = {config.num_completions}) "


            if SHOW:
                plot_aggregate_data(SAVE_DATA, config, title=fig_title,logy=False,negent_lims=negent_lims)
            if SAVE:
                os.makedirs(os.path.join(self.results_save_dir, "aggregate_plots"), exist_ok=True)
                save_name = os.path.basename(path).replace(".pkl", "") + "_aggregate.png"
                save_path = os.path.join(self.results_save_dir, "aggregate_plots", save_name)
                plot_aggregate_data(SAVE_DATA, config, title=fig_title,save_path=save_path,logy=False,negent_lims=negent_lims)


    def plot_top_k_across_scales(self, top_k_indices, SHOW=True, SAVE=False, save_dir=None):
        """
        Plots a grid where rows are models/scales (labeled just by their size, e.g., 1.5B, 7B, 14B, 32B),
        and columns are task indices.
        """
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(top_k_style)

        k = len(top_k_indices)
        n_scales = len(self.scaling_paths)

        # Parse scale (model size) info for row labels, e.g., "1.5B", "7B", "14B", "32B"
        def extract_model_size(config):
            # Try attribute 'scale' first, fallback to end of model_name
            scale_str = getattr(config, "scale", None)
            if scale_str is not None:
                return scale_str
            # Try to parse size from model_name string (e.g. "deepseek-32B" or "32B" suffix)
            mn = getattr(config, "model_name", "")
            for tok in reversed(mn.split("/")):
                if "B" in tok:
                    return tok
            # As last resort, use last token
            return mn.split("/")[-1] if "/" in mn else mn

        scales_info = []
        for path in self.scaling_paths:
            config, _ = read_save_data(path)
            size = extract_model_size(config)
            scales_info.append(size)

        # For each top-k task, collect metrics for all scales
        scaling_curve_data = {
            scale_label: {task_idx: None for task_idx in top_k_indices}
            for scale_label in scales_info
        }

        # Now scaling_curve_data[row][col], where
        # rows: model/scale size label (scales_info), cols: task_idx (top_k_indices)

        # Loop over models/scales and gather metrics for each task
        for scale_idx, path in enumerate(self.scaling_paths):
            config, SAVE_DATA = read_save_data(path)
            scale_label = scales_info[scale_idx]
            scores, probabilities, negentropies, rankings = get_budget_sample_completion_metrics(
                SAVE_DATA, config, average_across_completions=True
            )
            for task_idx in top_k_indices:
                if task_idx >= scores.shape[1]:
                    continue  # skip if task_idx is out of bounds
                scores_task = scores[:, task_idx]
                probabilities_task = probabilities[:, task_idx]
                negentropies_task = negentropies[:, task_idx]
                token_budgets = sorted(list(SAVE_DATA.keys()))
                scaling_curve_data[scale_label][task_idx] = {
                    "scores": scores_task,
                    "probabilities": probabilities_task,
                    "negentropies": negentropies_task,
                    "token_budgets": token_budgets
                }

        # --- Plotting ---
        fig, axes = plt.subplots(n_scales, k, figsize=(4 * k, 2.8 * n_scales), squeeze=False)
        cmap = plt.cm.get_cmap("tab10")

        # Set font sizes for labels/titles
        title_fontsize = 18
        label_fontsize = 16
        tick_fontsize = 13
        suptitle_fontsize = 22

        for i, scale_label in enumerate(scales_info):
            for j, task_idx in enumerate(top_k_indices):
                ax = axes[i, j]
                ax.grid(alpha=0.5, linestyle='--')
                metric_dict = scaling_curve_data[scale_label][task_idx]
                if metric_dict is None:
                    ax.set_visible(False)
                    continue
                scores = metric_dict["scores"]
                probabilities = metric_dict["probabilities"]
                negentropies = metric_dict["negentropies"]
                token_budgets = metric_dict["token_budgets"]

                ln1 = ax.plot(token_budgets, scores, label="Score", color="blue", marker='o')
                ln2 = ax.plot(token_budgets, probabilities, label="Probability", color="red", marker='o')
                ax2 = ax.twinx()
                ln3 = ax2.plot(token_budgets, -negentropies, label="Negentropy", color="green", marker='o', alpha=0.7)

                # Set bigger ticklabels
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                ax2.tick_params(axis='y', labelsize=tick_fontsize, labelcolor='green')

                # Column titles: Task index
                if i == 0:
                    ax.set_title(f"Task {task_idx}", fontsize=title_fontsize, pad=16)

                # Row labels: Model/size label
                if j == 0:
                    # Place at middle left of each row
                    ax.annotate(
                        f"{scale_label}",
                        xy=(-0.36, 0.52), xycoords='axes fraction',
                        fontsize=title_fontsize, ha='center', va='center', rotation=90,
                        fontweight='bold', color='black',
                        bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=0.7)
                    )
                ax.set_xscale("log", base=2)
                ax.set_xticks(token_budgets)
                ax.set_xticklabels([str(tb) for tb in token_budgets], rotation=45, fontsize=tick_fontsize)
                ax.set_ylim(-0.1, 1.1)
                # Style right axis for negentropy
                ax2.set_ylabel("Negentropy", color="green", fontsize=label_fontsize)
                ax2.tick_params(axis='y', labelcolor='green')
                # Left axis label only for leftmost column
                if j == 0:
                    ax.set_ylabel("Score / Probability", color="blue", fontsize=label_fontsize)
                    ax.tick_params(axis='y', labelcolor='blue')
                if i == n_scales - 1:
                    ax.set_xlabel("Token Budget", fontsize=label_fontsize)
                # Only add a legend for the top-left subfigure
                if i == 0 and j == 0:
                    lines = ln1 + ln2 + ln3
                    labels_legend = ["Score", "Probability", "Negentropy"]
                    fig.legend(
                        lines, labels_legend,
                        loc='lower center', bbox_to_anchor=(0.5, -0.07),
                        ncol=3, frameon=False, fontsize=label_fontsize
                    )

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.suptitle(
            f"Scaling Curves\nRows: Models ({', '.join(scales_info)}) | Columns: Tasks ({', '.join(map(str,top_k_indices))})",
            y=1.03, fontsize=suptitle_fontsize
        )
        if SAVE:
            save_dir_actual = os.path.join(self.results_save_dir, "top_k_samples") if save_dir is None else save_dir
            os.makedirs(save_dir_actual, exist_ok=True)
            topk_str = "_".join(str(idx) for idx in top_k_indices)
            sizes_str = "_".join(str(s) for s in scales_info)
            save_path = os.path.join(save_dir_actual, f"top_{k}_tasks_across_scales_{topk_str}_rowsizes_{sizes_str}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
        if SHOW:
            plt.show()
        else:
            plt.close()

        return scaling_curve_data


    def get_metric_tables(self,model_name,dataset_name,mode="full_view",score_average_type="mean"):
        
        selected_paths = self.get_selected_path(model_name=model_name,dataset_name=dataset_name)

        config, SAVE_DATA =  read_save_data(selected_paths)


        tables = get_metric_table(config,SAVE_DATA,mode=mode,score_average_type=score_average_type)
        
        return tables
