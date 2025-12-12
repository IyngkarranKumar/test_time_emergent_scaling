import pickletools
import numpy as np 
from scipy.stats import rankdata

from .core import breakthrough_score, differences_skew_score

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
            if sample_idx == 'metrics' or sample_idx == 'recomputed_metrics':
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
                force_continue_splits = generated_text_only.split(config.force_continue)


                results[token_budget].append({
                    'sample_idx': sample_idx, 
                    'completion_idx': completion_idx, 
                    'num_answer_indicators': num_answer_indicators, 
                    'num_force_continues': num_force_continues,
                    'n_generated_tokens': number_of_generated_tokens,
                    'force_continue_splits': force_continue_splits
                })
                texts_with_counts.append({
                    'sample_idx': sample_idx, 
                    'completion_idx': completion_idx, 
                    'text': generated_text_only, 
                    'num_answer_indicators': num_answer_indicators,
                    'num_force_continues': num_force_continues,
                    'n_generated_tokens': number_of_generated_tokens,
                    'force_continue_splits': force_continue_splits
                })

    # Sort by num_answer_indicators descending
    top_k_by_answer_indicators = sorted(texts_with_counts, key=lambda x: x['num_answer_indicators'], reverse=True)[:k]
    top_k_by_force_continues = sorted(texts_with_counts, key=lambda x: x['num_force_continues'], reverse=True)[:k]


    return results, top_k_by_answer_indicators, top_k_by_force_continues



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
    
    
def compute_emergence_scores(
    metric_table,
    breakthroughness_types=["median_squared", "mean_squared", "mean_biquadrate"],
    skewness_weighting=[False, True],
    square_numerator=[False, True],
):

    n_samples = metric_table.shape[0]

    x = np.array(metric_table.columns)
    y = np.array(metric_table.values, dtype=np.float64)

    metric_table_with_emergence = metric_table.copy()

    # Handle breakthroughness_types and square_numerator, both allow lists or single values
    breakthroughness_types_list = (
        [breakthroughness_types] if not isinstance(breakthroughness_types, list) else breakthroughness_types
    )
    square_numerator_list = (
        [square_numerator] if not isinstance(square_numerator, list) else square_numerator
    )

    for da in breakthroughness_types_list:
        for sq_num in square_numerator_list:
            col_name = (
                f'breakthroughness_{da}_squared_numerator' if sq_num else f'breakthroughness_{da}'
            )
            metric_table_with_emergence[col_name] = breakthrough_score(
                x, y, diff_average=da, square_numerator=sq_num
            )

    # For backward compatibility: fill 'breakthroughness' column with legacy=True only

    metric_table_with_emergence["breakthroughness"] = breakthrough_score(
        np.array(metric_table.columns),
        np.array(metric_table.values, dtype=np.float64),
        legacy=True
    )

    # Compute skewness (possibly weighted and unweighted) based on skewness_weighting parameter
    if isinstance(skewness_weighting, list):
        for weighted in skewness_weighting:
            col_name = f"skewness_weighted" if weighted else "skewness"
            metric_table_with_emergence[col_name] = differences_skew_score(y, magnitude_weight=weighted)
        # If only one was requested, also assign 'skewness' main column name for backward compatibility
        if len(skewness_weighting) == 1:
            skewness = metric_table_with_emergence[f"skewness_weighted" if skewness_weighting[0] else "skewness"]
        else:
            # By convention, set 'skewness' column to unweighted if available
            if "skewness" in metric_table_with_emergence.columns:
                skewness = metric_table_with_emergence["skewness"]
            else:
                skewness = metric_table_with_emergence["skewness_weighted"]
    else:
        skewness = differences_skew_score(y, magnitude_weight=skewness_weighting)
    
    metric_table_with_emergence['skewness'] = skewness

    # Ensure all emergence score columns (those starting with 'breakthroughness') are floats and not in scientific notation

    from decimal import Decimal

    for col in metric_table_with_emergence.columns:

        metric_table_with_emergence[col] = metric_table_with_emergence[col].apply(
                    lambda v: float(f"{float(v):f}")
                )


    return metric_table_with_emergence



def break_ties_randomly(ranks, seed=42):
    """
    Given ranks with possible ties, randomly break the ties.
    Example: [5, 2, 3, 3, 1] -> [5, 2, 3, 4, 1] or [5, 2, 4, 3, 1]
    """
    np.random.seed(seed)
    ranks = np.array(ranks)
    
    # Create random tiebreakers
    tiebreaker = 0.1*np.random.random(len(ranks))
    
    # Sort by (rank, tiebreaker) to determine final order
    sorted_indices = np.lexsort((tiebreaker, ranks))
    
    # Assign new ranks 1, 2, 3, ... based on this order
    final_ranks = np.empty(len(ranks), dtype=int)
    final_ranks[sorted_indices] = np.arange(1, len(ranks) + 1)

    
    return final_ranks