import copy 
import os 
import pickle
import sys 
import numpy as np


from utils import scoring_utils

'''
Run as python3 -m pdb -m analysis.update_save_files
'''


from utils import scoring_utils as scoring_utils
from .core import read_save_data
from utils import workflow_utils
from transformers import AutoTokenizer



PATHS=os.listdir("main_results_data")
PATHS = [p for p in PATHS if "gpqa" in p]
PATHS = ["main_results_data/DeepSeek-R1-Distill-Qwen-1.5B_AIME_2024","main_results_data/DeepSeek-R1-Distill-Qwen-7B_AIME_2024","main_results_data/DeepSeek-R1-Distill-Qwen-14B_AIME_2024"]


for path in PATHS:
    #path = os.path.join("main_results_data", path)
    
    
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config, SAVE_DATA = read_save_data(path)
    SAVE_DATA_keys = list(SAVE_DATA.keys())


    if 0: #true token count 
        print(f"Updating save files (TOKEN COUNT) for {path}")

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        token_budgets =  sorted([k for k in SAVE_DATA_keys if isinstance(k, int)])
        sample_idxs = sorted([k for k in SAVE_DATA[token_budgets[0]].keys() if isinstance(k, int)])
        if "metrics" in sample_idxs: sample_idxs.remove("metrics")
        if "recomputed_metrics" in sample_idxs: sample_idxs.remove("recomputed_metrics")
        completions = np.arange(config.num_completions)


        for token_budget in token_budgets:
            for sample_idx in sample_idxs:
                for completion_idx in completions:
                    text = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['text'][0]
                    encoded_text = tokenizer.encode(text, add_special_tokens=False)
                    end_of_input_idxs = workflow_utils.get_end_of_input_idxs(encoded_text, tokenizer)[0]
                    token_count = workflow_utils.count_generated_tokens(encoded_text, end_of_input_idxs, tokenizer)
                    SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['token_count'] = token_count
        
        # Save updated SAVE_DATA for each token budget
        for token_budget in token_budgets:
            num_samples = len([k for k in SAVE_DATA[token_budget].keys() if isinstance(k, int)])
            save_path = os.path.join(
                path, f"save_file__budget_{token_budget}.pkl"
            )
            # Only save the data for this budget
            with open(save_path, 'wb') as f:
                pickle.dump({token_budget: SAVE_DATA[token_budget]}, f)
            print(f"Saved token counts for {path} budget {token_budget} "
                f"({num_samples} samples) to {save_path}")
            


    if 1: #recompute metrics
        # Collect all token_budget keys (integer keys, not e.g. 'metrics')
        token_budget_keys = [k for k in SAVE_DATA_keys if isinstance(k, int)]

        for token_budget in token_budget_keys:

            # Get number of samples in this budget, from 'samples_idx'
            samples_idx = list(SAVE_DATA[token_budget].keys())
            if 'metrics' in samples_idx:
                samples_idx.remove('metrics')
            if 'recomputed_metrics' in samples_idx:
                samples_idx.remove('recomputed_metrics')
            num_samples = len(samples_idx)


            SAVE_DATA_resave_path = os.path.join(
                path, f"save_file__budget_{token_budget}.pkl"
            )

            recomputed_metrics_exist = "recomputed_metrics" in list(SAVE_DATA[token_budget].keys())

            # Check if "recomputed_metrics" already present for this token_budget
            if recomputed_metrics_exist:
                #raise ValueError(f"Recomputed metrics already exist for {path} budget {token_budget}")
                print(f"Recomputed metrics already exist for {path} budget {token_budget} - overwriting...")
                pass
            else:
                print(f"No recomputed metrics found for {path} budget {token_budget}")

            # Recompute metrics for just this budget
            recomputed_metrics = scoring_utils.recompute_metrics({token_budget: SAVE_DATA[token_budget]}, config)

            # Attach the recomputed metrics
            SAVE_DATA[token_budget]['recomputed_metrics'] = recomputed_metrics[token_budget]


            # Make sure any previous file is removed before saving, to guarantee overwrite
            if os.path.exists(SAVE_DATA_resave_path):
                try:
                    os.remove(SAVE_DATA_resave_path)
                except Exception as e:
                    print(f"Warning: Could not remove existing file {SAVE_DATA_resave_path}: {e}")
            with open(SAVE_DATA_resave_path, 'wb') as f:
                pickle.dump({token_budget: SAVE_DATA[token_budget]}, f)
                f.flush()
                os.fsync(f.fileno())

            print(f"Saved recomputed metrics for {path} budget {token_budget} "
                f"({num_samples} samples) to {SAVE_DATA_resave_path}")
                


        # Delete all files within the dir that isn't config.pkl OR doesn't have 'recomputed' in it
        if 0: 
            for fname in os.listdir(path):
                if fname == "config.pkl":
                    continue
                if "recomputed" in fname:
                    continue
                file_path = os.path.join(path, fname)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Could not delete {file_path}: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

