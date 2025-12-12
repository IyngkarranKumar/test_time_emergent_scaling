import analysis
import importlib
from transformers import AutoTokenizer
import numpy as np
import sys
import os
import json

importlib.reload(analysis.misc)
importlib.reload(analysis.core)
importlib.reload(analysis.text_processing)
importlib.reload(analysis)


path = "results_data/RESULTS/SCALING/DeepSeek-R1-Distill-Qwen-14B_gpqa_10-20_04-37-49"
config,SAVE_DATA = analysis.read_save_data(path)



if 1: 
    token_buget_to_view = 8192
    
    analysis.token_budget_text_view(config=config,SAVE_DATA=SAVE_DATA,token_budget=token_buget_to_view,save=False)
    
    #analysis.completions_text_view(config=config,SAVE_DATA=SAVE_DATA,token_budget=token_buget_to_view,save=False)

    all_text = analysis.text_processing.get_all_text(SAVE_DATA,config)

    save_dir = "text_data"
    path = os.path.join(save_dir,f"{config.run_name}.json")

    if not os.path.exists(path): 
        print(f"Saving text data to {path}")
        with open(path, "w",encoding="utf-8") as f:
            json.dump(all_text, f, ensure_ascii=False, indent=2)

if 1: 
    token_count_table = analysis.get_token_count_table(config,SAVE_DATA,mode="full_view")
    metric_table_names = ["answer_score", "answer_probability", "answer_entropy", "answer_ranking","solution_set_distribution"]
    token_budget=8192

    metric_tables = {}

    for metric_table_name in metric_table_names:
        metric_tables[metric_table_name] = analysis.get_metric_table(config,SAVE_DATA,metric_type=metric_table_name,mode="full_view")

    #solution_set_max = np.array([max(original_metric_tables["solution_set_distribution"][token_budget][sample_idx][completion_idx]) for sample_idx in original_metric_tables["solution_set_distribution"].index for completion_idx in range(config.num_completions)])


if 1: 
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    end_of_input_string = tokenizer.decode(config.end_of_input_tokens,skip_special_tokens=False)
    results, _, _ = analysis.find_repeating_sequences(SAVE_DATA,config,end_of_input_string,tokenizer,pad_token=tokenizer.pad_token, k=5)

    key = 8192 
    _ = [r for r in results[key] if r['num_force_continues'] == 0]
