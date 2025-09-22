#%% imports
import utils
import importlib
import numpy as np
from transformers import AutoTokenizer
import logging

from dotenv import load_dotenv
from utils.trace_analysis_utils import (
    setup_api_client,
    extract_questions_and_traces,
    load_prompts,
    run_llm_analysis,
    save_analysis_results,
    analyze_trace_identifiers,
    print_trace_analysis,
    load_annotated_traces
)

load_dotenv("envs/.env.local")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#%% setup api 

API = "openrouter"
model = "deepseek/deepseek-r1-distill-llama-70b:free"
client = setup_api_client(API)

#%% load raw data


path_gpqa_deepseek_7b_distill = "results_data/DeepSeek-R1-Distill-Qwen-7B_gpqa_08-22_20-01-27"
config_gpqa_deepseek_7b_distill, SAVE_DATA_gpqa_deepseek_7b_distill = utils.read_save_data(path_gpqa_deepseek_7b_distill)


#%% get reasoning traces

importlib.reload(utils)
importlib.reload(utils.analysis_utils)




ALL_TEXTS=False
padding_token = "!"
tokenizer = AutoTokenizer.from_pretrained(config_gpqa_deepseek_7b_distill.model_name)


if ALL_TEXTS:
    token_budgets = list(SAVE_DATA_gpqa_deepseek_7b_distill.keys())
    sample_idxs = np.arange(config_gpqa_deepseek_7b_distill.num_samples)
    completion_idxs = np.arange(config_gpqa_deepseek_7b_distill.num_completions)

else:
    token_budgets = [1024]
    sample_idxs = [0,1,2]
    completion_idxs = [0,1,2]

deepseek_end_of_input_text = tokenizer.decode(utils.deepseek_end_of_input_ids, skip_special_tokens=True)

texts = utils.get_text_completion(SAVE_DATA_gpqa_deepseek_7b_distill, config_gpqa_deepseek_7b_distill, token_budgets, sample_idxs, completion_idxs, remove_padding_token=True, padding_token=padding_token)

QUESTIONS, REASONING_TRACES = extract_questions_and_traces(texts, tokenizer, deepseek_end_of_input_text)


#%% setup user and system prompt

import utils
importlib.reload(utils)
importlib.reload(utils.trace_analysis_utils)

trace_analysis_prompt_path = "prompts/trace_decomposition.txt"
SYSTEM_PROMPT, USER_PROMPT_TEMPLATE = load_prompts(trace_analysis_prompt_path)

RESPONSES = run_llm_analysis(client, model, QUESTIONS, REASONING_TRACES, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, max_items=2)

#%% save

output_path = save_analysis_results(QUESTIONS, REASONING_TRACES, RESPONSES)

#%% analysis

trace_identifiers = {
    "definition": "<definition>",
    "bloom": "<bloom>",
    "reconstruction": "<reconstruction>", 
    "final_answer": "<final>",
}

annotated_trace_path = 'trace_analysis_outputs/trace_analysis_results_2025-09-15_22-58-21.json'
annotated_traces = load_annotated_traces(annotated_trace_path)

annotated_traces = analyze_trace_identifiers(annotated_traces, trace_identifiers, tokenizer)

print_trace_analysis(annotated_traces)
