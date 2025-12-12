from dotenv import load_dotenv
import os, sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

node_type="vastai"
if node_type=="vastai":
    print("Loading vastai environment")
    load_dotenv("envs/.env.vastai",override=True)
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH', '')
elif node_type=="local":
    print("Loading local environment")
    load_dotenv("envs/.env.local",override=True)
else:
    raise ValueError(f"Node type {node_type} not supported")

import utils
import importlib
import numpy as np
from transformers import AutoTokenizer
import logging
import analysis.text_processing
import trace_analysis.utils

from utils import *
from utils.tokenization_setup import *
from analysis.core import read_save_data
from analysis.text_processing import get_text_completion, get_all_text
import trace_analysis.config


importlib.reload(analysis.text_processing)
importlib.reload(trace_analysis.utils)
importlib.reload(trace_analysis.config)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


 
path_deepseek_aime24 = "results_data/DeepSeek-R1-Distill-Qwen-32B_AIME_2024"
config, SAVE_DATA = read_save_data(path_deepseek_aime24)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

trace_config = trace_analysis.config.TraceConfig()
trace_config.DEBUG_MODE = True

model = trace_config.tag_model
client = trace_analysis.utils.setup_api_client(trace_config.API)

#get text, questions, and traces
all_text = get_all_text(SAVE_DATA, config,CLEAN=True)

end_of_input_text = tokenizer.decode(config.end_of_input_tokens, skip_special_tokens=True)
target_token_budgets = [128]
target_sample_idxs = [0,1,2]
target_completion_idxs = range(config.num_completions)


QUESTIONS, REASONING_TRACES = [], []
for token_budget in target_token_budgets:
    for sample_idx in target_sample_idxs:
        for completion_idx in target_completion_idxs:
            text = all_text[token_budget][sample_idx][completion_idx]
            split_text = text.split(end_of_input_text)

            question, trace = split_text[0], split_text[1]
            QUESTIONS.append(question)
            REASONING_TRACES.append(trace)


RESPONSES = trace_analysis.utils.run_llm_api_tagging(trace_config,client, QUESTIONS, REASONING_TRACES, trace_config.system_prompt, max_items=10)




#run tagging 

