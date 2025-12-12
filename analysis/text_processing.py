import pandas as pd
import os
from tempfile import NamedTemporaryFile
import webbrowser
import numpy as np
import json 


from transformers import AutoTokenizer
from utils.dataset_utils import DatasetSetup
from utils.tokenization_setup import *


def add_delimiters_to_text(df, token_budgets, config, pad_token="<pad>",mode="token_budget_view"):

    """Add delimiters to text to show question boundaries and token budget cutoffs"""

    dataset_setup = DatasetSetup(config.dataset_name,config=config)
    dataset = dataset_setup.load_dataset()
    if "Qwen2.5" in config.model_name or "simplescaling" in config.model_name:
        end_of_input_ids = qwen_end_of_input_ids
    elif "google/gemma" in config.model_name:
        end_of_input_ids = gemma_end_of_input_ids
    elif "deepseek" in config.model_name:
        end_of_input_ids = deepseek_end_of_input_ids
    elif "QwQ" in config.model_name:
        end_of_input_ids = qwq_end_of_input_ids
    elif "Phi-4" in config.model_name:
        end_of_input_ids = phi_end_of_input_ids
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
    elif "QwQ" in config.model_name:
        pad_token = "<|endoftext|>"
    elif "Phi-4" in config.model_name:
        pad_token = "<|dummy_85|>"
    else:
        pad_token = "<pad>"
    df_with_delims = add_delimiters_to_text(df, token_budgets, config,mode="completions_view", pad_token=pad_token)

    df_to_html(df_with_delims,save=save)


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

def get_all_text(SAVE_DATA,config,CLEAN=True):

    if "phi" in config.model_name.lower():
        pad_token = "<|dummy_85|>"
    else:
        pad_token = "!"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    special_tokens_list = tokenizer.all_special_tokens
    special_tokens_list.append(pad_token)

    token_budgets = list(SAVE_DATA.keys())
    sample_idxs = list(SAVE_DATA[token_budgets[0]].keys())
    sample_idxs = [a for a in sample_idxs if isinstance(a, int)]
    all_text = []
    all_text_dict = {}
    for token_budget in token_budgets:
        all_text_dict[token_budget] = {}
        sample_idxs = list(SAVE_DATA[token_budget].keys())
        sample_idxs = [a for a in sample_idxs if isinstance(a, int)]
        if "metrics" in sample_idxs: 
            sample_idxs.remove("metrics")
        if "recomputed_metrics" in sample_idxs:
            sample_idxs.remove("recomputed_metrics")
        for sample_idx in sample_idxs:
            all_text_dict[token_budget][sample_idx] = {}
            for completion_idx in range(config.num_completions):
                text = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['text'][0]
                if CLEAN:
                    for special_token in special_tokens_list:
                        text = text.replace(special_token, "")
                all_text_dict[token_budget][sample_idx][completion_idx] = text

    return all_text_dict

def get_text_completion(SAVE_DATA,token_budgets=[],sample_idxs=[],completion_idxs=[],padding_token="!"):

    text_completions = []
    for token_budget in token_budgets:
        sample_idxs = list(SAVE_DATA[token_budget].keys())
        if "metrics" in sample_idxs: 
            sample_idxs.remove("metrics")
        if "recomputed_metrics" in sample_idxs:
            sample_idxs.remove("recomputed_metrics")

        for sample_idx in sample_idxs:
            for completion_idx in completion_idxs:
                text = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['text'][0]
                text = text.replace(padding_token, "")
                text_completions.append(text)
    return text_completions


def save_text(text_data,run_name):
    save_dir = "text_data"
    path = os.path.join(save_dir + f"{run_name}.json")
    with open(path, "w",encoding="utf-8") as f:
        json.dump(text_data, f, ensure_ascii=False, indent=2)

def load_text(path):

    with open(path, "r",encoding="utf-8") as f:
        text_data = json.load(f)
    return text_data