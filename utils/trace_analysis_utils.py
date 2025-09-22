"""
Utility functions for trace analysis operations.
"""
import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def setup_api_client(api_type="openrouter"):
    """
    Setup API client based on specified type.
    
    Args:
        api_type: Either "openai" or "openrouter"
        
    Returns:
        Tuple of (client, model_name)
    """
    from openai import OpenAI

    API_TYPES = ["openai", "openrouter"]
    
    assert api_type in API_TYPES, f"API must be one of {API_TYPES}"
    
    if api_type == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif api_type == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    else:
        raise ValueError(f"API must be one of {API_TYPES}")
        
    return client


def extract_questions_and_traces(texts: List[str], tokenizer: AutoTokenizer, 
                                end_of_input_text: str) -> Tuple[List[str], List[str]]:
    """
    Extract questions and reasoning traces from text completions.
    
    Args:
        texts: List of text completions
        tokenizer: Tokenizer for processing text
        end_of_input_text: Delimiter text separating question from reasoning
        
    Returns:
        Tuple of (questions, reasoning_traces)
    """
    questions, reasoning_traces = zip(*[
        (text.split(end_of_input_text)[0].strip(), 
         text.split(end_of_input_text)[-1].strip())
        for text in texts
    ])
    
    # Remove BOS and other special tokens
    questions = [tokenizer.decode(tokenizer.encode(q), skip_special_tokens=True) 
                for q in questions]
    reasoning_traces = list(reasoning_traces)
    
    return questions, reasoning_traces


def load_prompts(prompt_path: str) -> Tuple[str, str]:
    """
    Load system and user prompt templates from file.
    
    Args:
        prompt_path: Path to prompt template file
        
    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    with open(prompt_path, "r") as f:
        output = f.read()
    
    system_prompt = output.split("SYSTEM_PROMPT=")[1].split("\n")[0].strip()
    user_prompt_template = output.split("USER_PROMPT=")[1].split("=")[-1].strip()
    
    return system_prompt, user_prompt_template


def run_llm_analysis(client: Any, model: str, questions: List[str], 
                    reasoning_traces: List[str], system_prompt: str, 
                    user_prompt_template: str, max_items: int = None) -> List[str]:
    
    """
    Run LLM analysis on questions and reasoning traces.
    
    Args:
        client: OpenAI client instance
        model: Model name to use
        questions: List of input questions
        reasoning_traces: List of reasoning traces
        system_prompt: System prompt for LLM
        user_prompt_template: Template for user prompts
        max_items: Maximum number of items to process (None for all)
        
    Returns:
        List of LLM responses
    """

    responses = []

    num_items = len(questions) if max_items is None else min(max_items, len(questions))

    run_llm = input(f"Do you want to run the LLM API on {num_items} prompts? (y/n): ").strip().lower()
    if run_llm not in ["y", "yes"]:
        import sys
        sys.exit('Exiting without running LLM API.')
    
    
    for idx, (input_prompt, reasoning_trace) in enumerate(zip(questions[:num_items], 
                                                             reasoning_traces[:num_items])):
        logger.info(f"Running LLM API on prompt {idx} of {num_items}")
        
        user_prompt = user_prompt_template.replace("INSERT_INPUT_PROMPT", input_prompt)\
                                         .replace("INSERT_REASONING_TRACE", reasoning_trace)
        
        messages = [
            {
                "role": "developer",
                "content": system_prompt,
            },
            {
                "role": "user", 
                "content": user_prompt,
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        response_text = response.choices[0].message.content
        responses.append(response_text)
    
    return responses


def save_analysis_results(questions: List[str], reasoning_traces: List[str], 
                         responses: List[str], output_dir: str = "trace_analysis_outputs") -> str:
    """
    Save analysis results to JSON file.
    
    Args:
        questions: List of input questions
        reasoning_traces: List of reasoning traces
        responses: List of LLM responses
        output_dir: Directory to save output file
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = [
        {
            "question": question,
            "reasoning_trace": reasoning_trace,
            "response": response
        }
        for question, reasoning_trace, response in zip(questions, reasoning_traces, responses)
    ]
    
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"trace_analysis_results_{dt_str}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_path


def analyze_trace_identifiers(annotated_traces: List[Dict], trace_identifiers: Dict[str, str], 
                             tokenizer: AutoTokenizer) -> List[Dict]:
    """
    Analyze trace identifiers in annotated traces.
    
    Args:
        annotated_traces: List of trace dictionaries
        trace_identifiers: Dictionary mapping identifier names to tags
        tokenizer: Tokenizer for counting tokens
        
    Returns:
        Updated annotated_traces with identifier analysis
    """
    for trace_idx, trace in enumerate(annotated_traces):
        response = trace["response"]
        
        # Count identifier occurrences
        identifier_counts = {}
        for name, identifier in trace_identifiers.items():
            start_tag = identifier
            end_tag = identifier.replace("<", "</")
            start_count = response.count(start_tag)
            end_count = response.count(end_tag)
            count = min(start_count, end_count)
            identifier_counts[name] = count
        trace["identifier_counts"] = identifier_counts
        
        # Count tokens within identifiers
        identifier_token_counts = {}
        for name, identifier in trace_identifiers.items():
            start_tag = identifier
            end_tag = identifier.replace("<", "</")
            pattern = re.compile(
                re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
                re.DOTALL | re.IGNORECASE
            )
            matches = pattern.findall(response)
            token_counts = []
            for match in matches:
                tokens = tokenizer.encode(match, add_special_tokens=False)
                token_counts.append(len(tokens))
            identifier_token_counts[name] = token_counts
        trace["identifier_token_counts"] = identifier_token_counts
    
    return annotated_traces


def print_trace_analysis(annotated_traces: List[Dict]) -> None:
    """
    Print trace analysis results in a formatted way.
    
    Args:
        annotated_traces: List of analyzed trace dictionaries
    """
    for idx, trace in enumerate(annotated_traces):
        print(f"Trace {idx}:")
        print("  Identifier Counts:")
        for key, value in trace["identifier_counts"].items():
            print(f"    {key}: {value}")
        print("  Identifier Token Counts:")
        for key, value in trace["identifier_token_counts"].items():
            print(f"    {key}: {value}")
        print("-" * 40)


def load_annotated_traces(file_path: str) -> List[Dict]:
    """
    Load annotated traces from JSON file.
    
    Args:
        file_path: Path to JSON file containing annotated traces
        
    Returns:
        List of annotated trace dictionaries
    """
    with open(file_path, "r") as f:
        return json.load(f)