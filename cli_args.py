# cli_args.py
# Define all CLI arguments for the script



SCRIPT_ARGS = [
    '--config_file',
    '--model_name',
    '--dataset_name',
    '--start_token_budget',
    '--end_token_budget',
    '--num_samples',
    '--batch_size',
    '--inference_engine',
    '--vllm_gpu_memory_utilization',
    '--num_completions',
    '--quantization',
    '--scoring_batch_size',
    '--normalise_over_solution_set',
    '--SAVE_BOOL'
]

def has_script_arguments():
    """Check if any of our script arguments are present in sys.argv"""
    import sys
    return any(arg in sys.argv for arg in SCRIPT_ARGS)