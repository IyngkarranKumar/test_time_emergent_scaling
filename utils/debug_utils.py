import re
import statistics
import glob
from collections import defaultdict

def get_all_text(SAVE_DATA,config,token_budget):
    save_token_budget = SAVE_DATA[token_budget]
    texts = []
    sample_idxs = list(range(config.num_samples))
    completion_idxs = list(range(config.num_completions))
    for sample_idx in sample_idxs:
        for completion_idx in completion_idxs:
            texts.append(save_token_budget[sample_idx]['completions'][completion_idx]['text'])
    return texts


def get_time_lines_from_file(log_file):
    """Extract lines containing timing information from a log file."""
    identifiers = ["Total time", "took"]
    time_lines = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if any(identifier in line for identifier in identifiers):
                time_lines.append(line.rstrip('\n'))
    
    return time_lines

def extract_config_from_file(log_file):
    """Extract configuration information from log file."""
    config = {
        'model_name': None,
        'dataset_name': None,
        'num_samples': None,
        'batch_size': None,
        'num_completions': None
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract model name
        model_match = re.search(r'Model Name:\s*(.+)', content)
        if model_match:
            config['model_name'] = model_match.group(1).strip()
        
        # Extract dataset name
        dataset_match = re.search(r'Dataset Name:\s*(.+)', content)
        if dataset_match:
            config['dataset_name'] = dataset_match.group(1).strip()
        
        # Extract number of samples
        samples_match = re.search(r'Number of Samples:\s*(\d+)', content)
        if samples_match:
            config['num_samples'] = int(samples_match.group(1))
        
        # Extract batch size
        batch_match = re.search(r'Batch Size:\s*(\d+)', content)
        if batch_match:
            config['batch_size'] = int(batch_match.group(1))
        
        # Extract number of completions
        completions_match = re.search(r'Number of Completions:\s*(\d+)', content)
        if completions_match:
            config['num_completions'] = int(completions_match.group(1))
    
    return config


def parse_time_stats(log_files):
    """Parse timing statistics from multiple log files."""
    
    # Collect data for each run
    runs_data = []
    
    for run_number, log_file in enumerate(log_files, start=1):
        # Extract config
        config = extract_config_from_file(log_file)
        
        # Extract time lines
        time_lines = get_time_lines_from_file(log_file)
        log_text = '\n'.join(time_lines)
        
        # Extract token budget totals with their line positions
        total_pattern = r'Total time for token budget (\d+): ([\d.]+) seconds'
        total_matches = [(m.start(), m.group(1), m.group(2)) 
                        for m in re.finditer(total_pattern, log_text)]
        
        # Extract scoring times with their line positions
        scoring_pattern = r'Scoring for batch (\d+)/(\d+) took ([\d.]+) seconds'
        scoring_matches = [(m.start(), m.group(1), m.group(2), m.group(3)) 
                          for m in re.finditer(scoring_pattern, log_text)]
        
        # Group scoring times by token budget based on position
        for i, (total_pos, budget, total_time) in enumerate(total_matches):
            start_pos = total_matches[i-1][0] if i > 0 else 0
            end_pos = total_pos
            
            # Collect scoring times in this range
            scoring_times = [float(time) for pos, batch, total_batches, time in scoring_matches 
                           if start_pos < pos < end_pos]
            
            runs_data.append({
                'run_number': run_number,
                'model_name': config['model_name'],
                'dataset_name': config['dataset_name'],
                'num_samples': config['num_samples'],
                'batch_size': config['batch_size'],
                'num_completions': config['num_completions'],
                'token_budget': int(budget),
                'total_time': float(total_time),
                'total_scoring_time': sum(scoring_times),
                'num_batches': len(scoring_times)
            })
    
    return runs_data


def print_results(runs_data):
    """Print formatted results table."""
    
    print("=" * 160)
    print("TOTAL TIME AND SCORING TIME BY RUN AND TOKEN BUDGET")
    print("=" * 160)

    
    if not runs_data:
        print("No data found")
        return
    
    # Print table header (only minutes, not seconds)
    print(f"\n{'Run':<5} {'Model':<50} {'Dataset':<25} {'Samp':<6} {'Comp':<6} {'BS':<5} "
          f"{'Budget':<8} {'Total(m)':<10} {'Score(m)':<10} {'Batch#':<7}")
    print("-" * 120)

    
    # Print each run
    for data in sorted(runs_data, key=lambda x: (x['run_number'], x['token_budget'])):

        model_name = data['model_name'] or 'N/A'
        dataset_name = data['dataset_name'] or 'N/A'

        if model_name == "N/A" or dataset_name == "N/A":
            continue

        print(f"{data['run_number']:<5} "
              f"{model_name:<50} "
              f"{dataset_name:<25} "
              f"{data['num_samples'] if data['num_samples'] is not None else 'N/A':<6} "
              f"{data['num_completions'] if data['num_completions'] is not None else 'N/A':<6} "
              f"{data['batch_size'] if data['batch_size'] is not None else 'N/A':<5} "
              f"{data['token_budget']:<8} "
              f"{data['total_time']/60:<10.2f} "
              f"{data['total_scoring_time']/60:<10.2f} "
              f"{data['num_batches']:<7}")
    if 1: 
        # Print summary statistics
        print("\n" + "=" * 160)
        print("SUMMARY STATISTICS BY TOKEN BUDGET")
        print("=" * 160)
        
        # Group by token budget
        by_budget = defaultdict(list)
        for data in runs_data:
            by_budget[data['token_budget']].append(data)
        
        for budget in sorted(by_budget.keys()):
            budget_data = by_budget[budget]
            total_times = [d['total_time'] / 60 for d in budget_data]
            scoring_times = [d['total_scoring_time'] / 60 for d in budget_data]
            
            print(f"\nToken Budget {budget}:")
            print(f"  Runs: {len(budget_data)}")
            print(f"  Total Time - Mean: {statistics.mean(total_times):.2f}m, "
                  f"Min: {min(total_times):.2f}m, Max: {max(total_times):.2f}m")
            print(f"  Scoring Time - Mean: {statistics.mean(scoring_times):.2f}m, "
                  f"Min: {min(scoring_times):.2f}m, Max: {max(scoring_times):.2f}m")


def extract_config_and_host(job_id):
    error_file = f"slurm_logs/{job_id}/error.txt"
    output_file = f"slurm_logs/{job_id}/output.txt"
    config_lines = []
    host_line = None
    configs_to_track = ["Model Name", "Dataset Name", "Batch Size", "Start Token Budget", "End Token Budget", "Final Answer Budget", "Attention Type"]
    try:
        # Read config from error file
        with open(error_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            in_config = False
            for idx, line in enumerate(lines):
                if "CONFIG DETAILS" in line: #start check  
                    in_config = True
                    config_lines.append(line.rstrip("\n"))
                    continue

                if in_config:
                    if "Run Name" in line:
                        break

                if in_config:
                    for config_to_track in configs_to_track:
                        if config_to_track in line:
                            config_lines.append(line.rstrip("\n"))
                
                if "Number of GPUs" in line:
                    config_lines.append(line.rstrip("\n"))
        # Read host from output file
        with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "host:" in line.lower():
                    host_line = line.rstrip("\n")
                    break
    except Exception as e:
        print(f"Error reading files for job {job_id}: {e}")
    return config_lines, host_line

def get_job_ids_with_pattern(logs_dir="slurm_logs"):
    import glob
    import os
    pattern_match = ["/src/csrc/ops.cu", "/src/csrc/"]
    log_files = glob.glob(os.path.join(logs_dir, "*/error.txt"))
    job_ids_with_pattern = []
    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if any(pattern in content for pattern in pattern_match):
                    # Extract job id from path: slurm_logs/<jobid>/error.txt
                    parts = log_file.split("/")
                    if len(parts) >= 3:
                        job_id = parts[1]
                        job_ids_with_pattern.append(job_id)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    return job_ids_with_pattern

def print_configs_and_hosts_for_src_ops_errors(logs_dir="slurm_logs"):
    import glob
    import os
    pattern_match = ["/src/csrc/ops.cu", "/src/csrc/"]
    configs_to_track = ["Model Name", "Dataset Name", "Batch Size", "Start Token Budget", "End Token Budget", "Final Answer Budget", "Attention Type"]
    log_files = glob.glob(os.path.join(logs_dir, "*/error.txt"))
    job_ids_with_pattern = []
    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if any(pattern in content for pattern in pattern_match):
                    # Extract job id from path: slurm_logs/<jobid>/error.txt
                    parts = log_file.split("/")
                    if len(parts) >= 3:
                        job_id = parts[1]
                        job_ids_with_pattern.append(job_id)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")


    configs_and_hosts = {}
    for job_id in job_ids_with_pattern:
        config, host = extract_config_and_host(job_id)
        configs_and_hosts[job_id] = {
            "host_line": host,
            "config_details": config,
        }

    for job_id, details in configs_and_hosts.items():
        print(f"==== {job_id} ====")
        if details["config_details"]:
            print("\n".join(details["config_details"]))
        if details["host_line"]:
            print(details["host_line"])
        print()

def src_ops_error_print(logs_dir="slurm_logs"):
    print_configs_and_hosts_for_src_ops_errors(logs_dir)
