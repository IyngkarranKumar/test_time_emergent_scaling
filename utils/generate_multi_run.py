#!/usr/bin/env python3
"""
Generate multi-run bash script from YAML configuration
Usage: python generate_multi_run.py multi_run.yaml
"""

import yaml
import argparse
import sys
from itertools import product
from pathlib import Path

def load_config(yaml_file):
    """Load YAML configuration"""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_grid_and_fixed_params(config):
    """Separate list parameters (grid) from single parameters (fixed)"""
    grid_params = {}
    fixed_params = {}
    
    for key, value in config.items():
        if isinstance(value, list):
            grid_params[key] = value
        else:
            fixed_params[key] = value
    
    return grid_params, fixed_params

def generate_bash_script(config, output_file="run_experiments.sh"):
    """Generate bash script from configuration"""
    
    grid_params, fixed_params = extract_grid_and_fixed_params(config)
    
    # Start building the script
    script_lines = [
        "#!/bin/bash",
        "",
        "# Auto-generated multi-run script",
        "# Generated from multi_run.yaml",
        "",
    ]
    
    # If no grid parameters, just run once with fixed parameters
    if not grid_params:
        script_lines.extend(generate_single_command(fixed_params))
        
    else:
        # Generate cartesian product of grid parameters
        grid_keys = list(grid_params.keys())
        grid_values = [grid_params[key] for key in grid_keys]
        
        total_combinations = 1
        for values in grid_values:
            total_combinations *= len(values)
        
        script_lines.append(f"# Total combinations: {total_combinations}")
        script_lines.append("")
        
        experiment_num = 1
        for combination in product(*grid_values):
            # Create command for this combination
            experiment_params = fixed_params.copy()
            
            # Add grid parameters
            for key, value in zip(grid_keys, combination):
                experiment_params[key] = value
            
            # Add experiment header
            script_lines.append(f"# Experiment {experiment_num}/{total_combinations}")
            
            # Add parameter info as comment
            grid_info = []
            for key, value in zip(grid_keys, combination):
                grid_info.append(f"{key}={value}")
            script_lines.append(f"# {', '.join(grid_info)}")
            
            # Generate command
            script_lines.extend(generate_single_command(experiment_params))
            script_lines.append("")
            
            experiment_num += 1
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(script_lines))
    
    # Make executable
    Path(output_file).chmod(0o755)
    
    print(f"Generated bash script: {output_file}")
    print(f"Total experiments: {experiment_num - 1}")
    
    return output_file

def generate_single_command(params):
    """Generate a single python command with parameters"""
    
    cmd_lines = ["python3 main.py \\"]
    
    # Convert parameter names and format values
    for key, value in params.items():
        #we're assuming all param names in .yaml match config.py
        param_name = key
        
        # Format the value
        if isinstance(value, bool):
            if value:
                cmd_lines.append(f"        --{param_name} \\")
            # Don't add anything for False boolean flags
        elif isinstance(value, str):
            cmd_lines.append(f"        --{param_name} {value} \\")
        else:
            cmd_lines.append(f"        --{param_name} {value} \\")
    
    # Remove trailing backslash from last line
    if cmd_lines[-1].endswith(' \\'):
        cmd_lines[-1] = cmd_lines[-1][:-2]
    
    cmd_lines.append("")  # Empty line after command
    
    return cmd_lines

def main():
    parser = argparse.ArgumentParser(description="Generate multi-run bash script from YAML")
    parser.add_argument("yaml_file", help="Path to YAML configuration file")
    parser.add_argument("-o", "--output", default="scripts/run_experiments.sh", 
                       help="Output bash script filename (default: run_experiments.sh)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print script to stdout instead of writing to file")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.yaml_file)
    except FileNotFoundError:
        print(f"Error: Could not find {args.yaml_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        sys.exit(1)
    
    # Generate script
    if args.dry_run:
        # Print to stdout
        grid_params, fixed_params = extract_grid_and_fixed_params(config)
        if not grid_params:
            for line in generate_single_command(fixed_params):
                print(line)
        else:
            grid_keys = list(grid_params.keys())
            grid_values = [grid_params[key] for key in grid_keys]
            
            for i, combination in enumerate(product(*grid_values), 1):
                experiment_params = fixed_params.copy()
                for key, value in zip(grid_keys, combination):
                    experiment_params[key] = value
                
                print(f"# Experiment {i}")
                for line in generate_single_command(experiment_params):
                    print(line)
    else:
        generate_bash_script(config, args.output)

if __name__ == "__main__":
    main()