#!/bin/bash

# #SLURM setup 
# #SLURM basically puts us on a machine with our specs
#SBATCH --job-name=check_hf_models
#SBATCH --output=slurm_logs/%j/output.txt
#SBATCH --error=slurm_logs/%j/error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=PGR-Standard,PGR-Standard-Noble,Teach-Standard
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00

# Define models to test
MODELS=(
    "Qwen/Qwen2.5-32B-Instruct"
)



#source .rc file 
source ~/.bashrc

#print slurm info 
echo "Job running on ${SLURM_JOB_NODELIST}"

#env activate 
mamba activate ml_env

# setup machine env vars
source scripts/setup_shell_environment.sh


# Copy models from HF cache
for model in "${MODELS[@]}"; do
    # Convert model name to HF cache format (org/model -> models--org--model)
    cache_name="models--${model//\//--}"
    
    echo "Copying ${model} from HF cache..."
    rsync -ah --info=progress2 "$HEAD_NODE_HF_CACHE_PATH/$cache_name" "$HF_CACHE_PATH/"
done


#start 
echo "Starting HF models check"

#run the script
python3 -m utils.check_hf_models "${MODELS[@]}" > logs/check_hf_models_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1

#finish 
echo "HF models check finished"