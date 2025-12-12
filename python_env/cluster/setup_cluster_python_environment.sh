#!/bin/bash


#make conda available in current shell 
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"


#make mamba avaialble in current shell
eval "$($HOME/miniconda3/bin/mamba shell hook --shell bash)"

# Create environment from yml file
# use mamba for quicker dependency solving
echo "Creating environment with python 3.10 and uv pip"
mamba env create -f python_env_vars/environment_uv.yml

mamba activate ml_env


echo "Installing requirements with uv"
uv pip install --python $(which python) -r python_env_vars/cluster/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

#pytorch ecosystem install 
echo "Installing pytorch ecosystem with cuda 12.1"
uv pip install --python $(which python) torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

#vllm install 
echo "Installing vllm with uv (quicker)"
uv pip install --python $(which python) vllm --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir #no torch-backend auto flag because we set up env on cpu machine 




bash python_env_vars/cluster/check_installation.sh

