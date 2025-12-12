#!/bin/bash

#make conda available in current shell 
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"


#make mamba avaialble in current shell
eval "$($HOME/miniconda3/bin/mamba shell hook --shell bash)"

# Create environment from yml file
# use mamba for quicker dependency solving
mamba env create -f python_env_vars/environment_uv.yml

eval "$($HOME/miniconda3/bin/mamba shell hook --shell bash)"

mamba activate ml_env


#requirements 
echo "Installing requirements with uv"
uv pip install --python $(which python) -r python_env_vars/vastai/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118


#torch environment 
uv pip install --python $(which python) torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

#install vllm with torch-backend flag and uv (quicker)
echo "Installing vllm with torch-backend flag and uv (quicker)"
uv pip install --python $(which python) vllm --torch-backend=auto --extra-index-url https://download.pytorch.org/whl/cu118

uv pip install --python $(which python) --upgrade vllm==0.10.1.1 #this vllm seems to work. not vllm 0.5.x



#echo "Checking installation"
bash python_env_vars/vastai/check_installation.sh

