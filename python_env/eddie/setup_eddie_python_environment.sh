#!/bin/bash


mamba env create -f python_env_vars/environment_uv.yml



#requirements 
echo "Installing requirements with regular pip"
uv pip install --python $(which python) -r python_env_vars/vastai/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r python_env_vars/vastai/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118


#torch environment 
uv pip install --python $(which python) torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

#install vllm with torch-backend flag and uv (quicker)
echo "Installing vllm with torch-backend flag and  (quicker)"
pip install vllm==0.10.1.1 --extra-index-url https://download.pytorch.org/whl/cu118


#echo "Checking installation"
bash python_env_vars/vastai/check_installation.sh

