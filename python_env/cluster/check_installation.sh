echo "🧪 Testing installation..."
python -c "
import os
print('Testing imports')
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

load_dotenv('envs/.env.mlp_compute')

print(f'✅ PyTorch version: {torch.__version__}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'PyTorch compiled with CUDA: {torch.version.cuda is not None}')

print('Testing vllm model loading')
model_snapshot_path = f\"{os.getenv('HF_CACHE_PATH')}/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/\"
llm = LLM(model=model_snapshot_path, download_dir=os.getenv('HF_CACHE_PATH'))
print('✅ vLLM engine loaded successfully')

print('✅ All packages working!')
"

echo "Installation check complete"