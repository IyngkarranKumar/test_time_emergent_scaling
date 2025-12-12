echo "🧪 Testing installation..."
python3 -c "
print('Testing imports')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

try:
    from vllm._C import cache_ops
    print('✅ VLLM CUDA ops working!')
except ImportError as e:
    print(f'❌ VLLM CUDA ops failed: {e}')
    print('You may need to reinstall VLLM')

print(f'✅ PyTorch version: {torch.__version__}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'PyTorch compiled with CUDA: {torch.version.cuda is not None}')

print('Testing vllm model loading')
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', tensor_parallel_size=1, enforce_eager=True)
print('✅ vLLM engine loaded successfully')

print('✅ All packages working!')
print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Installation check complete"
