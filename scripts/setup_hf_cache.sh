#!/bin/bash
# More efficient download script

if [ "$node_type" == "vastai" ]; then
    source env_vars/.env.vastai
else
    source env_vars/.env.mlp_head
fi

source env_vars/.env.mlp_head

echo "Downloading to $HF_CACHE_PATH"

#No commas!
MODELS=(
   "microsoft/Phi-4-reasoning-plus"

)

#No commas!
DATASETS=(
   ""
)

echo "========================================"
echo "Starting download..."
echo "========================================"

for MODEL_NAME in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Downloading model files: $MODEL_NAME"
    
    python -c "
from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings('ignore')
try:
    print(f'Downloading $MODEL_NAME files to cache...')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', cache_dir='$HF_CACHE_PATH')
    model = AutoModelForCausalLM.from_pretrained('$MODEL_NAME', cache_dir='$HF_CACHE_PATH')
    print(f'✅ Model $MODEL_NAME files downloaded successfully')
except Exception as e:
    print(f'❌ Error downloading $MODEL_NAME: {e}')
"
done

echo "========================================"
echo "Starting dataset downloads..."
echo "========================================"

for DATASET_NAME in "${DATASETS[@]}"; do
    if [ -n "$DATASET_NAME" ]; then  # Skip empty dataset names
        echo "----------------------------------------"
        echo "Downloading dataset: $DATASET_NAME"
        
        python -c "
from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings('ignore')

try:
    print(f'Downloading $DATASET_NAME files to cache...')
    from datasets import load_dataset
    load_dataset(
        '$DATASET_NAME',
        cache_dir='$HF_CACHE_PATH',
        download_mode='force_redownload'
    )
    print(f'✅ Dataset $DATASET_NAME downloaded successfully')
except Exception as e:
    print(f'❌ Error downloading $DATASET_NAME: {e}')
"
    fi
done

echo "========================================"
echo "All downloads completed!"
echo "========================================"
