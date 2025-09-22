#!/usr/bin/env python3

import os
import torch
import sys, importlib 
from pathlib import Path
import pdb
from dataclasses import dataclass
from typing import Dict, List, Optional
import utils 
import gc
if torch.cuda.is_available():
    import pynvml

importlib.reload(utils)


@dataclass
class ModelResult:
    precision: str
    hf_VRAM_used: Optional[int] = None
    vllm_VRAM_used: Optional[int] = None
    hf_success: bool = False
    vllm_success: bool = False
    hf_generation: Optional[str] = None
    vllm_generation: Optional[str] = None

TEST_PROMPT = "The quick brown fox jumps over the lazy dog, continue this sentence:"


def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        if not torch.cuda.is_available():
            return "CUDA not available"
            
        pynvml.nvmlInit()
        gpu_count = torch.cuda.device_count()
        
        vram_info = []
        for gpu_id in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_info.append(f"GPU {gpu_id}: {info.used//1024//1024}MiB / {info.total//1024//1024}MiB")
        
        return f"Found {gpu_count} GPUs\n" + "\n".join(vram_info)
        
    except Exception as e:
        print(f"Error getting GPU info: {e}")

def get_gpu_memory_usage():
    """Get current GPU memory usage across all GPUs in MiB"""
    try:
        if not torch.cuda.is_available():
            return 0
        pynvml.nvmlInit()
        gpu_count = torch.cuda.device_count()
        total_usage = 0
        for gpu_id in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_usage += info.used // 1024 // 1024  # Convert to MiB
        return total_usage
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return 0

def find_model_path(model_input):
    """Find the actual model path from various formats"""
    hf_cache = os.environ.get('HF_CACHE_PATH', '')
    print(f"HF_CACHE_PATH: {hf_cache}")
    
    # Direct path
    if Path(model_input).is_dir() and (Path(model_input) / "config.json").exists():
        return model_input, True
    
    # Path under cache
    if hf_cache:
        cache_path = Path(hf_cache) / model_input
        if cache_path.is_dir() and (cache_path / "config.json").exists():
            return str(cache_path), True
        
        # HF hub format: org/model -> models--org--model/snapshots/hash
        hf_format = f"models--{model_input.replace('/', '--')}"
        hf_path = Path(hf_cache) / hf_format / "snapshots"
        if hf_path.is_dir():
            snapshots = list(hf_path.iterdir())
            if snapshots:
                return str(snapshots[0]), True
    
    # Not found locally, assume it's a HF model name
    return model_input, False

def test_hf(model_path, is_local, precision):
    """Test HuggingFace loading"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Get initial memory usage
        initial_mem = get_gpu_memory_usage()
        
        kwargs = {'local_files_only': True} if is_local else {}
        
        if precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        elif precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
        elif precision == "fp8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            kwargs["quantization_config"] = quantization_config
            kwargs["device_map"] = "auto"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        
        
        # Force garbage collection before loading model
        gc.collect()
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        
        # Force synchronization before measuring memory
        torch.cuda.synchronize()
        
        # Get memory usage after model load
        loaded_mem = get_gpu_memory_usage()
        vram_used = loaded_mem - initial_mem
        
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt")
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            first_device = next(iter(model.hf_device_map.values()))
            inputs = inputs.to(first_device)
        else:
            inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)


        #aggressive clean up 

        #remove all cuda tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            except:
                pass


        #clean up VRAM
        del model, inputs, outputs
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        return True, result, vram_used
    except Exception as e:
        print(f"✗ HF {model_path} ({precision}): {e}")
        return False, str(e), 0 #returns the error message as the generation

def test_vllm(model_path, is_local, precision):
    """Test vLLM loading"""
    try:
        from vllm import LLM, SamplingParams

        #clear cache before starting
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Get initial memory usage
        initial_mem = get_gpu_memory_usage()
        
        dtype = "float16" if precision in ["fp16", "fp8"] else "bfloat16"
        llm = LLM(model=model_path, dtype=dtype)
        
        # Get memory usage after model load
        loaded_mem = get_gpu_memory_usage()
        vram_used = loaded_mem - initial_mem
        
        outputs = llm.generate(TEST_PROMPT, SamplingParams(max_tokens=20))
        result = outputs[0].outputs[0].text

        #clean up
        del llm, outputs
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        return True, TEST_PROMPT + result, vram_used
    except ImportError:
        print(f"✗ vLLM {model_path} ({precision}): not installed")
        return False, None, 0
    except Exception as e:
        print(f"✗ vLLM {model_path} ({precision}): {e}")
        return False, str(e), 0 #returns error message as the generation

def test_models(models, precisions):
    """Test a list of models"""
    results: Dict[str, List[ModelResult]] = {}
    
    for model in models:
        print(f"\nTesting: {model}")
        model_path, is_local = find_model_path(model)
        print(f"Path: {model_path} ({'local' if is_local else 'remote'})")
        
        results[model] = []
        for precision in precisions:
            print(f"\nTesting with {precision}...")
            
            hf_ok, hf_gen, hf_vram = test_hf(model_path, is_local, precision)
            vllm_ok, vllm_gen, vllm_vram = test_vllm(model_path, is_local, precision)
            
            results[model].append(ModelResult(
                precision=precision,
                hf_VRAM_used=hf_vram,
                vllm_VRAM_used=vllm_vram,
                hf_success=hf_ok,
                vllm_success=vllm_ok,
                hf_generation=hf_gen,
                vllm_generation=vllm_gen
            ))
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)


    
    for model, model_results in results.items():
        print(f"\nModel: {model}")
        for result in model_results:
            print(f"\nPrecision: {result.precision}")
            print(f"HF Success: {'✓' if result.hf_success else '✗'}")
            if result.hf_success:
                print(f"HF VRAM Used: {result.hf_VRAM_used}MiB")
            if result.hf_generation:
                print(f"HF Generation: {result.hf_generation}")
            print(f"vLLM Success: {'✓' if result.vllm_success else '✗'}")
            if result.vllm_success:
                print(f"vLLM VRAM Used: {result.vllm_VRAM_used}MiB")
            if result.vllm_generation:
                print(f"vLLM Generation: {result.vllm_generation}")

def main():
   
    gpu_info = get_gpu_info()
    print(gpu_info)

    # Example usage - modify this list with your models
    models_to_test = [

        "Qwen/Qwen2.5-14B-Instruct",
    ]

    precisions = ["fp16", "fp8"]
    
    # Use command line args if provided, otherwise use the list above
    models = sys.argv[1:] if len(sys.argv) > 1 else models_to_test
    
    print(f"Testing {len(models)} models...")
    test_models(models, precisions)

if __name__ == "__main__":
    main()