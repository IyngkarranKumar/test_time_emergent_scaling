import torch
import numpy as np
import warnings
import gc
import inspect
import os
import time
import subprocess
from contextlib import contextmanager
import psutil

if torch.cuda.is_available():
    import pynvml



def get_model_info(model):
    num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else getattr(model.config, 'n_layer', None)
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else getattr(model.config, 'n_embd', None)
    vocab_size = model.config.vocab_size
    num_attention_heads = (
        model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads')
        else getattr(model.config, 'n_head', None)
    )
    return num_layers, hidden_size, vocab_size, num_attention_heads



#memory management util
def gpu_utilisation(device_idx=0,all=True):

    if torch.cuda.is_available():
        if not all:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            utilisation=mem_info.used*100/mem_info.total
            return utilisation

        else:
            pynvml.nvmlInit()
            utilisations=[]
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilisation=mem_info.used*100/mem_info.total
                utilisations.append(utilisation)
            pynvml.nvmlShutdown()
            return utilisations
    
    else:
        return "NaN"

def cpu_utilisation():
    import psutil
    memory = psutil.virtual_memory()
    utilisation = memory.used*100 / memory.total
    return utilisation

def disk_space_utilisation():
    """Return disk space usage as percentage for current working directory"""
    try:
        usage = psutil.disk_usage('.')
        return usage.percent
    except (PermissionError, FileNotFoundError):
        return 0.0

def memory_utilisation():
    if torch.cuda.is_available():
        return cpu_utilisation(),gpu_utilisation(all=True), disk_space_utilisation()
    else:
        return cpu_utilisation(), "NaN", disk_space_utilisation()

def tensor_size(tensor):
    """Get the size of a tensor in bytes"""
    return (tensor.numel() * tensor.element_size())/1024**2

def top_gpu_tensors(k=20,device_idx=0):
    """Find top-k GPU tensors by memory usage with variable names"""
    gpu_tensors = []
    
    # Force garbage collection to get accurate results
    gc.collect()
    torch.cuda.empty_cache()
    
    # Get caller's variables
    frame = inspect.currentframe().f_back
    all_vars = {**frame.f_locals, **frame.f_globals}
    
    # Iterate through all objects in memory
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.device.index==device_idx:
                size_mb = (obj.numel() * obj.element_size()) / (1024 * 1024)
                
                # Find variable name
                name = "unknown"
                for var_name, var_obj in all_vars.items():
                    try:
                        if var_obj is obj:
                            name = var_name
                            break
                        # Check if it's a model parameter
                        elif hasattr(var_obj, 'named_parameters'):
                            for param_name, param in var_obj.named_parameters():
                                if param is obj:
                                    name = f"{var_name}.{param_name}"
                                    break
                    except (ReferenceError, RuntimeError):
                        continue
                
                gpu_tensors.append({
                    'tensor': obj,
                    'size_mb': size_mb,
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                    'device': str(obj.device),
                    'name': name
                })
        except Exception:
            continue
    
    # Sort by size (largest first) and get top k
    gpu_tensors.sort(key=lambda x: x['size_mb'], reverse=True)
    top_k = gpu_tensors[:k]
    
    # Print fancy table
    print("\n🔍 Top GPU Memory Offenders:")
    print("=" * 90)
    print(f"{'Rank':<6} {'Size (MB)':<12} {'Shape':<20} {'DType':<12} {'Device':<10} {'Variable Name':<20}")
    print("-" * 90)
    
    total_size = 0
    for i, tensor_info in enumerate(top_k, 1):
        print(f"{i:<6} {tensor_info['size_mb']:<12.2f} {str(tensor_info['shape']):<20} "
              f"{tensor_info['dtype']:<12} {tensor_info['device']:<10} {tensor_info['name']:<20}")
        total_size += tensor_info['size_mb']
    
    if len(gpu_tensors) > k:
        remaining_size = sum(t['size_mb'] for t in gpu_tensors[k:])
        print("-" * 90)
        print(f"... and {len(gpu_tensors) - k} more tensors ({remaining_size:.2f} MB)")
        total_size += remaining_size
    
    print("=" * 90)
    print(f"📊 Total tensor memory: {total_size:.2f} MB ({total_size/1024:.2f} GB)")

def top_disk_usage(k=10, path=None):
    """Find top-k directories using most disk space with fancy formatting"""
    if path is None:
        path = "/" if os.name != 'nt' else "C:\\"
    
    directories = []
    
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                try:
                    size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk(item_path)
                             for filename in filenames)
                    directories.append({
                        'name': item,
                        'path': item_path,
                        'size_mb': size / (1024 * 1024),
                        'size_gb': size / (1024 * 1024 * 1024)
                    })
                except (PermissionError, OSError):
                    continue
    except PermissionError:
        print(f"❌ Permission denied accessing {path}")
        return
    
    # Sort by size (descending)
    directories.sort(key=lambda x: x['size_gb'], reverse=True)
    top_k = directories[:k]
    
    # Print fancy table
    print("\n💾 Top Disk Usage (Directories):")
    print("=" * 80)
    print(f"{'Rank':<6} {'Size (GB)':<12} {'Size (MB)':<12} {'Directory Name':<30}")
    print("-" * 80)
    
    total_size = 0
    for i, dir_info in enumerate(top_k, 1):
        print(f"{i:<6} {dir_info['size_gb']:<12.2f} {dir_info['size_mb']:<12.1f} "
              f"{dir_info['name']:<30}")
        total_size += dir_info['size_gb']
    
    if len(directories) > k:
        remaining_size = sum(d['size_gb'] for d in directories[k:])
        print("-" * 80)
        print(f"... and {len(directories) - k} more directories ({remaining_size:.2f} GB)")
        total_size += remaining_size
    
    print("=" * 80)
    print(f"📊 Total disk usage: {total_size:.2f} GB")

def process_vram_usage():
    pass

@contextmanager #tells pythhon that this can be used in "with" blocks
def timer(description="Operation", logger=None):
    start_time = time.perf_counter()
    yield #runs with block 
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    if logger is not None:
        logger.info(f"{description}: {elapsed_time:.4f} seconds")
    else:
        print(f"{description}: {elapsed_time:.4f} seconds")



def nvidia_smi_output():
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    return result.stdout




@contextmanager
def gpu_util_manager(description="Operation", logger=None, report_absolutes=False):

    if torch.cuda.is_available():
        # Get initial GPU utilizations and VRAM
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        start_utils = []
        total_vram_gb = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = mem_info.used * 100 / mem_info.total
            start_utils.append(util)
            total_vram_gb.append(mem_info.total / 1024**3)
        # Get initial CPU RAM usage
        start_ram = psutil.Process().memory_info().rss / 1024**3  # Convert to GB
        pynvml.nvmlShutdown()

        yield

        # Get final GPU utilizations
        pynvml.nvmlInit()
        end_utils = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = mem_info.used * 100 / mem_info.total
            end_utils.append(util)
        # Get final CPU RAM usage
        end_ram = psutil.Process().memory_info().rss / 1024**3  # Convert to GB
        ram_change = end_ram - start_ram
        pynvml.nvmlShutdown()

        changes = [end - start for end, start in zip(end_utils, start_utils)]

        if report_absolutes:
            # Calculate absolute VRAM change in GB
            abs_changes_gb = [((end - start) / 100.0) * vram for start, end, vram in zip(start_utils, end_utils, total_vram_gb)]
            if logger is not None:
                logger.info(f"{description} - GPU VRAM change (GB): {[f'{c:+.2f}GB' for c in abs_changes_gb]}")
                logger.info(f"{description} - CPU RAM change: {ram_change:+.1f}GB")
            else:
                print(f"{description} - GPU VRAM change (GB): {[f'{c:+.2f}GB' for c in abs_changes_gb]}")
                print(f"{description} - CPU RAM change: {ram_change:+.1f}GB")
        else:
            if logger is not None:
                logger.info(f"{description} - GPU utilisation before: {[f'{u:.1f}%' for u in start_utils]}")
                logger.info(f"{description} - GPU utilisation after: {[f'{u:.1f}%' for u in end_utils]}")
                logger.info(f"{description} - Changes: {[f'{c:+.1f}%' for c in changes]}")
                logger.info(f"{description} - CPU RAM change: {ram_change:+.1f}GB")
            else:
                print(f"{description} - GPU utilisation before: {[f'{u:.1f}%' for u in start_utils]}")
                print(f"{description} - GPU utilisation after: {[f'{u:.1f}%' for u in end_utils]}")
                print(f"{description} - Changes: {[f'{c:+.1f}%' for c in changes]}")
                print(f"{description} - CPU RAM change: {ram_change:+.1f}GB")

    else:
        start_ram = psutil.Process().memory_info().rss / 1024**3  # Convert to GB
        yield
        end_ram = psutil.Process().memory_info().rss / 1024**3  # Convert to GB
        ram_change = end_ram - start_ram
        if logger is not None:
            logger.info(f"{description} - CPU RAM change: {ram_change:+.1f}GB")
        else:
            print(f"{description} - CPU RAM change: {ram_change:+.1f}GB")



def bytes_to_gb(num_bytes):
    """Convert bytes to GB."""
    return num_bytes / (1024**3)

def regular_forward_pass_memory(vocab_size, hidden_size, num_layers,num_attention_heads, batch_size, seq_len, precision_bytes=2):
    """
    Memory for regular forward pass: model(input_ids)
    Returns logits for ALL tokens: [batch_size, seq_len, vocab_size]

    This is regular forward pass - where we unembed all token final hidden states
    """
    # Input tokens
    input_mem = batch_size * seq_len * 4  # int32 token IDs
    
    # Hidden states through layers
    hidden_mem = batch_size * seq_len * hidden_size * num_layers * precision_bytes
    
    # Attention matrices (quadratic in seq_len)
    attention_mem = batch_size * num_attention_heads * seq_len * seq_len * precision_bytes  # assume 32 heads
    
    # Full output logits [batch_size, seq_len, vocab_size]
    output_mem = batch_size * seq_len * vocab_size * precision_bytes
    
    total_bytes = input_mem + hidden_mem + attention_mem + output_mem
    return bytes_to_gb(total_bytes)

def hidden_states_only_memory(vocab_size, hidden_size, num_layers,num_attention_heads, batch_size, seq_len, precision_bytes=2):
    """
    Memory for hidden states approach: get last_hidden_state, then lm_head on final token only
    Returns logits for FINAL token only: [batch_size, 1, vocab_size]

    Note: This is an efficient forward pass - where we only get last hidden states, then unembed the final token.
    """
    # Input tokens
    input_mem = batch_size * seq_len * 4  # int32 token IDs
    
    # Hidden states through layers
    hidden_mem = batch_size * seq_len * hidden_size * num_layers * precision_bytes
    
    # Attention matrices (same as regular)
    attention_mem = batch_size * num_attention_heads * seq_len * seq_len * precision_bytes  # assume 32 heads
    
    # Output logits for FINAL token only [batch_size, 1, vocab_size]
    output_mem = batch_size * 1 * vocab_size * precision_bytes
    
    total_bytes = input_mem + hidden_mem + attention_mem + output_mem
    return bytes_to_gb(total_bytes)


def max_batch_size_for_memory(target_memory_gb, vocab_size, hidden_size, num_layers, num_attention_heads, seq_len, precision_bytes=2):
    """
    Find maximum batch size that fits in target_memory_gb.
    
    The memory formula is:
    total_bytes = batch_size * (seq_len * 4 + seq_len * hidden_size * num_layers * precision_bytes + 
                               num_attention_heads * seq_len * seq_len * precision_bytes + vocab_size * precision_bytes)
    
    So: batch_size = total_bytes / per_batch_bytes
    """
    
    # Memory per batch
    per_batch_bytes = (
        seq_len * 4 +  # input tokens
        seq_len * hidden_size * num_layers * precision_bytes +  # hidden states
        num_attention_heads * seq_len * seq_len * precision_bytes +  # attention
        vocab_size * precision_bytes  # output logits (final token only)
    )
    per_batch_gb = bytes_to_gb(per_batch_bytes).item()

    max_batch_size = int(target_memory_gb // per_batch_gb)
    
    return min(max(max_batch_size, 1), 100) #upper bounded at 100 to try prevent CUDA OOM. 

def is_oom_error(error):
    error_msg = str(error).lower()
    oom_patterns = [
        "out of memory", 
        "invalid configuration argument",
        "invalid configuration", 
        "cuda error",
        "cublas_status_alloc_failed",
        "insufficient memory",
        "cuda out of memory",
        "runtime error",  # Sometimes CUDA errors are wrapped
    ]
    return any(pattern in error_msg for pattern in oom_patterns)

def auto_force_end_batch_scale_oom_guard(initial_size,operation_func,logger=None,description="Operation",scale_factor=0.8,min_size=1,**kwargs):
    
    size = initial_size 

    while size >=min_size:
        try:
            return operation_func(**kwargs,batch_size=size)
        except Exception as e:
            if isinstance(e,torch.cuda.OutOfMemoryError) or is_oom_error(e):
                torch.cuda.empty_cache()
                size = max(int(size * scale_factor), min_size)
                if logger is not None:
                    logger.info(f"{description} - Batch size scaled down to {size} due to CUDA OOM")
                else:
                    print(f"{description} - Batch size scaled down to {size} due to CUDA OOM")
            else:
                raise e


def auto_scoring_batch_scale_oom_guard(initial_size,operation_func,logger=None,description="Operation",scale_factor=0.8,min_size=1,**kwargs):

    size = initial_size 

    while size >= min_size:
        try:
            return operation_func(**kwargs,batch_size=size)
        except Exception as e:
            if isinstance(e,torch.cuda.OutOfMemoryError) or is_oom_error(e):
                torch.cuda.empty_cache()
                size = max(int(size * scale_factor), min_size)
                if logger is not None:
                    logger.info(f"{description} - Batch size scaled down to {size} due to CUDA OOM")
                else:
                    print(f"{description} - Batch size scaled down to {size} due to CUDA OOM")
            else:
                raise e


def create_memory_filler_tensor(gpu_id,target_gb):

    """Fill GPU memory to target GB"""
    current_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
    needed_gb = target_gb - current_gb
    
    if needed_gb <= 0:
        return None
    
    # 4 bytes per float32 element
    elements = int(needed_gb * (1024**3) / 4)
    
    try:
        return torch.randn(elements, device=f'cuda:{gpu_id}')
    except:
        return torch.randn(int(elements * 0.9), device=f'cuda:{gpu_id}')





