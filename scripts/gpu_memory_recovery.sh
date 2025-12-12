#this is all claude written
# VAST.AI GPU MEMORY RECOVERY (NO REBOOT) 

echo "🔧 Vast.ai GPU Recovery - No Reboot Method"

# Step 1: Kill everything that might be holding GPU memory
echo "=== Killing all potential GPU processes ==="
pkill -9 python
pkill -9 jupyter
pkill -9 ipython
pkill -9 tensorboard
pkill -9 wandb

# Step 2: Check for any remaining processes using GPU
echo "=== Checking for remaining GPU processes ==="
fuser -k /dev/nvidia0 2>/dev/null || echo "No processes on GPU 0"
fuser -k /dev/nvidia1 2>/dev/null || echo "No processes on GPU 1"

# Step 3: Force GPU context cleanup
echo "=== Force clearing GPU contexts ==="
python3 -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

try:
    import torch
    import gc
    
    print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'Cleaning GPU {i}...')
            with torch.cuda.device(i):
                # Force clear all cached memory
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                
                # Try to allocate and immediately free memory to force cleanup
                try:
                    dummy = torch.zeros(1024, 1024, device=f'cuda:{i}')
                    del dummy
                    torch.cuda.empty_cache()
                    print(f'  GPU {i}: Force allocation/cleanup successful')
                except:
                    print(f'  GPU {i}: Could not allocate (memory may be stuck)')
    
    # Force garbage collection
    gc.collect()
    
except Exception as e:
    print(f'Error during cleanup: {e}')
"

# Step 4: NVIDIA persistence daemon reset (this sometimes works)
echo "=== Attempting nvidia-ml reset ==="
python3 -c "
try:
    import pynvml
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    print(f'Found {count} GPUs via NVML')
    
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # Try to reset GPU state
        try:
            pynvml.nvmlDeviceResetApplicationsClocks(handle)
            print(f'Reset application clocks for GPU {i}')
        except:
            pass
            
    pynvml.nvmlShutdown()
    print('NVML cleanup complete')
except Exception as e:
    print(f'NVML not available: {e}')
"

# Step 5: Memory mapping cleanup
echo "=== Clearing shared memory ==="
rm -f /dev/shm/* 2>/dev/null || echo "Shared memory cleared"

# Step 6: Check current status
echo "=== Current GPU Status ==="
nvidia-smi

# Step 7: Final attempt - try to initialize and clear CUDA contexts
echo "=== Final context reset attempt ==="
python3 -c "
import ctypes
import os

# Try to load CUDA runtime and force reset
try:
    cuda_rt = ctypes.CDLL('libcudart.so')
    
    # Reset all devices
    for i in range(2):  # You have 2 GPUs
        cuda_rt.cudaSetDevice(i)
        cuda_rt.cudaDeviceReset()
        print(f'Attempted cudaDeviceReset on GPU {i}')
        
except Exception as e:
    print(f'CUDA runtime reset failed: {e}')
"

echo "=== Final Status Check ==="
nvidia-smi

echo ""
echo "If GPU memory is still allocated, try:"
echo "1. Run this script again"
echo "2. Wait 5-10 minutes (sometimes GPU memory gets released automatically)"
echo "3. As last resort, you might need to destroy and recreate the vast.ai instance"