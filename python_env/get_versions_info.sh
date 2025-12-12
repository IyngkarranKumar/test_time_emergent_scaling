#!/bin/bash

echo "=== CUDA & vLLM Compatibility Check ==="

# Python & PyTorch
python -c "
import sys
print(f'Python: {sys.version.split()[0]}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'PyTorch CUDA: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        if torch.cuda.device_count() > 0:
            print(f'GPU 0: {torch.cuda.get_device_name(0)}')
except Exception as e: 
    print(f'PyTorch import error: {e}')
"

echo ""

# vLLM comprehensive check
python -c "
print('=== vLLM Check ===')
try:
    import vllm
    print(f'✅ vLLM version: {vllm.__version__}')
except Exception as e:
    print(f'❌ vLLM import error: {e}')
    exit()

# Check vLLM C extensions
try:
    import vllm._C
    print('✅ vLLM C extensions imported successfully')
    
    # Try to get CUDA info from C extensions
    c_lib_path = vllm._C.__file__
    print(f'   C library path: {c_lib_path}')
    
    # Check if CUDA version info is available
    if hasattr(vllm._C, 'cuda_version'):
        print(f'   vLLM compiled CUDA: {vllm._C.cuda_version}')
    
except Exception as e:
    print(f'❌ vLLM C extensions failed: {e}')

# Check vLLM platform detection
try:
    from vllm.platforms import current_platform
    print(f'✅ Platform detected: {current_platform.__class__.__name__}')
    
    if hasattr(current_platform, 'get_device_capability') and hasattr(current_platform, 'get_device_name'):
        try:
            cap = current_platform.get_device_capability()
            name = current_platform.get_device_name()
            print(f'   Device: {name} (capability: {cap})')
        except:
            pass
            
except Exception as e:
    print(f'❌ Platform detection failed: {e}')

# Test basic vLLM functionality
try:
    from vllm import LLM, SamplingParams
    print('✅ vLLM core classes imported successfully')
except Exception as e:
    print(f'❌ vLLM core import failed: {e}')

# Check for CUDA runtime dependencies
try:
    import subprocess
    import vllm._C
    lib_path = vllm._C.__file__
    result = subprocess.run(['ldd', lib_path], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        cuda_deps = [line for line in result.stdout.split('\n') if 'cuda' in line.lower()]
        if cuda_deps:
            print('✅ CUDA library dependencies:')
            for dep in cuda_deps[:5]:  # Show first 5
                print(f'   {dep.strip()}')
        else:
            print('ℹ️  No CUDA dependencies found in ldd output')
    else:
        print('ℹ️  Could not check library dependencies (ldd failed)')
except Exception as e:
    print(f'ℹ️  Could not check dependencies: {e}')
"

echo ""

# System CUDA
echo "=== System CUDA ==="
echo -n "System CUDA: "
if nvidia-smi &>/dev/null; then
    nvidia-smi | grep "CUDA Version" | awk '{print $9}'
else
    echo "Not found"
fi

echo -n "NVCC: "
if nvcc --version &>/dev/null; then
    nvcc --version | grep release | awk '{print $6}' | cut -c2-
else
    echo "Not found"
fi

echo "CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-Not set}"

# Check for CUDA runtime libraries
echo ""
echo "=== CUDA Runtime Libraries ==="
for version in 11.8 12.0 12.1 12.2; do
    major=$(echo $version | cut -d. -f1)
    if find /usr/local/cuda* /opt/cuda* $CONDA_PREFIX 2>/dev/null | xargs -I {} find {} -name "libcudart.so.$major*" 2>/dev/null | head -1 | grep -q .; then
        echo "✅ CUDA $version runtime found"
    else
        echo "❌ CUDA $version runtime not found"
    fi
done

# vLLM installation info
echo ""
echo "=== vLLM Installation Info ==="
python -c "
try:
    import vllm
    import pkg_resources
    dist = pkg_resources.get_distribution('vllm')
    print(f'Installation location: {dist.location}')
    print(f'Installation method: {dist.project_name} {dist.version}')
except Exception as e:
    print(f'Could not get installation info: {e}')
"

# Check for wheel info
if command -v uv &> /dev/null; then
    echo "uv pip show vllm:"
    uv pip show vllm 2>/dev/null | grep -E "(Version|Location|Name)" || echo "No uv info available"
fi

echo ""
echo "=== Summary ==="
python -c "
import sys
try:
    import torch, vllm, vllm._C
    print('✅ All components working: Python + PyTorch + vLLM + C extensions')
except ImportError as e:
    print(f'❌ Missing components: {e}')
except Exception as e:
    print(f'❌ Runtime error: {e}')
"