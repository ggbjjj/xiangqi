"""Run with the remote interpreter before training: python tools/check_training_env.py."""
import platform
import sys
import torch

print('Python:',sys.version.split()[0],platform.platform())
print('Interpreter:',sys.executable)
print('PyTorch:',torch.__version__,'wheel CUDA:',torch.version.cuda)
print('CUDA available:',torch.cuda.is_available())
if torch.cuda.is_available():
    print('Compiled GPU architectures:',torch.cuda.get_arch_list())
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {p.name}, CC {p.major}.{p.minor}, VRAM {p.total_memory/2**30:.1f} GiB')
    x=torch.randn(256,256,device='cuda',requires_grad=True)
    (x@x).square().mean().backward(); torch.cuda.synchronize()
    q=torch.randn(1,4,90,32,device='cuda',dtype=torch.float16,requires_grad=True)
    torch.nn.functional.scaled_dot_product_attention(q,q,q).float().sum().backward()
    torch.cuda.synchronize()
    print('CUDA matmul + SDPA forward/backward: OK; BF16:',torch.cuda.is_bf16_supported())
else:
    print('CPU inference/tests are available. For GPU: uv sync --locked --extra gpu; NVIDIA driver required.')
