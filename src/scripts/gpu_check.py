"""
GPU smoke test for fine-tuning — run BEFORE training to confirm the GPU
is actually used, in ~30 seconds, without loading the full model.

Usage:
    python scripts/gpu_check.py

Expected output:
    CUDA available: True
    GPU name: NVIDIA GeForce RTX 3050 Laptop GPU
    GPU compute test: PASS (took X ms on cuda:0)
    fp16 matmul: PASS
"""
import time
import torch

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: PyTorch can't see your GPU. Reinstall with the CUDA build:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    raise SystemExit(1)

print("GPU name:", torch.cuda.get_device_name(0))
print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")

# 1) basic GPU compute test — a matmul on cuda
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
dt = (time.time() - t0) * 1000
print(f"GPU matmul 2048x2048: {dt:.1f} ms (should be < 100ms)")

# 2) fp16 matmul (the dtype we train in)
a16 = a.half()
b16 = b.half()
t0 = time.time()
c16 = a16 @ b16
torch.cuda.synchronize()
dt16 = (time.time() - t0) * 1000
print(f"GPU fp16 matmul: {dt16:.1f} ms (should be < 100ms)")

# 3) the critical check — is a tensor actually computing on GPU?
x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
y = (x @ x).sum()
y.backward()
torch.cuda.synchronize()
assert x.grad is not None and x.grad.device.type == "cuda"
print("Backward pass on GPU: PASS")

print("\n=== GPU IS WORKING — training will use it ===")
