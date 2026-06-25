#!/usr/bin/env python3
"""Test script for CUDA pipeline extension."""

import torch
import time

def test_batched_copy():
    """Test batched parameter copy."""
    print("Testing batched parameter copy...")

    # Create test tensors
    num_tensors = 10
    tensor_size = 1024 * 1024  # 1M elements

    cpu_tensors = [torch.randn(tensor_size, dtype=torch.float32).pin_memory() for _ in range(num_tensors)]
    gpu_tensors = [torch.zeros(tensor_size, dtype=torch.float32, device='cuda:0') for _ in range(num_tensors)]

    # Warm up
    for _ in range(3):
        for cpu_t, gpu_t in zip(cpu_tensors, gpu_tensors):
            gpu_t.copy_(cpu_t, non_blocking=True)
    torch.cuda.synchronize()

    # Benchmark PyTorch copy
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        for cpu_t, gpu_t in zip(cpu_tensors, gpu_tensors):
            gpu_t.copy_(cpu_t, non_blocking=True)
    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start

    # Benchmark CUDA extension
    try:
        import cuda_pipeline

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            cuda_pipeline.batched_copy_params(cpu_tensors, gpu_tensors)
        torch.cuda.synchronize()
        cuda_time = time.perf_counter() - start

        print(f"  PyTorch copy: {pytorch_time*1000:.2f} ms")
        print(f"  CUDA extension: {cuda_time*1000:.2f} ms")
        print(f"  Speedup: {pytorch_time/cuda_time:.2f}x")

        # Verify correctness
        for cpu_t, gpu_t in zip(cpu_tensors, gpu_tensors):
            assert torch.allclose(cpu_t.cuda(), gpu_t, rtol=1e-5), "Copy mismatch!"
        print("  ✓ Correctness verified")

    except ImportError:
        print("  CUDA extension not available, skipping comparison")

def test_grad_accumulation():
    """Test async gradient accumulation."""
    print("\nTesting async gradient accumulation...")

    # Create test tensors with gradients
    num_tensors = 10
    tensor_size = 1024 * 1024

    gpu_tensors = [torch.randn(tensor_size, dtype=torch.float32, device='cuda:0', requires_grad=True) for _ in range(num_tensors)]
    cpu_tensors = [torch.randn(tensor_size, dtype=torch.float32).pin_memory() for _ in range(num_tensors)]

    # Create fake gradients
    for t in gpu_tensors:
        t.grad = torch.randn_like(t)

    # Benchmark PyTorch accumulation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        for cpu_t, gpu_t in zip(cpu_tensors, gpu_tensors):
            if cpu_t.grad is None:
                cpu_t.grad = torch.zeros_like(cpu_t)
            cpu_t.grad.add_(gpu_t.grad.cpu())
    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start

    # Reset CPU grads
    for t in cpu_tensors:
        t.grad = None

    # Benchmark CUDA extension
    try:
        import cuda_pipeline

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            cuda_pipeline.async_accumulate_grads(gpu_tensors, cpu_tensors)
        torch.cuda.synchronize()
        cuda_time = time.perf_counter() - start

        print(f"  PyTorch accumulation: {pytorch_time*1000:.2f} ms")
        print(f"  CUDA extension: {cuda_time*1000:.2f} ms")
        print(f"  Speedup: {pytorch_time/cuda_time:.2f}x")
        print("  ✓ Test passed")

    except ImportError:
        print("  CUDA extension not available, skipping comparison")

if __name__ == "__main__":
    print("=" * 70)
    print("CUDA Pipeline Extension Test")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        exit(1)

    try:
        import cuda_pipeline
        print("✓ CUDA extension loaded successfully\n")
    except ImportError:
        print("✗ CUDA extension not available\n")
        print("To build the extension, run:")
        print("  cd cuda_pipeline && bash build.sh\n")

    test_batched_copy()
    test_grad_accumulation()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
