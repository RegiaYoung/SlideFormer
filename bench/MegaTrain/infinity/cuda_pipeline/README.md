# CUDA Pipeline Extension

This directory contains optimized C++/CUDA extensions for the CPU-GPU training pipeline.

## Features

1. **Batched Parameter Copy**: Efficient batch copying of multiple tensors
2. **Async Gradient Accumulation**: Optimized GPU-to-CPU gradient transfer
3. **Stream-based Pipelining**: Overlapping compute and memory transfers

## Two Build Modes

### Simple Mode (Recommended)
Uses C++ with PyTorch operations - **more compatible**, works with any PyTorch version:
```bash
cd cuda_pipeline
bash build.sh simple
```

### CUDA Mode (Maximum Performance)
Uses custom CUDA kernels - **maximum performance**, requires nvcc and compatible GPU:
```bash
cd cuda_pipeline
bash build.sh cuda
```

## Quick Start

```bash
# Build (simple mode by default)
cd /work/nvme/bemy/zyuan2/code/Infinity/examples/cuda_pipeline
bash build.sh

# Test
python test_extension.py

# Run training with extension
cd ..
python train_cpu_master_v6.py
```

## Troubleshooting

### API Compatibility Issues
If you see errors like `namespace "at::cuda" has no member "getCurrentCUDAStream"`:
- Use **simple mode**: `bash build.sh simple`
- This uses only PyTorch C++ API without direct CUDA calls

### Missing nvcc
If nvcc is not found:
- Use **simple mode** (doesn't require nvcc)
- Or install CUDA toolkit for your system

### Compute Capability Mismatch
If you see compute capability errors in CUDA mode:
- Edit `setup.py` and adjust the `-gencode` flags for your GPU
- Or use **simple mode**

## Performance Benefits

- **Reduced Python overhead**: C++ bypasses Python GIL
- **Batched operations**: Single call for multiple tensors
- **Better stream utilization**: Explicit stream management
- **Pinned memory**: Faster CPU-GPU transfers

## Requirements

- PyTorch with CUDA support
- C++17 compatible compiler
- (Optional) CUDA Toolkit 11.0+ for CUDA mode
- (Optional) GPU with compute capability 8.0+ for CUDA mode
