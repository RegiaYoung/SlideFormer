from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import sys

# Determine which version to build
use_simple = '--simple' in sys.argv
if use_simple:
    sys.argv.remove('--simple')

if use_simple:
    # Simple C++ version (no custom CUDA kernels)
    print("Building simple C++ version (no custom CUDA kernels)")
    ext_modules = [
        CppExtension(
            name='cuda_pipeline',
            sources=['simple_pipeline.cpp'],
            extra_compile_args={'cxx': ['-O3', '-std=c++17']}
        )
    ]
else:
    # Full CUDA version with custom kernels
    print("Building full CUDA version with custom kernels")
    ext_modules = [
        CUDAExtension(
            name='cuda_pipeline',
            sources=['batched_copy.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                    '-gencode=arch=compute_90,code=sm_90',  # H100
                ]
            }
        )
    ]

setup(
    name='cuda_pipeline',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
