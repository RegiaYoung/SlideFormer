"""Setup script for MegaTrain: Single-GPU Large Model Training Toolkit."""

from setuptools import setup, find_packages
from pathlib import Path
import sys
import os

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Try to import torch and CUDA extension builder
ext_modules = []
cmdclass = {}

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA detected! Building CUDA pipeline extension...")
        
        # CUDA extension for optimized operations
        cuda_ext = CUDAExtension(
            name='cuda_pipeline',
            sources=['infinity/cuda_pipeline/batched_copy.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                ]
            }
        )
        ext_modules.append(cuda_ext)
        cmdclass['build_ext'] = BuildExtension
        print("✓ CUDA extension will be built")
    else:
        print("CUDA not available, skipping CUDA extension")
        
except ImportError:
    print("PyTorch not installed yet, skipping CUDA extension")
    print("You can install CUDA extension later by running: pip install -e .")

setup(
    name="megatrain",
    version="0.2.0",
    author="MegaTrain Team",
    description="MegaTrain: Single-GPU Large Model Training with CPU-backed parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DLYuanGod/MegaTrain",
    packages=find_packages(include=["infinity", "infinity.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "psutil>=5.9.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "flash-attn": [
            "flash-attn>=2.0.0",
        ],
        "deepspeed": [
            "deepspeed>=0.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="deep-learning, large-language-models, training, gpu, cpu-offloading, megatrain",
)
