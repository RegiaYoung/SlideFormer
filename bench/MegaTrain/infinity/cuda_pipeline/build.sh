#!/bin/bash

# Build script for CUDA pipeline extension

set -e

echo "Building CUDA pipeline extension..."

cd "$(dirname "$0")"

# Check PyTorch installation
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'" || {
    echo "ERROR: PyTorch with CUDA support is required."
    exit 1
}

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info *.so

# Determine build mode
BUILD_MODE="${1:-simple}"

if [ "$BUILD_MODE" = "simple" ]; then
    echo ""
    echo "Building SIMPLE version (C++ only, no custom CUDA kernels)..."
    echo "This version is more compatible but may be slightly slower."
    echo ""
    python setup.py build_ext --inplace --simple
elif [ "$BUILD_MODE" = "cuda" ]; then
    # Check if CUDA is available
    if ! command -v nvcc &> /dev/null; then
        echo "ERROR: nvcc not found. Please install CUDA toolkit or use 'simple' mode."
        echo "Usage: $0 [simple|cuda]"
        exit 1
    fi
    echo ""
    echo "Building CUDA version (with custom CUDA kernels)..."
    echo "This version has custom kernels for maximum performance."
    echo ""
    python setup.py build_ext --inplace
else
    echo "Usage: $0 [simple|cuda]"
    echo "  simple - Build C++ version (default, more compatible)"
    echo "  cuda   - Build CUDA version (requires nvcc, maximum performance)"
    exit 1
fi

# Install extension
echo ""
echo "Installing extension..."
pip install -e .

echo ""
echo "Build complete!"
echo ""
echo "To verify installation, run:"
echo "  python test_extension.py"
