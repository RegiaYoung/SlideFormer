#!/usr/bin/env python3
"""
Simple compilation test for CUDA extension.
This script attempts to build the extension and reports any errors.
"""

import sys
import subprocess
import os

def test_compilation():
    print("=" * 70)
    print("CUDA Extension Compilation Test")
    print("=" * 70)
    print()

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available in PyTorch")
            return False
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print()
    except ImportError:
        print("❌ PyTorch not installed")
        return False

    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'],
                              capture_output=True, text=True, check=True)
        print("✓ nvcc found:")
        print(result.stdout.split('\n')[-2])
        print()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ nvcc not found. Please install CUDA toolkit.")
        return False

    # Try to build
    print("Attempting to build extension...")
    print("-" * 70)

    try:
        result = subprocess.run(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:")
            print(result.stderr)
        print("-" * 70)
        print("✓ Build successful!")
        print()

        # Try to import
        try:
            import cuda_pipeline
            print("✓ Extension imported successfully!")
            print()
            return True
        except ImportError as e:
            print(f"❌ Failed to import extension: {e}")
            return False

    except subprocess.CalledProcessError as e:
        print("❌ Build failed!")
        print()
        print("STDOUT:")
        print(e.stdout)
        print()
        print("STDERR:")
        print(e.stderr)
        print()

        # Try to identify common issues
        stderr_lower = e.stderr.lower()
        if 'no member' in stderr_lower or 'not declared' in stderr_lower:
            print("💡 Hint: API compatibility issue detected.")
            print("   This usually means the CUDA extension code uses outdated PyTorch APIs.")
        elif 'compute capability' in stderr_lower:
            print("💡 Hint: GPU compute capability mismatch.")
            print("   Try adjusting the -gencode flags in setup.py.")
        elif 'cannot find' in stderr_lower:
            print("💡 Hint: Missing library or header file.")
            print("   Ensure CUDA toolkit and PyTorch are properly installed.")

        return False

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = test_compilation()
    sys.exit(0 if success else 1)
