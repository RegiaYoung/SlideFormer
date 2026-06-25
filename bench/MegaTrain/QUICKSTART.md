# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd MegaTrain

# Install (automatically builds CUDA extension if CUDA is available)
pip install -e .

# Optional: Install with extra features
pip install -e ".[flash-attn,deepspeed]"
```

That's it! The installation will automatically:
- Install all Python dependencies
- Build and install the CUDA pipeline extension (if CUDA is available)
- Set up the infinity library for import

## Usage

### Using YAML Configuration (Recommended)

```bash
# Train with Qwen 32B
python examples/train_cpu_master_v10.py --config examples/configs/qwen_32b.yaml

# Train with Qwen 7B
python examples/train_cpu_master_v10.py --config examples/configs/qwen_7b.yaml

# Train with LLaMA 3 8B
python examples/train_cpu_master_v10.py --config examples/configs/llama3_8b.yaml
```

### Override Configuration

```bash
# Override specific parameters
python examples/train_cpu_master_v10.py \
    --config examples/configs/qwen_32b.yaml \
    --batch-size 64 \
    --num-steps 500
```

### Using in Python

```python
from infinity import CPUMasterModel, CPUMasterConfig, MetaMathDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load configuration
config = CPUMasterConfig(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    dataset_path="/path/to/dataset",
    batch_size=96,
    max_seq_len=1024,
)

# Load model
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
hf_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

# Create CPU Master model
model = CPUMasterModel(hf_model, config)

# Train...
```

## Configuration Files

Configuration files are located in `examples/configs/`:

- **qwen_32b.yaml** - Qwen 2.5 32B (production)
- **qwen_7b.yaml** - Qwen 2.5 7B (development/testing)
- **llama3_8b.yaml** - LLaMA 3 8B
- **train_config.yaml** - Template with detailed comments

See `examples/configs/README.md` for detailed configuration guide.

## What Gets Installed

When you run `pip install -e .`:

1. **Python Package**: `megatrain` package with all modules
2. **CUDA Extension**: `cuda_pipeline` for optimized GPU operations (if CUDA available)
3. **Dependencies**: PyTorch, Transformers, Datasets, etc.

## Verify Installation

```python
# Test imports
from infinity import CPUMasterModel, CPUMasterConfig, MetaMathDataset
print("✓ Infinity library installed successfully")

# Test CUDA extension (optional)
try:
    import cuda_pipeline
    print("✓ CUDA pipeline extension available")
except ImportError:
    print("✗ CUDA pipeline extension not available (optional)")
```

## Requirements

- **GPU**: NVIDIA GPU with 40GB+ VRAM (A100, H100)
- **CPU RAM**: 256GB+ for 32B models
- **CUDA**: 11.8+ (for CUDA extension)
- **PyTorch**: 2.0+
- **Python**: 3.9+

## Troubleshooting

### CUDA Extension Build Failed

If CUDA extension fails to build, the package will still install without it. The training will work but may be slower.

To manually build CUDA extension:
```bash
cd infinity/cuda_pipeline
python setup.py install
```

### Import Errors

Make sure you're in the correct directory:
```bash
cd MegaTrain
pip install -e .
```

### Out of Memory

Reduce batch size in your config file:
```yaml
training:
  batch_size: 64  # Reduce from 96
```

## Next Steps

- Read the [Configuration Guide](examples/configs/README.md)
- Check the [Main README](README.md) for architecture details
- Explore example configurations in `examples/configs/`
