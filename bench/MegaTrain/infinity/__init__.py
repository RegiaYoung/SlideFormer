"""MegaTrain: RAM-Centric Architecture for LLM/VLM Training.

Supports any HuggingFace decoder-only model and VLM:
- Dense models: Llama 2/3/4, Qwen 2/3, Mistral, Phi, Gemma, etc.
- Hybrid attention: Qwen 3.5 (linear + full attention)
- MoE models: Mixtral, DeepSeek-MoE, Qwen3-Next
- VLM: Qwen2-VL, Qwen3-VL, LLaVA, Gemma3-VL, InternVL, GLM-4V, etc.
"""

from .model.cpu_master import CPUMasterModel
from .config.training import CPUMasterConfig
from .data.datasets import ChatDataset, MetaMathDataset, collate_fn

__version__ = "0.3.0"

__all__ = [
    "CPUMasterModel",
    "CPUMasterConfig",
    "ChatDataset",
    "MetaMathDataset",
    "collate_fn",
]
