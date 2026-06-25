"""
HuggingFace Decoder-Only Transformer Adapter for Infinity.

This module provides a structural adapter that exposes HuggingFace decoder-only
models (LLaMA, Qwen, GPT-2, Mistral, etc.) to Infinity's explicit scheduling system.

SCOPE LIMITATIONS (by design):
- Only decoder-only autoregressive models are supported
- No encoder-decoder (T5, BART), no MoE, no arbitrary graphs
- Model must have a sequential list of transformer layers
- Forward execution is strictly layer-by-layer

WHAT THIS ADAPTER DOES:
- Discovers model structure via attribute introspection
- Extracts embedding, layers, and output head as separate components
- Exposes them for Infinity to schedule (prefetch/compute/evict)

WHAT THIS ADAPTER DOES NOT DO:
- Does NOT move tensors to GPU (Infinity handles this)
- Does NOT call model.forward() (Infinity schedules layer-by-layer)
- Does NOT register autograd hooks (Infinity uses manual backward)
- Does NOT modify HuggingFace internals
"""

import logging
from typing import List, Optional, Any
from dataclasses import dataclass

import torch.nn as nn

logger = logging.getLogger(__name__)


# Attribute paths to try for discovering decoder layers.
# Order matters: more common paths first.
LAYER_ATTR_PATHS = [
    ("model", "layers"),       # LLaMA, Qwen, Mistral
    ("model", "decoder", "layers"),  # Some decoder variants
    ("transformer", "h"),      # GPT-2, GPT-Neo
    ("decoder", "layers"),     # Generic decoder
    ("layers",),               # Direct attribute
]

# Attribute paths for embedding layer
EMBED_ATTR_PATHS = [
    ("model", "embed_tokens"),      # LLaMA, Qwen, Mistral
    ("transformer", "wte"),         # GPT-2
    ("model", "decoder", "embed_tokens"),
    ("embed_tokens",),
]

# Attribute paths for output head
HEAD_ATTR_PATHS = [
    ("lm_head",),                   # Most models
    ("transformer", "lm_head"),
]

# Attribute paths for final layer norm (applied before lm_head)
FINAL_NORM_PATHS = [
    ("model", "norm"),              # LLaMA, Qwen, Mistral
    ("transformer", "ln_f"),        # GPT-2
    ("model", "decoder", "final_layer_norm"),
    ("norm",),
]


def _get_nested_attr(obj: Any, path: tuple) -> Optional[Any]:
    """Traverse nested attributes. Returns None if any step fails."""
    for attr in path:
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def _find_attr(obj: Any, paths: List[tuple], name: str) -> Any:
    """Try multiple attribute paths, return first match or raise."""
    for path in paths:
        result = _get_nested_attr(obj, path)
        if result is not None:
            logger.info(f"Found {name} at: {'.'.join(path)}")
            return result

    tried = [".".join(p) for p in paths]
    raise AttributeError(
        f"Could not find {name}. Tried: {tried}. "
        f"This model may not be a supported decoder-only architecture."
    )


@dataclass
class HFModelComponents:
    """Extracted components from a HuggingFace model."""
    embedding: nn.Module
    layers: List[nn.Module]
    final_norm: Optional[nn.Module]
    head: nn.Module
    config: Any
    layer_path: str  # For debugging: which path was used


class HFDecoderAdapter:
    """
    Adapter that exposes HuggingFace decoder-only model structure to Infinity.

    This adapter performs STRUCTURAL INTROSPECTION ONLY. It does not:
    - Execute forward passes (Infinity does this layer-by-layer)
    - Move parameters between devices (Infinity's memory manager does this)
    - Register hooks or modify the model

    Infinity will use the exposed components as follows:

        # Infinity's execution loop (pseudocode):
        for layer in adapter.layers:
            prefetch(layer)      # Copy layer params CPU -> GPU
            x = layer.forward(x) # Compute on GPU
            evict(layer)         # Copy layer params GPU -> CPU

    Usage:
        from transformers import AutoModelForCausalLM
        from infinity.adapters import HFDecoderAdapter

        hf_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Keep on CPU; Infinity manages GPU
        )

        adapter = HFDecoderAdapter(hf_model)

        # Pass to Infinity engine
        engine = InfinityEngine(
            embedding=adapter.embedding,
            layers=adapter.layers,
            head=adapter.head,
        )
    """

    def __init__(self, model: nn.Module):
        """
        Initialize adapter by discovering model structure.

        Args:
            model: A HuggingFace AutoModelForCausalLM instance.
                   Must be a decoder-only transformer.
                   Should be loaded with device_map="cpu".

        Raises:
            AttributeError: If model structure cannot be discovered.
            TypeError: If model is not an nn.Module.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model)}")

        self._model = model
        self._components = self._discover_structure(model)

        logger.info(
            f"HFDecoderAdapter initialized: "
            f"{len(self._components.layers)} layers, "
            f"layer_path={self._components.layer_path}"
        )

    def _discover_structure(self, model: nn.Module) -> HFModelComponents:
        """
        Discover decoder-only transformer structure via introspection.

        Structural assumptions:
        1. Model has a list/tuple of decoder layers (attention + MLP)
        2. Model has an embedding layer (token -> hidden)
        3. Model has an output head (hidden -> vocab logits)
        4. Model may have a final layer norm before the head

        These assumptions hold for: LLaMA, Qwen, Mistral, GPT-2, GPT-Neo,
        Falcon, MPT, and most other decoder-only HuggingFace models.
        """
        # Find decoder layers (required)
        layers = None
        layer_path = None
        for path in LAYER_ATTR_PATHS:
            result = _get_nested_attr(model, path)
            if result is not None and hasattr(result, '__len__') and len(result) > 0:
                layers = list(result)
                layer_path = ".".join(path)
                logger.info(f"Found {len(layers)} decoder layers at: {layer_path}")
                break

        if layers is None:
            tried = [".".join(p) for p in LAYER_ATTR_PATHS]
            raise AttributeError(
                f"Could not find decoder layers. Tried: {tried}. "
                f"Model type: {type(model).__name__}. "
                f"This may not be a decoder-only transformer."
            )

        # Find embedding (required)
        embedding = _find_attr(model, EMBED_ATTR_PATHS, "embedding")

        # Find output head (required)
        head = _find_attr(model, HEAD_ATTR_PATHS, "lm_head")

        # Find final norm (optional, some models apply it inside layers)
        final_norm = None
        for path in FINAL_NORM_PATHS:
            result = _get_nested_attr(model, path)
            if result is not None:
                final_norm = result
                logger.info(f"Found final_norm at: {'.'.join(path)}")
                break

        if final_norm is None:
            logger.warning(
                "No final_norm found. If generation quality is poor, "
                "the model may require a final norm before lm_head."
            )

        # Extract config if available
        config = getattr(model, "config", None)

        return HFModelComponents(
            embedding=embedding,
            layers=layers,
            final_norm=final_norm,
            head=head,
            config=config,
            layer_path=layer_path,
        )

    # =========================================================================
    # Public interface for Infinity
    # =========================================================================

    @property
    def embedding(self) -> nn.Module:
        """
        Token embedding module.

        Infinity execution:
            prefetch(adapter.embedding)
            hidden = adapter.embedding(input_ids)
            evict(adapter.embedding)  # Optional: embedding is small
        """
        return self._components.embedding

    @property
    def layers(self) -> List[nn.Module]:
        """
        List of decoder layers in execution order.

        Each layer is a transformer block (attention + MLP + norms).
        Infinity executes these sequentially with sliding-window scheduling:

            for i, layer in enumerate(adapter.layers):
                # Infinity scheduler handles:
                # - prefetch(layer)           # Async CPU->GPU copy
                # - wait_for_prefetch(layer)  # Ensure params ready
                # - hidden = layer(hidden)    # Forward on GPU
                # - evict(layer)              # Async GPU->CPU copy
        """
        return self._components.layers

    @property
    def final_norm(self) -> Optional[nn.Module]:
        """
        Final layer normalization (applied after last decoder layer).

        May be None if the model applies norm inside each layer.
        If present, Infinity should apply it before the output head:

            hidden = adapter.final_norm(hidden)  # If not None
            logits = adapter.head(hidden)
        """
        return self._components.final_norm

    @property
    def head(self) -> nn.Module:
        """
        Output projection (hidden -> vocab logits).

        Infinity execution:
            prefetch(adapter.head)
            logits = adapter.head(hidden)
            evict(adapter.head)  # Optional: head may be large (vocab_size)
        """
        return self._components.head

    @property
    def config(self) -> Optional[Any]:
        """
        HuggingFace model config (passthrough).

        Useful for extracting:
            - config.hidden_size
            - config.num_hidden_layers
            - config.num_attention_heads
            - config.vocab_size
        """
        return self._components.config

    @property
    def num_layers(self) -> int:
        """Number of decoder layers."""
        return len(self._components.layers)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def layer_parameter_count(self, layer_idx: int) -> int:
        """Count parameters in a specific layer (for memory planning)."""
        layer = self._components.layers[layer_idx]
        return sum(p.numel() for p in layer.parameters())

    def total_parameter_count(self) -> int:
        """Total parameters across all components."""
        total = sum(p.numel() for p in self._components.embedding.parameters())
        for layer in self._components.layers:
            total += sum(p.numel() for p in layer.parameters())
        if self._components.final_norm is not None:
            total += sum(p.numel() for p in self._components.final_norm.parameters())
        total += sum(p.numel() for p in self._components.head.parameters())
        return total

    def __repr__(self) -> str:
        return (
            f"HFDecoderAdapter("
            f"num_layers={self.num_layers}, "
            f"layer_path='{self._components.layer_path}', "
            f"has_final_norm={self.final_norm is not None})"
        )
