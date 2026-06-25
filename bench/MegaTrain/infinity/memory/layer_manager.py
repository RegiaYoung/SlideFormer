"""Layer residency manager: track and move layers between CPU/GPU."""

import torch
from typing import Dict, List, Any
from enum import Enum


class Device(Enum):
    CPU = "cpu"
    GPU = "cuda"


class LayerResidencyManager:
    """Tracks and controls layer residency on CPU/GPU."""

    def __init__(self, gpu_device: int = 0):
        self.gpu_device = f"cuda:{gpu_device}"
        self.layers: Dict[int, Any] = {}  # layer_id -> layer object
        self.residency: Dict[int, Device] = {}  # layer_id -> current device

    def register(self, layer_id: int, layer: Any, initial_device: Device = Device.CPU) -> None:
        """Register a layer for management."""
        self.layers[layer_id] = layer
        self.residency[layer_id] = initial_device

    def unregister(self, layer_id: int) -> None:
        """Remove a layer from management."""
        self.layers.pop(layer_id, None)
        self.residency.pop(layer_id, None)

    def where(self, layer_id: int) -> Device:
        """Return current device of a layer."""
        return self.residency.get(layer_id, Device.CPU)

    def is_on_gpu(self, layer_id: int) -> bool:
        return self.residency.get(layer_id) == Device.GPU

    def ensure_on_gpu(self, layer_id: int) -> None:
        """Move layer to GPU if not already there."""
        if layer_id not in self.layers:
            raise KeyError(f"Layer {layer_id} not registered")

        if self.residency[layer_id] == Device.GPU:
            return

        layer = self.layers[layer_id]
        self._move_to_device(layer, self.gpu_device)
        self.residency[layer_id] = Device.GPU

    def evict_from_gpu(self, layer_id: int) -> None:
        """Move layer to CPU if on GPU."""
        if layer_id not in self.layers:
            raise KeyError(f"Layer {layer_id} not registered")

        if self.residency[layer_id] == Device.CPU:
            return

        layer = self.layers[layer_id]
        self._move_to_device(layer, "cpu")
        self.residency[layer_id] = Device.CPU

    def gpu_resident_layers(self) -> List[int]:
        """Return list of layer_ids currently on GPU."""
        return [lid for lid, dev in self.residency.items() if dev == Device.GPU]

    def _move_to_device(self, layer: Any, device: str) -> None:
        """Move layer parameters to device."""
        if hasattr(layer, 'to'):
            layer.to(torch.device(device))
        elif hasattr(layer, 'parameters'):
            for param in layer.parameters():
                param.data = param.data.to(device)
