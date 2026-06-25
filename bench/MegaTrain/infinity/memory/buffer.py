"""Pinned CPU buffers and GPU cache slots."""

import torch
from typing import Optional


class PinnedBuffer:
    """A pinned CPU memory buffer for async transfers."""

    def __init__(self, size: int, dtype: torch.dtype = torch.float32):
        self.size = size
        self.dtype = dtype
        self.data: Optional[torch.Tensor] = None

    def allocate(self) -> None:
        """Allocate pinned memory. Call explicitly before use."""
        self.data = torch.empty(self.size, dtype=self.dtype, pin_memory=True)

    def free(self) -> None:
        """Release the buffer."""
        self.data = None


class GPUCacheSlot:
    """A fixed-size slot in GPU memory cache."""

    def __init__(self, size: int, dtype: torch.dtype = torch.float32, device: int = 0):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.data: Optional[torch.Tensor] = None
        self.occupied_by: Optional[int] = None  # tensor_id currently in this slot

    def allocate(self) -> None:
        """Allocate GPU memory. Call explicitly."""
        self.data = torch.empty(self.size, dtype=self.dtype, device=f"cuda:{self.device}")

    def free(self) -> None:
        """Release GPU memory."""
        self.data = None
        self.occupied_by = None

    def is_free(self) -> bool:
        return self.occupied_by is None
