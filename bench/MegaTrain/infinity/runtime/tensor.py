"""CPU-backed tensor with explicit GPU residency control."""

import torch
from typing import Optional
from ..memory import MemoryManager, GPUCacheSlot
from .stream import Stream, copy_async


_next_tensor_id = 0


def _get_tensor_id() -> int:
    global _next_tensor_id
    _next_tensor_id += 1
    return _next_tensor_id


class ManagedTensor:
    """Tensor stored on CPU with explicit GPU caching."""

    def __init__(
        self,
        data: torch.Tensor,
        memory_manager: Optional[MemoryManager] = None,
        pin_memory: bool = True
    ):
        self.id = _get_tensor_id()
        self.shape = data.shape
        self.dtype = data.dtype
        self.numel = data.numel()

        # CPU is the backing store (pinned for async transfers)
        if pin_memory and not data.is_pinned():
            self.cpu_data = data.pin_memory()
        else:
            self.cpu_data = data.cpu() if data.is_cuda else data

        self.memory_manager = memory_manager
        self._gpu_view: Optional[torch.Tensor] = None  # View into cache slot

    def prefetch(self, stream: Optional[Stream] = None) -> None:
        """Copy to GPU cache. Must have memory_manager set."""
        if self._gpu_view is not None:
            return  # Already resident

        assert self.memory_manager is not None, "No memory manager"

        slot = self.memory_manager.acquire_slot(self.id)
        if slot is None:
            # Need to evict
            evicted_id = self.memory_manager.evict_lru()
            assert evicted_id is not None, "No slot available"
            slot = self.memory_manager.acquire_slot(self.id)

        assert slot is not None and slot.data is not None
        self._gpu_view = slot.data[:self.numel].view(self.shape)
        copy_async(self.cpu_data, self._gpu_view, stream)

    def evict(self, stream: Optional[Stream] = None) -> None:
        """Copy back to CPU and release GPU slot."""
        if self._gpu_view is None:
            return

        # Sync before evict to ensure compute is done
        if stream:
            stream.synchronize()

        copy_async(self._gpu_view, self.cpu_data, stream)
        if stream:
            stream.synchronize()

        if self.memory_manager:
            self.memory_manager.release(self.id)
        self._gpu_view = None

    def gpu(self) -> torch.Tensor:
        """Get GPU tensor. Must be prefetched first."""
        assert self._gpu_view is not None, "Tensor not on GPU. Call prefetch() first."
        return self._gpu_view

    def cpu(self) -> torch.Tensor:
        """Get CPU tensor."""
        return self.cpu_data

    def is_resident(self) -> bool:
        return self._gpu_view is not None
