"""Memory manager: allocation, eviction policy, accounting."""

from typing import Dict, List, Optional
from .buffer import PinnedBuffer, GPUCacheSlot


class MemoryManager:
    """Manages GPU cache slots and CPU pinned buffers."""

    def __init__(self, num_gpu_slots: int, slot_size: int, device: int = 0):
        self.num_gpu_slots = num_gpu_slots
        self.slot_size = slot_size
        self.device = device

        self.gpu_slots: List[GPUCacheSlot] = []
        self.tensor_to_slot: Dict[int, int] = {}  # tensor_id -> slot_index
        self.access_order: List[int] = []  # LRU tracking: most recent at end

    def initialize(self) -> None:
        """Allocate all GPU cache slots."""
        for _ in range(self.num_gpu_slots):
            slot = GPUCacheSlot(self.slot_size, device=self.device)
            slot.allocate()
            self.gpu_slots.append(slot)

    def shutdown(self) -> None:
        """Free all GPU memory."""
        for slot in self.gpu_slots:
            slot.free()
        self.gpu_slots.clear()
        self.tensor_to_slot.clear()
        self.access_order.clear()

    def acquire_slot(self, tensor_id: int) -> Optional[GPUCacheSlot]:
        """Get a GPU slot for tensor_id. Returns None if eviction needed."""
        # Already resident?
        if tensor_id in self.tensor_to_slot:
            self._touch(tensor_id)
            return self.gpu_slots[self.tensor_to_slot[tensor_id]]

        # Find free slot
        for i, slot in enumerate(self.gpu_slots):
            if slot.is_free():
                slot.occupied_by = tensor_id
                self.tensor_to_slot[tensor_id] = i
                self._touch(tensor_id)
                return slot

        return None  # Caller must evict first

    def evict_lru(self) -> Optional[int]:
        """Evict least recently used tensor. Returns evicted tensor_id."""
        if not self.access_order:
            return None

        victim_id = self.access_order.pop(0)
        slot_idx = self.tensor_to_slot.pop(victim_id)
        self.gpu_slots[slot_idx].occupied_by = None
        return victim_id

    def release(self, tensor_id: int) -> None:
        """Explicitly release a tensor's GPU slot."""
        if tensor_id in self.tensor_to_slot:
            slot_idx = self.tensor_to_slot.pop(tensor_id)
            self.gpu_slots[slot_idx].occupied_by = None
            if tensor_id in self.access_order:
                self.access_order.remove(tensor_id)

    def is_resident(self, tensor_id: int) -> bool:
        return tensor_id in self.tensor_to_slot

    def _touch(self, tensor_id: int) -> None:
        """Update LRU order."""
        if tensor_id in self.access_order:
            self.access_order.remove(tensor_id)
        self.access_order.append(tensor_id)
