"""Sliding window executor with admission control to avoid GPU stalls."""

import torch
from typing import List, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock


class LayerState(Enum):
    CPU_COLD = 0
    PREFETCHING = 1
    GPU_READY = 2
    COMPUTING = 3


@dataclass
class LayerSlot:
    layer: Any
    state: LayerState = LayerState.CPU_COLD
    prefetch_event: Optional[torch.cuda.Event] = None
    compute_event: Optional[torch.cuda.Event] = None


class AdmissionController:
    """
    Controls execution advancement based on buffer occupancy.

    Invariants:
        - min_buffer <= ready_count <= max_buffer (window_size)
        - Compute advances only when ready_count >= min_buffer
        - Prefetch runs ahead up to max_buffer layers

    Steady-state:
        prefetch_ptr = compute_ptr + max_buffer
        ready_count oscillates between min_buffer and max_buffer
    """

    def __init__(
        self,
        layers: List[Any],
        window_size: int = 4,
        min_buffer: int = 2,
        device: int = 0
    ):
        assert min_buffer <= window_size, "min_buffer must be <= window_size"

        self.slots = [LayerSlot(layer=l) for l in layers]
        self.num_layers = len(layers)
        self.window_size = window_size  # max_buffer
        self.min_buffer = min_buffer
        self.device = f"cuda:{device}"

        # Execution pointers
        self.prefetch_ptr = 0  # Next layer to prefetch
        self.compute_ptr = 0   # Next layer to compute
        self.evict_ptr = 0     # Next layer to evict

        # Streams
        self.compute_stream = torch.cuda.Stream(device=device)
        self.copy_stream = torch.cuda.Stream(device=device)

        # Counts
        self._ready_count = 0  # Layers in GPU_READY state
        self._inflight_count = 0  # Layers in PREFETCHING state

    @property
    def buffer_occupancy(self) -> int:
        """Layers ready or in-flight."""
        return self._ready_count + self._inflight_count

    def can_prefetch(self) -> bool:
        """Check if we can issue another prefetch."""
        return (
            self.prefetch_ptr < self.num_layers and
            self.buffer_occupancy < self.window_size
        )

    def can_compute(self) -> bool:
        """Check if we can advance compute (admission control)."""
        return (
            self.compute_ptr < self.num_layers and
            self._ready_count >= self.min_buffer
        )

    def must_wait(self) -> bool:
        """Check if compute must stall for prefetch."""
        return (
            self.compute_ptr < self.num_layers and
            self._ready_count < self.min_buffer
        )

    def _prefetch_one(self) -> None:
        """Issue prefetch for next layer."""
        if not self.can_prefetch():
            return

        idx = self.prefetch_ptr
        slot = self.slots[idx]

        slot.state = LayerState.PREFETCHING
        self._inflight_count += 1

        with torch.cuda.stream(self.copy_stream):
            slot.layer.to(torch.device(self.device))

        slot.prefetch_event = torch.cuda.Event()
        slot.prefetch_event.record(self.copy_stream)

        self.prefetch_ptr += 1

    def _check_prefetch_completion(self) -> None:
        """Promote completed prefetches to GPU_READY."""
        for slot in self.slots:
            if slot.state == LayerState.PREFETCHING:
                if slot.prefetch_event and slot.prefetch_event.query():
                    slot.state = LayerState.GPU_READY
                    self._inflight_count -= 1
                    self._ready_count += 1

    def _evict_one(self, idx: int) -> None:
        """Evict a layer back to CPU."""
        slot = self.slots[idx]
        if slot.state not in (LayerState.GPU_READY, LayerState.COMPUTING):
            return

        if slot.compute_event:
            self.copy_stream.wait_event(slot.compute_event)

        with torch.cuda.stream(self.copy_stream):
            slot.layer.to(torch.device("cpu"))

        slot.state = LayerState.CPU_COLD
        slot.prefetch_event = None
        slot.compute_event = None
        self._ready_count = max(0, self._ready_count - 1)

    def _wait_for_layer(self, idx: int) -> None:
        """Block until layer idx is ready."""
        slot = self.slots[idx]
        if slot.state == LayerState.PREFETCHING:
            slot.prefetch_event.synchronize()
            slot.state = LayerState.GPU_READY
            self._inflight_count -= 1
            self._ready_count += 1

        if slot.prefetch_event:
            self.compute_stream.wait_event(slot.prefetch_event)

    def step_forward(self, x: torch.Tensor) -> tuple:
        """
        Execute one forward step with admission control.
        Returns (output, activation, done).
        """
        # Fill prefetch buffer
        while self.can_prefetch():
            self._prefetch_one()

        self._check_prefetch_completion()

        # Admission control: wait if buffer too low
        if self.must_wait():
            # Force wait for next prefetch to complete
            if self._inflight_count > 0:
                for slot in self.slots:
                    if slot.state == LayerState.PREFETCHING:
                        slot.prefetch_event.synchronize()
                        slot.state = LayerState.GPU_READY
                        self._inflight_count -= 1
                        self._ready_count += 1
                        break

        if self.compute_ptr >= self.num_layers:
            return x, None, True

        # Execute compute
        idx = self.compute_ptr
        slot = self.slots[idx]

        self._wait_for_layer(idx)

        activation = x.detach().clone()

        slot.state = LayerState.COMPUTING
        self._ready_count -= 1

        with torch.cuda.stream(self.compute_stream):
            x = slot.layer.forward(x)

        slot.compute_event = torch.cuda.Event()
        slot.compute_event.record(self.compute_stream)

        self.compute_ptr += 1

        return x, activation, False

    def step_backward(
        self,
        grad: torch.Tensor,
        activation: torch.Tensor,
        backward_fn: Callable
    ) -> tuple:
        """
        Execute one backward step with admission control.
        Returns (grad, done).
        """
        # For backward, we go in reverse
        # Prefetch layers we'll need (going backwards)
        while self.can_prefetch():
            self._prefetch_one()

        self._check_prefetch_completion()

        if self.must_wait() and self._inflight_count > 0:
            for slot in self.slots:
                if slot.state == LayerState.PREFETCHING:
                    slot.prefetch_event.synchronize()
                    slot.state = LayerState.GPU_READY
                    self._inflight_count -= 1
                    self._ready_count += 1
                    break

        if self.compute_ptr < 0:
            return grad, True

        idx = self.compute_ptr
        slot = self.slots[idx]

        self._wait_for_layer(idx)

        slot.state = LayerState.COMPUTING
        self._ready_count -= 1

        with torch.cuda.stream(self.compute_stream):
            grad = backward_fn(slot.layer, grad, activation)

        slot.compute_event = torch.cuda.Event()
        slot.compute_event.record(self.compute_stream)

        # Evict layer we're done with
        if idx + 1 < self.num_layers:
            self._evict_one(idx + 1)

        self.compute_ptr -= 1

        return grad, False

    def forward(self, x: torch.Tensor) -> tuple:
        """Full forward pass with admission control."""
        activations = []

        while True:
            x, act, done = self.step_forward(x)
            if act is not None:
                activations.append(act)
            if done:
                break

        return x, activations

    def backward(
        self,
        grad: torch.Tensor,
        activations: List[torch.Tensor],
        backward_fn: Callable
    ) -> torch.Tensor:
        """Full backward pass with admission control."""
        # Reset pointers for backward
        self.prefetch_ptr = self.num_layers - 1
        self.compute_ptr = self.num_layers - 1
        self._ready_count = 0
        self._inflight_count = 0

        # Reset layer states
        for slot in self.slots:
            slot.state = LayerState.CPU_COLD
            slot.prefetch_event = None

        act_idx = len(activations) - 1
        while True:
            act = activations[act_idx] if act_idx >= 0 else None
            grad, done = self.step_backward(grad, act, backward_fn)
            act_idx -= 1
            if done:
                break

        # Evict remaining
        self._evict_one(0)

        return grad

    def sync(self) -> None:
        self.compute_stream.synchronize()
        self.copy_stream.synchronize()

    def stats(self) -> dict:
        """Return current state for debugging."""
        return {
            "prefetch_ptr": self.prefetch_ptr,
            "compute_ptr": self.compute_ptr,
            "ready_count": self._ready_count,
            "inflight_count": self._inflight_count,
            "buffer_occupancy": self.buffer_occupancy,
        }
