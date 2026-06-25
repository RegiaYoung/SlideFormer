"""Sliding window executor for layer-by-layer training."""

import torch
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class LayerState(Enum):
    CPU_COLD = "cpu_cold"
    PREFETCHING = "prefetching"
    GPU_READY = "gpu_ready"
    COMPUTING = "computing"
    EVICTING = "evicting"


@dataclass
class LayerHandle:
    """Wrapper for a layer with state tracking."""
    layer: Any
    state: LayerState = LayerState.CPU_COLD
    prefetch_event: Optional[torch.cuda.Event] = None
    compute_event: Optional[torch.cuda.Event] = None


class SlidingWindowExecutor:
    """Execute layers with a sliding GPU window."""

    def __init__(
        self,
        layers: List[Any],
        window_size: int = 3,
        device: int = 0
    ):
        self.handles = [LayerHandle(layer=l) for l in layers]
        self.window_size = window_size
        self.device = f"cuda:{device}"

        self.compute_stream = torch.cuda.Stream(device=device)
        self.copy_stream = torch.cuda.Stream(device=device)

    def _prefetch(self, idx: int) -> None:
        """Async prefetch layer to GPU."""
        h = self.handles[idx]
        if h.state != LayerState.CPU_COLD:
            return

        h.state = LayerState.PREFETCHING
        with torch.cuda.stream(self.copy_stream):
            h.layer.to(torch.device(self.device))
        h.prefetch_event = torch.cuda.Event()
        h.prefetch_event.record(self.copy_stream)
        h.state = LayerState.GPU_READY

    def _evict(self, idx: int) -> None:
        """Async evict layer to CPU."""
        h = self.handles[idx]
        if h.state not in (LayerState.GPU_READY, LayerState.COMPUTING):
            return

        # Wait for compute to finish
        if h.compute_event:
            self.copy_stream.wait_event(h.compute_event)

        h.state = LayerState.EVICTING
        with torch.cuda.stream(self.copy_stream):
            h.layer.to(torch.device("cpu"))
        h.state = LayerState.CPU_COLD
        h.prefetch_event = None
        h.compute_event = None

    def _wait_ready(self, idx: int) -> None:
        """Ensure layer is ready for compute."""
        h = self.handles[idx]
        if h.prefetch_event:
            self.compute_stream.wait_event(h.prefetch_event)

    def forward(
        self,
        x: torch.Tensor,
        save_activations: bool = True
    ) -> tuple:
        """Forward pass with sliding window."""
        N = len(self.handles)
        W = self.window_size
        activations = [] if save_activations else None

        # Prefetch initial window
        for i in range(min(W, N)):
            self._prefetch(i)

        for i in range(N):
            # Prefetch layer i+W
            if i + W < N:
                self._prefetch(i + W)

            # Wait for layer i
            self._wait_ready(i)

            # Save activation
            if save_activations:
                activations.append(x.detach().clone())

            # Compute
            h = self.handles[i]
            h.state = LayerState.COMPUTING
            with torch.cuda.stream(self.compute_stream):
                x = h.layer.forward(x)
            h.compute_event = torch.cuda.Event()
            h.compute_event.record(self.compute_stream)
            h.state = LayerState.GPU_READY

            # Note: Don't evict during forward for training
            # Layers stay on GPU until backward needs them

        return x, activations

    def backward(
        self,
        grad: torch.Tensor,
        activations: List[torch.Tensor],
        backward_fn: Callable
    ) -> torch.Tensor:
        """Backward pass with sliding window (reverse order)."""
        N = len(self.handles)
        W = self.window_size

        # Ensure last W layers are on GPU
        for i in range(max(0, N - W), N):
            self._prefetch(i)

        for i in range(N - 1, -1, -1):
            # Prefetch layer i-W
            if i - W >= 0:
                self._prefetch(i - W)

            # Wait for layer i
            self._wait_ready(i)

            # Compute backward
            h = self.handles[i]
            h.state = LayerState.COMPUTING
            with torch.cuda.stream(self.compute_stream):
                grad = backward_fn(h.layer, grad, activations[i])
            h.compute_event = torch.cuda.Event()
            h.compute_event.record(self.compute_stream)

            # Evict layer i+1 (done with it)
            if i + 1 < N:
                self._evict(i + 1)

        # Evict layer 0
        self._evict(0)

        return grad

    def sync(self) -> None:
        """Wait for all operations to complete."""
        self.compute_stream.synchronize()
        self.copy_stream.synchronize()
