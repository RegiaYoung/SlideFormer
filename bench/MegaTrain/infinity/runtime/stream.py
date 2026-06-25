"""CUDA stream abstraction for explicit copy/compute overlap."""

import torch
from typing import Optional


class Stream:
    """Wrapper around torch.cuda.Stream for explicit scheduling."""

    def __init__(self, device: int = 0):
        self.device = device
        self._stream: Optional[torch.cuda.Stream] = None

    def create(self) -> None:
        """Create the CUDA stream. Call explicitly."""
        self._stream = torch.cuda.Stream(device=self.device)

    def destroy(self) -> None:
        """Release stream resources."""
        self._stream = None

    @property
    def raw(self) -> torch.cuda.Stream:
        """Access underlying torch stream."""
        assert self._stream is not None, "Stream not created"
        return self._stream

    def synchronize(self) -> None:
        """Block until all ops on this stream complete."""
        if self._stream:
            self._stream.synchronize()

    def record_event(self) -> torch.cuda.Event:
        """Record an event on this stream."""
        event = torch.cuda.Event()
        event.record(self._stream)
        return event

    def wait_event(self, event: torch.cuda.Event) -> None:
        """Make this stream wait for an event."""
        if self._stream:
            self._stream.wait_event(event)

    def __enter__(self):
        """Context manager for stream scope."""
        if self._stream:
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        if self._stream:
            self._stream.__exit__(*args)


def copy_async(
    src: torch.Tensor,
    dst: torch.Tensor,
    stream: Optional[Stream] = None
) -> None:
    """Async copy from src to dst on given stream."""
    ctx = stream if stream else torch.cuda.default_stream()
    with ctx:
        dst.copy_(src, non_blocking=True)
