"""Python wrapper for C++/CUDA memory operations."""

import torch
from typing import List, Optional

try:
    import infinity_memory_ops as _C
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False
    _C = None


def _check_ext():
    if not HAS_CUDA_EXT:
        raise RuntimeError("CUDA extension not built. Run: cd csrc && pip install .")


def _get_stream_ptr(stream: Optional[torch.cuda.Stream] = None) -> int:
    """Get raw CUDA stream pointer."""
    if stream is None:
        return _C.get_current_stream_ptr()
    else:
        return stream.cuda_stream


class PinnedPool:
    """Pinned memory buffer pool."""

    def __init__(self, buffer_size: int, num_buffers: int):
        _check_ext()
        _C.init_pool(buffer_size, num_buffers)
        self.buffer_size = buffer_size
        self.num_buffers = num_buffers

    def __del__(self):
        if HAS_CUDA_EXT:
            _C.destroy_pool()

    def acquire(self) -> int:
        """Acquire a buffer. Returns -1 if none available."""
        return _C.pool_acquire()

    def release(self, idx: int) -> None:
        """Release a buffer back to pool."""
        _C.pool_release(idx)

    def num_free(self) -> int:
        return _C.pool_num_free()

    def as_tensor(self, idx: int, shape: List[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """View buffer as a pinned tensor (no copy)."""
        return _C.pool_to_tensor(idx, shape, dtype)


class Event:
    """CUDA event wrapper."""

    def __init__(self):
        _check_ext()
        self._ptr = _C.event_create()

    def __del__(self):
        if HAS_CUDA_EXT and self._ptr:
            _C.event_destroy(self._ptr)

    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        stream_ptr = _get_stream_ptr(stream)
        _C.event_record(self._ptr, stream_ptr)

    def query(self) -> bool:
        """Returns True if event completed."""
        return _C.event_query(self._ptr)

    def synchronize(self) -> None:
        """Block until event completes."""
        _C.event_synchronize(self._ptr)

    def elapsed_time(self, end_event: 'Event') -> float:
        """Milliseconds between this event and end_event."""
        return _C.event_elapsed_time(self._ptr, end_event._ptr)


def stream_wait_event(stream: torch.cuda.Stream, event: Event) -> None:
    """Make stream wait for event."""
    _check_ext()
    stream_ptr = _get_stream_ptr(stream)
    _C.stream_wait_event(stream_ptr, event._ptr)


def copy_h2d_async(
    dst: torch.Tensor,
    pool_idx: int,
    num_bytes: int,
    stream: Optional[torch.cuda.Stream] = None
) -> None:
    """Async copy from pinned pool buffer to GPU tensor."""
    _check_ext()
    stream_ptr = _get_stream_ptr(stream)
    _C.memcpy_h2d_async(dst, pool_idx, num_bytes, stream_ptr)


def copy_d2h_async(
    pool_idx: int,
    src: torch.Tensor,
    num_bytes: int,
    stream: Optional[torch.cuda.Stream] = None
) -> None:
    """Async copy from GPU tensor to pinned pool buffer."""
    _check_ext()
    stream_ptr = _get_stream_ptr(stream)
    _C.memcpy_d2h_async(pool_idx, src, num_bytes, stream_ptr)
