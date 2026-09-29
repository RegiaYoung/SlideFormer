# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0
"""Single-node data parallelism with shared weights and rank 0 CPU updates.

Rank 0 averages gradients and owns all Adam moments. All ranks read the same
FP32 weights. Gradient payloads travel through a bounded shared-memory buffer;
Gloo carries synchronization only.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import os
import socket
import tempfile

import torch
import torch.distributed as dist


@dataclass
class LayerState:
    params: torch.Tensor
    exp_avg: torch.Tensor | None
    exp_avg_sq: torch.Tensor | None
    step: int = 0


class SharedDataParallel:
    """Collectively construct on all local ranks before creating the offloader.

    Call reduce_chunk in identical layer/chunk order from one update thread per
    rank. Other collectives (logging, data loading) use the default process group.
    The caller owns process-group initialization and must drain work before close.
    """

    def __init__(self, chunk_numel=32 * 1024 * 1024, dtype=torch.bfloat16,
                 shm_dir="/dev/shm", timeout_seconds=120):
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before SharedDataParallel")
        if chunk_numel < 1:
            raise ValueError("chunk_numel must be positive")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.chunk_numel = int(chunk_numel)
        self.dtype = dtype
        self.layers = {}
        self.error = None
        self._closed = False
        self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))
        identities = [None] * self.world_size
        dist.all_gather_object(identities, (socket.gethostname(), self.chunk_numel, str(dtype)), group=self.group)
        if len(set(identities)) != 1:
            raise ValueError("SharedDataParallel requires one host and matching chunk size/dtype")
        location = [None]
        if self.rank == 0:
            try:
                location[0] = (tempfile.mkdtemp(prefix="slideformer-dp-", dir=shm_dir), None)
            except OSError as exc:
                location[0] = (None, str(exc))
        dist.broadcast_object_list(location, src=0, group=self.group)
        directory, error = location[0]
        if error:
            raise RuntimeError(f"Cannot create shared-memory directory: {error}")
        self.directory = Path(directory)
        self.gradients = self._allocate("gradients", self.world_size * self.chunk_numel, dtype)
        self.gradients = self.gradients.view(self.world_size, self.chunk_numel)

    def _allocate(self, name, numel, dtype):
        path = self.directory / name
        status = [None]
        if self.rank == 0:
            try:
                with path.open("xb") as stream:
                    # Reserve backing storage now: a later SIGBUS cannot be
                    # propagated as a normal Python allocation error.
                    nbytes = numel * torch.empty((), dtype=dtype, device="cpu").element_size()
                    if nbytes:
                        os.posix_fallocate(stream.fileno(), 0, nbytes)
            except OSError as exc:
                path.unlink(missing_ok=True)
                status[0] = str(exc)
        dist.broadcast_object_list(status, src=0, group=self.group)
        if status[0]:
            raise RuntimeError(f"Cannot allocate {name} in shared memory: {status[0]}")
        tensor = torch.from_file(str(path), shared=True, size=numel, dtype=dtype)
        self.barrier()
        if self.rank == 0:
            path.unlink()  # Mappings survive; process exit releases the storage.
        return tensor

    def register_layer(self, layer_idx, numel):
        specs = [None] * self.world_size
        dist.all_gather_object(specs, (layer_idx, numel), group=self.group)
        if len(set(specs)) != 1 or layer_idx in self.layers:
            raise ValueError("Ranks must register identical layers exactly once")
        params = self._allocate(f"layer-{layer_idx}", numel, torch.float32)
        state = LayerState(
            params,
            torch.zeros(numel, dtype=torch.float32, device="cpu") if self.rank == 0 else None,
            torch.zeros(numel, dtype=torch.float32, device="cpu") if self.rank == 0 else None,
        )
        self.layers[layer_idx] = state
        return state

    def barrier(self):
        dist.barrier(group=self.group)

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError("A distributed update failed") from self.error

    @torch.no_grad()
    def reduce_chunk(self, layer_idx, start, end, local_grad, update):
        """Collect a chunk from every rank, then average and update on rank 0.

        update receives FP32 views: (parameters, mean_gradient, first_moment,
        second_moment). The final barrier publishes all writes and protects the
        shared gradient buffer until rank 0 has consumed its inputs.
        """
        self.raise_if_failed()
        state = self.layers[layer_idx]
        count = end - start
        if not 0 <= start < end <= state.params.numel() or count > self.chunk_numel:
            raise ValueError("Invalid gradient chunk")
        if local_grad.device.type != "cpu" or local_grad.numel() != count or local_grad.dtype != self.dtype:
            raise ValueError("Gradient chunk must match the configured CPU dtype and size")
        self.gradients[self.rank, :count].copy_(local_grad.view(-1))
        self.barrier()
        if self.rank == 0:
            mean = torch.zeros(count, dtype=torch.float32, device="cpu")
            for peer in range(self.world_size):
                mean.add_(self.gradients[peer, :count])
            mean.div_(self.world_size)
            update(state.params[start:end], mean,
                   state.exp_avg[start:end], state.exp_avg_sq[start:end])
        self.barrier()

    def close(self):
        """Release local resources after all ranks have drained their updates."""
        if self._closed:
            return
        self._closed = True
        self.layers.clear()
        self.gradients = None
        if self.rank == 0:
            self.directory.rmdir()
        dist.destroy_process_group(self.group)
