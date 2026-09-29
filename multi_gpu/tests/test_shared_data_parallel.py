# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only numerical and multi-process checks; no CUDA or native build needed."""

from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_data_parallel import SharedDataParallel


def worker(rank, world_size, rendezvous, native):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=45))
    try:
        if native:
            from optimizer import LayerAdam
            adam = LayerAdam(lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.1)
        for dtype in (torch.float32, torch.bfloat16):
            dp = SharedDataParallel(chunk_numel=5, dtype=dtype, timeout_seconds=30)
            sizes = (19, 3, 0)
            references = []
            optimizers = []
            for layer, size in enumerate(sizes):
                state = dp.register_layer(layer, size)
                initial = torch.linspace(-0.5, 0.5, size)
                if rank == 0:
                    state.params.copy_(initial)
                dp.barrier()
                reference = torch.nn.Parameter(initial.clone())
                references.append(reference)
                optimizers.append(torch.optim.AdamW([reference], lr=0.01, betas=(0.8, 0.95),
                                                     eps=1e-6, weight_decay=0.1, foreach=False))
                if rank == 0:
                    assert state.exp_avg.numel() == state.exp_avg_sq.numel() == size
                else:
                    assert state.exp_avg is None and state.exp_avg_sq is None
                assert list(dp.directory.iterdir()) == []  # Mapped files already unlinked.

            state_count = torch.tensor(sum(s.exp_avg.numel() for s in dp.layers.values()
                                           if s.exp_avg is not None))
            dist.all_reduce(state_count)
            assert state_count.item() == sum(sizes)  # Exactly one global copy of each moment.

            for step in range(1, 6):
                for layer, size in enumerate(sizes):
                    layer_state = dp.layers[layer]
                    gradients = [((torch.arange(size) * 0.13 + peer + step * 0.2).sin() * 0.1)
                                 .to(dtype) for peer in range(world_size)]
                    expected_grad = torch.stack([g.float() for g in gradients]).mean(0)
                    lr = 0.01 if step < 3 else 0.003
                    optimizers[layer].param_groups[0]["lr"] = lr

                    update_calls = 0

                    def update(params, grads, first, second):
                        nonlocal update_calls
                        assert rank == 0, "Only rank 0 may update weights or Adam moments"
                        update_calls += 1
                        # Delay the updater to exercise shared-buffer reuse.
                        time.sleep(0.001)
                        if native:
                            adam.step_chunk(params, grads, first, second,
                                            dict(adam.defaults, step=step, lr=lr))
                            return
                        first.mul_(0.8).add_(grads, alpha=0.2)
                        second.mul_(0.95).addcmul_(grads, grads, value=0.05)
                        params.mul_(1 - lr * 0.1)
                        denom = (second / (1 - 0.95 ** step)).sqrt().add_(1e-6)
                        params.addcdiv_(first, denom, value=-lr / (1 - 0.8 ** step))

                    for start in range(0, size, dp.chunk_numel):
                        end = min(start + dp.chunk_numel, size)
                        time.sleep(0.001 * ((rank + step + start) % world_size))
                        dp.reduce_chunk(layer, start, end, gradients[rank][start:end], update)
                    expected_calls = (size + dp.chunk_numel - 1) // dp.chunk_numel if rank == 0 else 0
                    assert update_calls == expected_calls
                    reference = references[layer]
                    reference.grad = expected_grad
                    optimizers[layer].step()
                    torch.testing.assert_close(layer_state.params, reference, rtol=1e-6, atol=1e-7)
                    state = optimizers[layer].state[reference]
                    if rank == 0:
                        torch.testing.assert_close(layer_state.exp_avg, state["exp_avg"])
                        torch.testing.assert_close(layer_state.exp_avg_sq, state["exp_avg_sq"])
            directory = dp.directory
            dp.barrier()
            dp.close()
            dist.barrier()
            assert not directory.exists()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    native = "--native" in sys.argv
    if native:
        from optimizer.layer_adam.builder import CPUAdamLoader
        CPUAdamLoader().load()  # Build once before spawning workers.
    for world in (2, 4):
        with tempfile.TemporaryDirectory(prefix="slideformer-dp-test-") as directory:
            mp.spawn(worker, args=(world, str(Path(directory) / "rendezvous"), native), nprocs=world, join=True)
        print(f"PASS: {world} ranks, FP32/BF16, five AdamW steps, rank 0 updates only, partial chunks and empty layers", flush=True)
