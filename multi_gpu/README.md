# SlideFormer multi-GPU extension

This directory provides a reference multi-GPU extension of
[SlideFormer](https://github.com/RegiaYoung/SlideFormer) for single-node
synchronous data-parallel training. It preserves SlideFormer's layer-streaming
execution with replicated parameter delivery and CPU-side gradient reduction.

The optimized [SlideDP](https://github.com/RegiaYoung/SlideDP) runtime described
in the paper is not included in this release.

## Execution

Each GPU loads the complete current layer through SlideFormer's existing
transfer path (`full`). FP32 weights have one shared-memory backing store.
Ranks place their local gradients in a bounded shared-memory buffer; rank 0
averages each chunk in FP32 and invokes the existing CPU Adam kernel
(`cpu_reduce`). Gloo barriers publish gradients and completed updates;
gradient payloads are read from shared memory.

Rank 0 alone stores the Adam moments and updates all weights. Other ranks
allocate no Adam moments and read the shared weights after updates complete.
Rank 0 also initializes parameters and saves model weights.

## Installation

Run the commands below from this directory. Use Linux, Python 3.12, a
CUDA-enabled PyTorch installation, and a C++ compiler with OpenMP support for
the CPU Adam extension. The functional checks used PyTorch 2.11.0 with CUDA
13.0 and H100 GPUs. Install the remaining tested dependencies with:

```bash
pip install -r requirements.txt
```

The vendored loss uses the Liger 0.7.0 kernel interface. Liger 0.8.2 has a
changed interface and is incompatible with this entry point.

For newer models such as Qwen3.6 and Gemma 4, we recommend using the updated
dependencies in [`requirements-models.txt`](requirements-models.txt):

```bash
pip install packaging ninja==1.13.0
pip install --no-build-isolation -r requirements-models.txt
```

We recommend `--attn-implementation sdpa` for these models.

## Quick start

Run a tiny synthetic smoke check:

```bash
torchrun --standalone --nproc_per_node=2 scripts/main_dp.py \
  --tiny --steps 3 --chunk-numel 4096 --cpu-threads 2
```

For training, prepare a `torch.save` file containing a two-dimensional `int64`
tensor of packed, equal-length token sequences. Every token is a training
target after the causal shift; padding/masked targets are not supported by
this minimal entry point. Data is divided using `DistributedSampler`, and
incomplete batches are dropped. `--batch-size` is per rank.

```bash
torchrun --standalone --nproc_per_node=4 scripts/main_dp.py \
  --model /path/to/checkpoint --tokens /path/to/packed_tokens.pt \
  --batch-size 1 --steps 100 --cpu-threads 4 --save-model ./trained-model
```

Use a NUMA-local GPU group with matching CPU/memory placement on multi-socket
hosts. Budget CPU threads across ranks. `/dev/shm` must fit the FP32 weights
plus the shared gradient buffer (`world_size * chunk_numel * 2` bytes for
BF16). One full copy of the Adam moments resides on rank 0; activations,
conversion buffers and local gradients remain per rank. Model loading may
temporarily hold a model copy per process before shared views replace it.

## Scope and integration

This version supports single-node BF16 training, CPU-resident Adam states,
one optimizer step per backward, and the decoder layout supported by the
single-GPU offloader. It uses two barriers per gradient chunk. The data-parallel
entry point provides full parameter transfers and SHM CPU reduction;
NVMe offload and optimizer checkpoint/resume are unsupported in this mode.

Applications can pass a collectively constructed `SharedDataParallel` as
`shared_dp` to `SlideFormerOffloader`. Select the local CUDA device before
importing the offloader. All ranks must process the same layer/chunk sequence
and call `wait_for_completion()` after each backward; use equal effective
loss denominators across ranks, or scale local losses before averaging.
All ranks must call `save_pretrained()` together. It saves model weights
without detaching the live shared storage. Drain work before closing the
shared context.

## Checks

CPU-only numerical checks compare five AdamW steps against PyTorch with two
and four ranks, FP32/BF16 gradients, partial chunks, and empty layers. They
also verify that only rank 0 invokes the update and holds Adam moments:

```bash
python tests/test_shared_data_parallel.py
python tests/test_shared_data_parallel.py --native
```

GPU smoke checks cover one, two and four ranks, tied embeddings, double
buffering, and model saving/reloading. These are functional checks, not
performance benchmarks.

## Source and license

This version extends the published SlideFormer revision
`062322fbb1b1f078b4a25dae60640914d0521e0a`.
`SHA256SUMS` lists the checksums of the released files.
See `LICENSE`, `NOTICE`, and `LICENSES/` for attribution and licensing.
