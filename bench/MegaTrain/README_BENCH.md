# MegaTrain Baseline (SlideFormer bench)

[MegaTrain](https://github.com/DLYuanGod/MegaTrain) is a RAM-centric single-GPU
training engine: model parameters live in host (CPU) memory, transformer layers
are streamed to the GPU for compute and evicted immediately afterwards, and the
optimizer state stays on CPU (DeepSpeed CPUAdam). This makes it a direct
single-GPU comparison point for SlideFormer:

| | SlideFormer | MegaTrain |
|---|---|---|
| Parameters | FP32 CPU master, materialized as BF16 on GPU | BF16 CPU parameters, streamed to GPU per layer |
| Optimizer state | FP32 on CPU (LayerAdam) | FP32 on CPU (DeepSpeed CPUAdam) |
| Activations | per-layer full checkpoint | per-interval checkpoint (`checkpoint_interval`) |
| NVMe | optional (off by default) | not used here |
| GPUs | single | single (multi-GPU DP also available upstream) |

MegaTrain's evaluated BF16 path does not keep a separate FP32 master-parameter
copy. Its CPU-memory measurements therefore do not have the same
mixed-precision storage semantics as SlideFormer and are reported for reference.

## Fair-comparison alignment

`mt_bench.py` is deliberately aligned with `bench/DeepSpeed/ds_bench.py` on the
three axes that would otherwise skew the comparison:

- **Liger kernel** — on by default (`--use_liger`, toggle off with `--no_liger`),
  loaded via `AutoLigerKernelForCausalLM`, exactly like the DeepSpeed baseline.
  liger 0.6.5 fuses RMSNorm/SwiGLU/RoPE/CE for llama, qwen2 and qwen3.
- **Dataset** — the shared `utils.datasets.DummyDataset` (random fixed-length
  tokens), identical to what ds_bench feeds, so every framework sees the same
  token distribution and sequence length. (Loss will sit near `ln(vocab)` since
  the data is random — expected, and matches the other baselines.)
- **TFLOPS** — `utils.metric.calculate_flops_per_batch` (Megatron-LM formula with
  GQA + SwiGLU), the same estimator the other baselines use. This is a hard
  dependency, not a fallback, so the TFLOPS column is comparable across all
  frameworks. (It reads the HF model config, which MegaTrain keeps internally as
  `_model_config`; the driver passes the captured HF config to the estimator.)

## Source version

This tree is the upstream MegaTrain `main` (`infinity.__version__ == 0.3.0`),
copied verbatim **except**:

- `benchmark/` and `examples/sft/train_benchmark.py` were dropped — those are
  SSDP-specific harness files (they import SSDP's `experiment_runner` /
  `_slim_metric`). We use `mt_bench.py` (below) instead.

The benchmark driver added for SlideFormer:

- `mt_bench.py` — wraps MegaTrain's `CPUMasterModel` training loop with the same
  warm-up / test-step timing and CSV columns as `bench/DeepSpeed/ds_bench.py`,
  so its `Avg_Time / Tokens_Per_Second / TFLOPS / Max_*_Memory` line up
  column-for-column with the DeepSpeed and ColossalAI baselines (it adds one
  leading `Framework` column).
- `mt_bench.sh` — runner mirroring `ds_bench.sh` (model list, batch sizes,
  seq len, warm/test steps). Builds the CUDA extensions on first run.

## Environment

MegaTrain's `requirements.txt` declares `transformers>=5.0.0`, but that is only
a soft install pin — the code uses generic `AutoModelForCausalLM` / `AutoConfig`
APIs and imports without any runtime version assertion.

**Verified on SlideFormer's `new` env** (transformers 4.51.3, torch 2.5.1+cu124,
liger 0.6.5): both CUDA extensions compile and a Qwen3-1.7B run trains
end-to-end (params on CPU, GPU peak ~2.9 GB, loss decreasing). So the `new` env
is fine for benchmarking Qwen2.5 / Qwen3 / Llama models.

Caveat: MegaTrain's newest model configs (e.g. Qwen3.5-27B, GLM-4.6V) may need
the model class from transformers 5.x. For those, use an env with
`transformers>=5.0` (e.g. SSDP's `ssdp` env), which is also what MegaTrain's
authors validate against.

## Build (one-time, per env)

The runner builds these automatically if missing, or do it manually:

```bash
# 1) memory ops extension (pinned-pool, async H2D/D2H, CUDA events)
cd csrc && pip install . --no-build-isolation && cd ..

# 2) batched-copy pipeline extension
python setup.py build_ext --inplace
```

Requires a CUDA toolkit matching your torch build (cu124 here) on `PATH`/`CUDA_HOME`.

## Run

```bash
# edit model list / batch sizes at the top of mt_bench.sh, then:
./mt_bench.sh
```

Or a single configuration directly:

```bash
python mt_bench.py \
    --model_path /home/scc/models/Qwen3-1.7B \
    --seq_len 1024 --batch_size 8 --use_bf16 \
    --warm_step 2 --test_step 3 \
    --result_file mt.csv
```

Results append to a CSV with the same schema as the other baselines (plus a
`Framework` column), so all three can be concatenated for comparison.
