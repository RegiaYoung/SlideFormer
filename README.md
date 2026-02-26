# SlideFormer

**Paper (DAC 2026): _An Efficient Heterogeneous Co-Design for Fine-Tuning on a Single GPU_**

**SlideFormer** is a single-GPU LLM fine-tuning system that treats the entire heterogeneous platform (GPU + CPU RAM + NVMe) as a unified memory hierarchy. It enables fine-tuning of **100B+ models on a single RTX 4090** via three co-designed innovations:

- **Lightweight Asynchronous Engine**: maintains a small active window on the GPU and overlaps GPU compute with CPU optimizer updates and hierarchical I/O.
- **Heterogeneous Memory Management**: pre-allocated GPU cache units and shared host-side gradient buffers reduce peak CPU/GPU memory by ~25–50%.
- **Advanced I/O & Triton Kernels**: NVMe offload via io_uring/GPUDirect Storage, plus fused Triton kernels that resolve bottlenecks overlooked by prior systems.

Compared to existing frameworks: **1.40×–6.27× throughput**, ~50% GPU memory reduction, ~40% CPU memory reduction, 8× larger batch sizes, >95% peak GPU utilization on NVIDIA and AMD.

## Repository layout

```
SlideFormer/
├── offload_transformer.py      # End-to-end SlideFormer engine
├── transformer_layer.py        # Async layer wrappers
├── sliding_checkpoint.py       # CPU / NVMe activation offload implementation
├── optimizer/layer_adam/       # LayerAdam optimizer (AVX-512/OpenMP C++ kernels)
├── utils/                      # Metrics, dataset helpers, GPU monitor, timers
│
├── main_dummy.py               # Default entry – sanity-check
├── main_bench.py               # Benchmark runner
├── bench.sh                    # Sweep script calling main_bench.py
├── main_profile.py             # torch.profiler friendly
├── main_real.py                # Real fine-tuning example on MathFusionQA
├── main_gpt.py                 # Comparison using GPT-2 model
├── custom_gpt_model.py         # GPT-2 model to match prior-work
│
└── bench/                      # Baselines & unit experiments
    ├── DeepSpeed/              # ZeRO-Offload baseline
    ├── ColossalAI/             # Gemini baseline
    └── cpu_adam_profile/       # CPU Adam microbenchmark
```

## Setup

```bash
conda env create -f environment.yml
conda activate slideformer
```

Key dependencies: `torch`, `transformers`, `flash-attn`, `liger-kernel`, `tensornvme`.  
`flash-attn`, `liger-kernel`, and `tensornvme` are **optional** — the code falls back gracefully when they are not installed.

For NVMe offload, set `--offload_dir` to a path on a fast local SSD.

## Supported models

- LLaMA family (1B–70B)
- Qwen2.5 family
- Mistral family
- Other HuggingFace Transformers decoder-only models

## Quickstart — `main_dummy.py`

`main_dummy.py` is the recommended first run. It trains on a `DummyDataset` (no real data needed) to verify the full pipeline works.

```bash
python main_dummy.py
```

On a **multi-NUMA server**, binding to the GPU's NUMA node meaningfully improves CPU–GPU bandwidth:

```bash
numactl --cpunodebind=0 --membind=0 python main_dummy.py
```

To find which NUMA node your GPU is on: `nvidia-smi topo -m`

**Common options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | Llama-3.1-8B-Instruct | Local path or HuggingFace model ID |
| `--seq_len` | 1024 | Sequence length |
| `--batch_size` | 64 | Training batch size |
| `--attn_implementation` | `flash_attention_2` | `flash_attention_2` or `sdpa` |
| `--no-use_liger` | — | Disable Liger-kernel (auto-used when installed) |
| `--ac_offload_nvme` | off | Offload activations to NVMe (default: CPU offload) |
| `--offload_dir` | `./offload_dir` | Directory for NVMe / activation offload files |
| `--no-use_bf16` | — | Disable BF16 (fall back to FP16) |

## Benchmarking — `main_bench.py` + `bench.sh`

Single run with CSV output:

```bash
python main_bench.py \
  --model_path /home/scc/models/Llama-3.1-8B-Instruct/ \
  --seq_len 1024 --batch_size 64 \
  --warm_step 3 --test_step 10 \
  --result_file ./outputs/bench.csv
```

Sweep over multiple models and batch sizes (edit `MODEL_PATHS` / `BATCH_SIZES` in the script):

```bash
bash bench.sh
```

Results are written to `./outputs/` (gitignored). Metrics logged per step: tokens/s, TFLOPS, peak CPU RAM, peak VRAM, and GPU utilization.

## Profiling — `main_profile.py`

Generates a **Chrome trace** and **CUDA memory timeline** via `torch.profiler`:

```bash
python main_profile.py
```

Traces are saved to `./profile/` (gitignored). For full `nsys` profiling:

```bash
nsys profile -o ./outputs/nsys_trace python main_dummy.py
```

## Real fine-tuning — `main_real.py`

End-to-end fine-tuning on **MathFusionQA** (auto-downloaded from HuggingFace Hub):

```bash
python main_real.py \
  --model_path /home/scc/models/Llama-3.1-8B-Instruct/ \
  --seq_len 4096 --batch_size 16 --epochs 1 \
  --output_dir ./mathfusion-ft-results
```

The script saves the trained model, tokenizer, loss curve (PDF), and loss data (CSV) to `--output_dir`.

## Fair comparison — `main_gpt.py`

Uses a **custom GPT-style model** (`custom_gpt_model.py`) so that layer count, hidden size, and head count can exactly match configurations reported in prior work, enabling apple-to-apple throughput comparisons:

```bash
python main_gpt.py \
  --num_layers 40 --hidden_dim 5120 --num_heads 80 \
  --seq_len 1024 --batch_size 32
```

## Baselines & unit experiments — `bench/`

| Directory | Baseline | How to run |
|-----------|----------|-----------|
| `bench/DeepSpeed/` | ZeRO-Offload (DeepSpeed) | `bash bench/DeepSpeed/ds_bench.sh` |
| `bench/ColossalAI/` | Gemini (ColossalAI) | `bash bench/ColossalAI/bench_colossalai.sh` |
| `bench/cpu_adam_profile/` | CPU Adam microbench | `bash bench/cpu_adam_profile/run_full_benchmark.sh` |

Unit experiments in `bench/`: `peak_tflops_test.py`, `bench_layer_transfer.py`, `test_kvikio_async.py`, `gsm8k_eval.py` (accuracy evaluation), and `find_batch_size_overlap_point.py`.

All output files (CSVs, traces, figures) are gitignored by default.

## License

Apache-2.0
