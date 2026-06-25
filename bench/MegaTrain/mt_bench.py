#!/usr/bin/env python
"""MegaTrain baseline benchmark driver, aligned with SlideFormer's bench harness.

MegaTrain (https://github.com/DLYuanGod/MegaTrain) is a RAM-centric single-GPU
training engine: parameters live in host (CPU) memory, layers are streamed to
the GPU for compute and evicted afterwards, and the optimizer state stays on CPU
(DeepSpeed CPUAdam). This makes it directly comparable to SlideFormer's offload
behavior (param + optimizer on CPU, per-layer full activation checkpoint).

This script wraps MegaTrain's CPUMasterModel training loop with the same
warm-up / test-step timing and CSV columns used by ds_bench.py and the
ColossalAI benchmark, so the resulting numbers line up column-for-column with
the other baselines.

Run via mt_bench.sh, or directly, e.g.:

    PYTHONPATH=. python mt_bench.py \
        --model_path /home/scc/models/Qwen3-1.7B \
        --seq_len 1024 --batch_size 8 --use_bf16 \
        --warm_step 2 --test_step 3 --result_file mt.csv
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch
import psutil

# --- import paths -----------------------------------------------------------
# bench/MegaTrain  -> MegaTrain source (infinity package, examples)
# repo root        -> SlideFormer utils.metric (FLOPS parity with other baselines)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig

# Liger kernel — same acceleration the DeepSpeed baseline enables by default, so
# the model-internal compute (RMSNorm / SwiGLU / RoPE / fused CE) is on equal
# footing across frameworks.
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# Shared dummy dataset — identical random fixed-length samples to the DeepSpeed
# baseline (utils.datasets.DummyDataset), so all frameworks train on the same
# token distribution and sequence length.
from utils.datasets import DummyDataset

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    CPU_ADAM_AVAILABLE = True
except Exception:
    CPU_ADAM_AVAILABLE = False

# Use SlideFormer's FLOPS estimator so the TFLOPS column matches the other
# baselines exactly. This is a required dependency (not optional) — a different
# FLOPS formula would make the TFLOPS column incomparable across frameworks.
from utils.metric import calculate_flops_per_batch


def parse_args():
    parser = argparse.ArgumentParser(description="MegaTrain baseline benchmark")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warm_step", type=int, default=2)
    parser.add_argument("--test_step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--use_bf16", action="store_true",
                        help="bf16 (matches SlideFormer); otherwise fp16")
    parser.add_argument("--use_liger", action="store_true", default=True,
                        help="Use Liger-kernel (default on, matches ds_bench)")
    parser.add_argument("--no_liger", dest="use_liger", action="store_false",
                        help="Disable Liger-kernel")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--attn_implementation", type=str,
                        default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa"])
    # MegaTrain memory knobs (its equivalent of activation checkpointing).
    parser.add_argument("--checkpoint_interval", type=int, default=4,
                        help="MegaTrain activation checkpoint interval (layers)")
    parser.add_argument("--num_grad_slabs", type=int, default=12)
    parser.add_argument("--result_file", type=str, default="")
    return parser.parse_args()


def append_result_to_csv(result_file, row):
    """Write one row using the same columns as ds_bench.py (+ a Framework col)."""
    fieldnames = [
        'Framework', 'Model', 'Sequence_Length', 'Batch_Size', 'Precision',
        'Use_Liger', 'Zero_Stage', 'Offload', 'Offload_NVMe',
        'Avg_Time', 'Avg_Tokens_Per_Second', 'Avg_TFLOPS',
        'Max_Memory_MB', 'Max_CUDA_Memory_Allocated_MB',
        'Max_CUDA_Memory_Reserved_MB',
    ]
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    precision = "bf16" if args.use_bf16 else "fp16"
    device = args.device
    model_name = os.path.basename(args.model_path.rstrip("/"))

    print(f"===== MegaTrain bench: {model_name} | seq={args.seq_len} "
          f"bs={args.batch_size} {precision} | liger={'on' if args.use_liger else 'off'} =====")
    print(f"Warm-up steps: {args.warm_step}, test steps: {args.test_step}")
    if not CPU_ADAM_AVAILABLE:
        print("WARNING: DeepSpeedCPUAdam unavailable, falling back to torch AdamW "
              "(CPU). Optimizer placement still on CPU, but not SIMD-accelerated.")

    # --- tokenizer + model (CPU-resident params, GPU is transient compute) ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Match the DeepSpeed baseline: load through the Liger kernel loader (fused
    # RMSNorm/SwiGLU/RoPE/CE) by default, so MegaTrain's per-layer GPU compute is
    # accelerated the same way before being streamed back to CPU.
    load_cls = AutoLigerKernelForCausalLM if args.use_liger else AutoModelForCausalLM
    hf_model = load_cls.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cpu",          # keep on CPU; MegaTrain manages GPU streaming
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

    # MegaTrain config: params on CPU, per-interval activation checkpoint,
    # optimizer state on CPU — the offload setup compared against SlideFormer.
    # A dummy dataset_path keeps CPUMasterConfig happy (we feed batches manually).
    config = CPUMasterConfig(
        model_name=args.model_path,
        dtype=dtype,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        num_steps=args.warm_step + args.test_step,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_interval=args.checkpoint_interval,
        num_grad_slabs=args.num_grad_slabs,
        attn_implementation=args.attn_implementation,
        dataset_path="__dummy__",
    )

    model = CPUMasterModel(hf_model, config)
    # Capture the underlying HF config for the FLOPS estimator before dropping
    # the HF model. CPUMasterModel.config is the CPUMasterConfig (no hidden_size),
    # so we must use the real transformer config here.
    hf_config = hf_model.config
    del hf_model

    if CPU_ADAM_AVAILABLE:
        optimizer = DeepSpeedCPUAdam(
            model.get_parameters(), lr=args.lr,
            betas=(0.9, 0.999), eps=1e-8,
            weight_decay=args.weight_decay, adamw_mode=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.get_parameters(), lr=args.lr,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay,
        )

    # --- dataset (shared DummyDataset, identical to the DeepSpeed baseline) --
    from torch.utils.data import DataLoader
    dataset = DummyDataset(
        size=(args.warm_step + args.test_step) * args.batch_size,
        tokenizer=tokenizer,
        max_length=args.seq_len,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, drop_last=True,
    )
    data_iter = iter(dataloader)

    # --- FLOPS-per-batch (parity with other baselines) ----------------------
    # calculate_flops_per_batch reads obj.config.<hf fields>; wrap the captured
    # HF config so the formula matches DeepSpeed/ColossalAI exactly.
    class _FlopsModelShim:
        def __init__(self, cfg):
            self.config = cfg
    total_flops = calculate_flops_per_batch(
        _FlopsModelShim(hf_config), args.batch_size, args.seq_len)
    tokens_per_batch = args.batch_size * args.seq_len

    torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()

    print(f"Bench Begin: {args.warm_step} warmup + {args.test_step} test steps")
    test_times = []
    max_cpu_mem_mb = 0.0
    n_total = args.warm_step + args.test_step

    for step in range(n_total):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step_start = time.perf_counter()
        loss_val, n_tokens, timing = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"],
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)
        optimizer.step()
        model._sync_params_to_gpu()
        model.zero_grad()
        optimizer.zero_grad()
        step_time = time.perf_counter() - step_start

        cpu_mem_mb = process.memory_info().rss / (1024 ** 2)
        max_cpu_mem_mb = max(max_cpu_mem_mb, cpu_mem_mb)

        tag = "warmup" if step < args.warm_step else "test"
        print(f"Step {step+1}/{n_total} ({tag}), Time: {step_time:.4f}s, "
              f"Loss: {loss_val:.6f}")
        if step >= args.warm_step:
            test_times.append(step_time)

    # --- aggregate ----------------------------------------------------------
    avg_time = sum(test_times) / len(test_times) if test_times else 0.0
    avg_tps = tokens_per_batch / avg_time if avg_time > 0 else 0.0
    avg_tflops = (total_flops / 1e12) / avg_time if avg_time > 0 else 0.0
    max_cuda_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    max_cuda_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    print("===== Result =====")
    print(f"Avg time/step: {avg_time:.4f}s | Tokens/s: {avg_tps:.1f} | "
          f"TFLOPS: {avg_tflops:.2f}")
    print(f"Peak CUDA alloc: {max_cuda_alloc_mb:.1f} MB | "
          f"reserved: {max_cuda_reserved_mb:.1f} MB | "
          f"Peak CPU: {max_cpu_mem_mb:.1f} MB")

    if args.result_file:
        append_result_to_csv(args.result_file, {
            'Framework': 'MegaTrain',
            'Model': model_name,
            'Sequence_Length': args.seq_len,
            'Batch_Size': args.batch_size,
            'Precision': precision,
            'Use_Liger': 'Yes' if args.use_liger else 'No',
            'Zero_Stage': 'n/a',          # not a ZeRO framework
            'Offload': 'cpu',             # params + optimizer on CPU
            'Offload_NVMe': 'no',
            'Avg_Time': f"{avg_time:.4f}",
            'Avg_Tokens_Per_Second': f"{avg_tps:.2f}",
            'Avg_TFLOPS': f"{avg_tflops:.2f}",
            'Max_Memory_MB': f"{max_cpu_mem_mb:.2f}",
            'Max_CUDA_Memory_Allocated_MB': f"{max_cuda_alloc_mb:.2f}",
            'Max_CUDA_Memory_Reserved_MB': f"{max_cuda_reserved_mb:.2f}",
        })
        print(f"Result appended to {args.result_file}")


if __name__ == "__main__":
    main()
