# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import gc
import time
import torch
import argparse
import numpy as np
import psutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.log_mem import log_memory_stats
from offload_transformer import SlideFormerOffloader
from custom_gpt_model import CustomGPTForCausalLM, CustomGPTConfig
# from torch.nn import CrossEntropyLoss

def parse_args():
    parser = argparse.ArgumentParser(description="SlideFormer evaluation on GPT2 model")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=5120, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=80, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=40, help="Number of transformer layers")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--use_bf16", action=argparse.BooleanOptionalAction, default=True, help="Use bf16 precision (enabled by default)")
    parser.add_argument("--ac_offload_nvme", action="store_true", help="Offload activations to NVME")
    parser.add_argument("--nvme_offload_fraction", type=float, default=0.0, help="NVME offload fraction")
    parser.add_argument("--offload_dir", type=str, default="./offload_dir", help="Offload directory path")
    parser.add_argument("--warm_step", type=int, default=1, help="Number of warmup steps")
    parser.add_argument("--test_step", type=int, default=3, help="Number of test steps")
    return parser.parse_args()

def track_memory_usage():
    """Track memory and CUDA memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    
    cuda_memory_allocated_mb = 0
    cuda_memory_reserved_mb = 0
    if torch.cuda.is_available():
        cuda_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
        cuda_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)  # Convert to MB
        torch.cuda.reset_peak_memory_stats()
        
    return memory_usage_mb, cuda_memory_allocated_mb, cuda_memory_reserved_mb

def benchmark_gpt_model(
    hidden_dim=5120,
    num_heads=80,
    num_layers=40,
    max_seq_length=1024,
    batch_size=64,
    vocab_size=50257,
    warm_steps=1,
    test_steps=3,
    use_bf16=True,
    ac_offload_nvme=False,
    nvme_offload_fraction=0.0,
    offload_dir="./offload_dir"
):
    # Force clean GPU memory before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize memory tracking variables
    max_memory_mb = 0
    max_cuda_memory_allocated_mb = 0
    max_cuda_memory_reserved_mb = 0

    # Select data type to use
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using {dtype} for benchmarking.")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of attention heads: {num_heads}")
    print(f"Number of transformer layers: {num_layers}")
    print(f"Activation NVME offload: {ac_offload_nvme}")
    print(f"NVME offload fraction: {nvme_offload_fraction}")
    print(f"Offload directory: {offload_dir}")
    print(f"Sequence length: {max_seq_length}, Batch size: {batch_size}")
    print(f"Warmup steps: {warm_steps}, Test steps: {test_steps} ")
    
    os.makedirs(offload_dir, exist_ok=True)

    # 1. Create GPT configuration and model
    config = CustomGPTConfig(
        vocab_size=vocab_size,
        n_positions=max_seq_length,
        n_embd=hidden_dim,
        n_layer=num_layers,
        n_head=num_heads,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        embd_pdrop=0.1
    )
    
    # Ensure base model is loaded on CPU
    base_model = CustomGPTForCausalLM(config)
    
    # Create LayerAdam configuration
    optimizer_kwargs = {
        "lr": 1e-5,
        "bias_correction": True,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "fp32_optimizer_state": True,
        "num_layer": len(base_model.get_decoder().layers) + 2,  # Include embedding and output layers
        "nvme_offload_fraction": nvme_offload_fraction, 
        "offload_dir": offload_dir,
        "prefetch": True
    }
    
    # 2. Create SlideFormerOffloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SlideFormerOffloader(
        model=base_model,
        device=device,
        dtype=dtype,
        ac_offload_nvme=ac_offload_nvme,
        offload_dir=offload_dir,
        enable_timing=True,
        enable_memory_stats=True,
        optimizer_kwargs=optimizer_kwargs
    )
    
    # model.train()

    # Calculate FLOPS
    seq_len = max_seq_length
    h = hidden_dim
    L = num_layers
    b = batch_size
    
    # Estimated floating point operations per token
    flops_per_token = 6 * L * h * h  # Simplified estimation
    total_flops = flops_per_token * seq_len * b
    
    tokens_per_batch = batch_size * max_seq_length

    # Initialize statistics
    tokens_per_sec_list = []
    tflops_list = []
    batch_time_list = []
    
    # Generate random input data, similar to LoHan approach
    input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    # attention_mask = torch.ones_like(input_ids, device=device)

    print(f"Starting benchmark: {warm_steps} warmup steps and {test_steps} test steps")
    # running_loss=0
    # Start loop
    total_steps = warm_steps + test_steps
    for step in range(total_steps):
        step_start = time.perf_counter()
        
        # Forward and backward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=None,
            labels=labels
        )

        loss = outputs.loss
        
        # Calculate performance metrics
        # running_loss += loss.item()
        
        # Free outputs
        del outputs
        
        # Update memory usage statistics
        current_memory_mb, current_cuda_memory_allocated_mb, current_cuda_memory_reserved_mb = track_memory_usage()
        max_memory_mb = max(max_memory_mb, current_memory_mb)
        max_cuda_memory_allocated_mb = max(max_cuda_memory_allocated_mb, current_cuda_memory_allocated_mb)
        max_cuda_memory_reserved_mb = max(max_cuda_memory_reserved_mb, current_cuda_memory_reserved_mb)
        
        # Calculate statistics
        step_time = time.perf_counter() - step_start
        tokens_per_sec = tokens_per_batch / step_time
        tflops = total_flops / (10**12 * step_time)
        
        # Record test phase data
        if step >= warm_steps:
            tokens_per_sec_list.append(tokens_per_sec)
            tflops_list.append(tflops)
            batch_time_list.append(step_time)
            
            print(f"Step {step+1}/{total_steps}, Loss: {loss:.4f}, Time: {step_time:.4f}s, "
                  f"Speed: {tokens_per_sec:.1f} tokens/s, "
                  f"TFLOPS: {tflops:.2f}")
        else:
            print(f"Warmup {step+1}/{warm_steps}")
    
    # Calculate averages
    avg_tokens_per_sec = np.mean(tokens_per_sec_list)
    avg_tflops = np.mean(tflops_list)
    avg_batch_time = np.mean(batch_time_list)
    
    # Print results summary
    print("\n" + "="*50)
    print("Performance Statistics Summary:")
    print("="*50)
    
    # Print configuration information
    print(f"Model Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {max_seq_length}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Number of attention heads: {num_heads}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Precision: {'BF16' if use_bf16 else 'FP16'}")
    print(f"  - Activation NVME offload: {ac_offload_nvme}")
    print(f"  - NVME offload fraction: {nvme_offload_fraction}")
    
    # Print performance results
    print(f"\nAverage performance over steps {test_steps}:")
    print(f"  - Average batch time: {avg_batch_time:.4f} s")
    print(f"  - Average speed: {avg_tokens_per_sec:.1f} tokens/s")
    print(f"  - Average TFLOPS: {avg_tflops:.2f}")
    
    # Memory usage statistics
    print(f"\nMemory Usage:")
    print(f"  - Max CPU memory: {max_memory_mb / 1024:.2f} GB")
    print(f"  - Max CUDA memory allocated: {max_cuda_memory_allocated_mb:.2f} MB")
    print(f"  - Max CUDA memory reserved: {max_cuda_memory_reserved_mb:.2f} MB")
    print("="*50)
    
    # Return results for script collection
    return {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "seq_len": max_seq_length,
        "batch_size": batch_size,
        "precision": "BF16" if use_bf16 else "FP16",
        "ac_offload_nvme": ac_offload_nvme,
        "nvme_offload_fraction": nvme_offload_fraction,
        "avg_time": avg_batch_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "avg_tflops": avg_tflops,
        "max_memory_mb": max_memory_mb,
        "max_cuda_memory_allocated_mb": max_cuda_memory_allocated_mb,
        "max_cuda_memory_reserved_mb": max_cuda_memory_reserved_mb
    }

if __name__ == "__main__":
    args = parse_args()
    benchmark_gpt_model(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_length=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warm_steps=args.warm_step,
        test_steps=args.test_step,
        use_bf16=args.use_bf16,
        ac_offload_nvme=args.ac_offload_nvme,
        nvme_offload_fraction=args.nvme_offload_fraction,
        offload_dir=args.offload_dir
    )
