# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import gc
import csv
import time
import torch
import argparse
import numpy as np
import psutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.metric import calculate_flops_per_batch
# from utils.log_mem import log_memory_stats
from utils.datasets import DummyDataset, FullLengthDummyDataset
from utils.gpu_monitor import GPUMonitor
from offload_transformer import SlideFormerOffloader

try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
except ModuleNotFoundError:
    AutoLigerKernelForCausalLM = None

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark for SlideFormerOffloader")
    parser.add_argument("--seq_len", type=int, default=1024, help="序列长度")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--use_bf16", action=argparse.BooleanOptionalAction, default=True, help="BF16精度（默认开启）")
    parser.add_argument("--use_liger", action=argparse.BooleanOptionalAction, default=True, help="Use Liger-kernel (enabled by default)")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa"], help="Attention implementation")
    parser.add_argument("--ac_offload_nvme", action="store_true", help="激活值卸载到NVME")
    parser.add_argument("--nvme_offload_fraction", type=float, default=0.0, help="NVME卸载比例")
    parser.add_argument("--offload_dir", type=str, default="./offload_dir", help="卸载目录路径")
    parser.add_argument("--model_path", type=str, default="/home/scc/models/Llama-3.1-8B-Instruct/", help="模型路径或HuggingFace模型ID")
    parser.add_argument("--warm_step", type=int, default=3, help="预热步数")
    parser.add_argument("--test_step", type=int, default=10, help="测试步数")
    parser.add_argument("--full_length_data", action="store_true", help="Use full-length non-padded dummy samples")
    parser.add_argument("--result_file", type=str, default="", help="结果CSV文件路径（可选）")
    return parser.parse_args()

def track_memory_usage():
    """跟踪内存和显存使用"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
    
    cuda_memory_allocated_mb = 0
    cuda_memory_reserved_mb = 0
    if torch.cuda.is_available():
        cuda_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)  # 转换为MB
        cuda_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)  # 转换为MB
        torch.cuda.reset_peak_memory_stats()
        
    return memory_usage_mb, cuda_memory_allocated_mb, cuda_memory_reserved_mb

def append_result_to_csv(result_file, result_dict):
    """将结果追加到CSV文件中"""
    # 检查文件是否存在，如果不存在则创建并写入头部
    file_exists = os.path.isfile(result_file)
    
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Sequence_Length', 'Batch_Size', 'Precision', 'Use_Liger',
                      'AC_Offload_NVME', 'NVME_Offload_Fraction',
                      'Avg_Time', 'Avg_Tokens_Per_Second', 'Avg_TFLOPS', 
                      'Max_Memory_MB', 'Max_CUDA_Memory_Allocated_MB', 'Max_CUDA_Memory_Reserved_MB', 'Avg_GPU_Utilization']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Model': result_dict['model'],
            'Sequence_Length': result_dict['seq_len'],
            'Batch_Size': result_dict['batch_size'],
            'Precision': result_dict['precision'],
            'Use_Liger': result_dict['use_liger'],
            'AC_Offload_NVME': result_dict['ac_offload_nvme'],
            'NVME_Offload_Fraction': result_dict['nvme_offload_fraction'],
            'Avg_Time': f"{result_dict['avg_time']:.4f}",
            'Avg_Tokens_Per_Second': f"{result_dict['avg_tokens_per_sec']:.1f}",
            'Avg_TFLOPS': f"{result_dict['avg_tflops']:.2f}",
            'Max_Memory_MB': f"{result_dict['max_memory_mb']:.1f}",
            'Max_CUDA_Memory_Allocated_MB': f"{result_dict['max_cuda_memory_allocated_mb']:.1f}",
            'Max_CUDA_Memory_Reserved_MB': f"{result_dict['max_cuda_memory_reserved_mb']:.1f}",
            'Avg_GPU_Utilization': f"{result_dict['avg_gpu_utilization']:.1f}"
        })

def benchmark_model(
    model_path,
    max_seq_length=1024,
    batch_size=64,
    warm_steps=3,
    test_steps=10,
    use_bf16=True,
    use_liger=True,
    attn_implementation="flash_attention_2",
    ac_offload_nvme=False,
    nvme_offload_fraction=0.0,
    offload_dir="./offload_dir",
    full_length_data=False,
    result_file=None
):
    # 在开始前强制清理 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()

    # 内存跟踪变量初始化
    max_memory_mb = 0
    max_cuda_memory_allocated_mb = 0
    max_cuda_memory_reserved_mb = 0

    # 选择要使用的数据类型
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Use {dtype} for benchmarking.")
    print(f"Model: {model_path}")
    print(f"Use Liger Kernel: {use_liger}")
    print(f"AC NVME Offload: {ac_offload_nvme}")
    print(f"NVME Offload Fraction: {nvme_offload_fraction}")
    print(f"Offload Directory: {offload_dir}")
    print(f"Seq_len: {max_seq_length}, batch_size: {batch_size}")
    print(f"Full-length dummy data: {full_length_data}")
    print(f"Warm-up steps: {warm_steps}, test steps: {test_steps} ")

    os.makedirs(offload_dir, exist_ok=True)

    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 确保在 CPU 上加载基础模型
    if use_liger and AutoLigerKernelForCausalLM is None:
        import warnings
        warnings.warn(
            "liger-kernel is not installed; falling back to standard AutoModelForCausalLM. "
            "Install liger-kernel for better performance, or pass --no-use_liger to suppress this warning."
        )
        use_liger = False
    model_cls = AutoLigerKernelForCausalLM if use_liger else AutoModelForCausalLM
    base_model = model_cls.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # 创建LayerAdam Config
    optimizer_kwargs = {
        "lr": 1e-5,
        "bias_correction": True,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "fp32_optimizer_state": True,
        "num_layer": len(base_model.get_decoder().layers) + 2,  # 包括嵌入层和输出层
        "nvme_offload_fraction": nvme_offload_fraction, # 使用传入的参数值
        "offload_dir": offload_dir,
        "prefetch": True
    }
    
    # 3. 创建 SlideFormerOffloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SlideFormerOffloader(
        model=base_model,
        device=device,
        dtype=dtype,
        ac_offload_nvme=ac_offload_nvme,  # 使用传入的参数值
        offload_dir=offload_dir,
        enable_timing=True,
        enable_memory_stats=False,
        optimizer_kwargs=optimizer_kwargs
    )
    
    # 4. 准备数据集
    dataset_cls = FullLengthDummyDataset if full_length_data else DummyDataset
    dataset = dataset_cls(
        size=(warm_steps + test_steps) * batch_size,
        tokenizer=tokenizer,
        max_length=max_seq_length
    )

    # 5. 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    # 6. 训练模式
    model.train()

    # 计算FLOPS
    total_flops = calculate_flops_per_batch(base_model, batch_size, max_seq_length)
    tokens_per_batch = batch_size * max_seq_length

    # 统计数据初始化
    tokens_per_sec_list = []
    tflops_list = []
    batch_time_list = []
    
    # GPU Monitor
    gpu_monitor = GPUMonitor(device_index=0, sample_interval=0.1)
    all_step_utilizations = []

    print(f"Bench Begin: {warm_steps} warmup steps and {test_steps} test steps")
    
    # 开始循环
    step = 0
    for batch in dataloader:
        step_start = time.perf_counter()
        gpu_monitor.start()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向和后向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        
        # 释放输出
        del outputs
        
        # 更新内存使用统计
        current_memory_mb, current_cuda_memory_allocated_mb, current_cuda_memory_reserved_mb = track_memory_usage()
        max_memory_mb = max(max_memory_mb, current_memory_mb)
        max_cuda_memory_allocated_mb = max(max_cuda_memory_allocated_mb, current_cuda_memory_allocated_mb)
        max_cuda_memory_reserved_mb = max(max_cuda_memory_reserved_mb, current_cuda_memory_reserved_mb)
        
        # 计算统计数据
        gpu_monitor.stop()
        avg_util = gpu_monitor.get_average_utilization()

        step_time = time.perf_counter() - step_start
        tokens_per_sec = tokens_per_batch / step_time
        tflops = total_flops / (10**12 * step_time)
        
        # 记录测试阶段的数据
        if step >= warm_steps:
            tokens_per_sec_list.append(tokens_per_sec)
            tflops_list.append(tflops)
            batch_time_list.append(step_time)
            all_step_utilizations.append(avg_util)
            
            print(f"Step {step+1}/{warm_steps + test_steps}, Time: {step_time:.4f}s, "
                  f"Speed: {tokens_per_sec:.1f} tokens/s, "
                  f"TFLOPS: {tflops:.2f}, "
                  f"GPU Utilization: {avg_util:.2f}%")
        else:
            print(f"Warmup {step+1}/{warm_steps}")
        
        step += 1
        if step >= warm_steps + test_steps:
            break
    
    # 计算平均值
    avg_tokens_per_sec = np.mean(tokens_per_sec_list)
    avg_tflops = np.mean(tflops_list)
    avg_batch_time = np.mean(batch_time_list)
    avg_gpu_utilization = np.mean(all_step_utilizations)

    # 打印结果摘要
    print("\n========== Benchmark Result ==========")
    print(f"Model: {os.path.basename(model_path.rstrip('/'))}")
    print(f"Seq_len: {max_seq_length}")
    print(f"Batch_size: {batch_size}")
    print(f"Precision: {'BF16' if use_bf16 else 'FP16'}")
    print(f"Use Liger: {use_liger}")
    print(f"AC NVME Offload: {ac_offload_nvme}")
    print(f"NVME Offload Fraction: {nvme_offload_fraction}")
    print(f"Offload Directory: {offload_dir}")
    print(f"Average batch time: {avg_batch_time:.4f} s")
    print(f"Average speed: {avg_tokens_per_sec:.1f} tokens/s")
    print(f"Average TFLOPS: {avg_tflops:.2f}")
    print(f"Max Memory: {max_memory_mb:.1f} MB")
    print(f"Max CUDA Memory Allocated: {max_cuda_memory_allocated_mb:.1f} MB")
    print(f"Max CUDA Memory Reserved: {max_cuda_memory_reserved_mb:.1f} MB")
    print(f"All Step GPU Utilizations: {avg_gpu_utilization}")
    print("===================================\n")
    
    # 准备结果
    result = {
        "model": os.path.basename(model_path.rstrip("/")),
        "seq_len": max_seq_length,
        "batch_size": batch_size,
        "precision": "BF16" if use_bf16 else "FP16",
        "use_liger": "Yes" if use_liger else "No",
        "ac_offload_nvme": "Yes" if ac_offload_nvme else "No",
        "nvme_offload_fraction": nvme_offload_fraction,
        "offload_dir": offload_dir,
        "avg_time": avg_batch_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "avg_tflops": avg_tflops,
        "max_memory_mb": max_memory_mb,
        "max_cuda_memory_allocated_mb": max_cuda_memory_allocated_mb,
        "max_cuda_memory_reserved_mb": max_cuda_memory_reserved_mb,
        "avg_gpu_utilization": avg_gpu_utilization
    }
    
    # 如果提供了结果文件，将结果追加到CSV
    if result_file:
        append_result_to_csv(result_file, result)
        print(f"结果已追加到文件: {result_file}")
    
    # 返回结果，用于脚本收集
    return result    

if __name__ == "__main__":
    args = parse_args()
    benchmark_model(
        model_path=args.model_path,
        max_seq_length=args.seq_len,
        batch_size=args.batch_size,
        warm_steps=args.warm_step,
        test_steps=args.test_step,
        use_bf16=args.use_bf16,
        use_liger=args.use_liger,
        attn_implementation=args.attn_implementation,
        ac_offload_nvme=args.ac_offload_nvme,
        nvme_offload_fraction=args.nvme_offload_fraction,
        offload_dir=args.offload_dir,
        full_length_data=args.full_length_data,
        result_file=args.result_file
    )
