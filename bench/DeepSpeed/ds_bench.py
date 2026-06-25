import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import gc
import csv
import time
import torch
import argparse
import deepspeed
import numpy as np
import json
import psutil
import re # 导入 re 模块用于正则表达式
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.metric import calculate_flops_per_batch
from utils.log_mem import log_memory_stats
from utils.datasets import DummyDataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Benchmark")
    parser.add_argument("--seq_len", type=int, default=1024, help="序列长度")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--use_bf16", action="store_true", help="使用bf16精度")
    parser.add_argument("--use_liger", action="store_true", help="使用Liger-kernel")
    parser.add_argument("--model_path", type=str, default="/home/scc/models/Llama-3.1-8B-Instruct/", help="模型路径")
    parser.add_argument("--warm_step", type=int, default=3, help="预热步数")
    parser.add_argument("--test_step", type=int, default=10, help="测试步数")
    parser.add_argument("--result_file", type=str, default="", help="结果CSV文件路径")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    # DeepSpeed特定参数
    parser.add_argument("--zero_stage", type=int, default=1, choices=[0, 1, 2, 3], help="DeepSpeed ZeRO阶段")
    parser.add_argument("--offload", action="store_true", help="启用ZeRO-Offload")
    parser.add_argument("--offload_nvme", action="store_true", help="启用ZeRO-Infinity (NVMe offload)")
    parser.add_argument("--nvme_path", type=str, default="/RAID0", help="NVMe存储路径")
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

def get_model_size_gb(model_path):
    """从模型路径中提取模型大小（单位：Billion parameters）"""
    match = re.search(r'(\d+(\.\d+)?)[bB]', model_path)
    if match:
        return float(match.group(1))
    print(f"警告: 无法从路径 '{model_path}' 中提取模型大小。")
    return None # 返回 None 表示无法确定大小

def create_ds_config(args):
    """根据命令行参数创建DeepSpeed配置"""
    config = {
        "train_batch_size": args.batch_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "steps_per_print": 10,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "fp16": {
            "enabled": not args.use_bf16,
        },
        "bf16": {
            "enabled": args.use_bf16,
        },
        "zero_optimization": {
            "stage": args.zero_stage,
            # 增加零优化内存效率的参数
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
        }
    }
    
    # 注意: 不在此处配置 DeepSpeed 的 activation_checkpointing。
    # DeepSpeed 的 activation_checkpointing 只有在模型代码显式调用
    # deepspeed.checkpointing.checkpoint() 时才生效, 而本 benchmark 走的是
    # HF 的 model.gradient_checkpointing_enable() (见下方模型加载处),
    # 与 SlideFormer 的逐层 full activation checkpoint 对齐。

    # 设置ZeRO-Offload (CPU)
    if args.offload:
        if args.zero_stage < 2:
            print("警告: ZeRO-Offload需要至少ZeRO-2阶段，自动设置为ZeRO-2")
            config["zero_optimization"]["stage"] = 2
        
        # CPU offload参数
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": True
        }
        
        # 对于stage 3，启用完全参数分片和offload
        if args.zero_stage >= 3:
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }
            
            # 优化ZeRO-3的内存使用
            # 5e7 (而非 1e8/5e8): 限制 GPU 常驻参数与预取缓冲, 强制更激进 offload,
            # 这是在 24GB 卡上跑 8B + bs32 (无梯度累积) 不 OOM 的关键, 与 OOM 仓库对齐。
            config["zero_optimization"]["stage3_param_persistence_threshold"] = 0
            config["zero_optimization"]["stage3_max_live_parameters"] = 5e7
            config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e7

    # 设置ZeRO-Infinity (NVMe)
    if args.offload_nvme:
        if args.zero_stage < 3:
            print("警告: ZeRO-Infinity需要ZeRO-3阶段，自动设置为ZeRO-3")
            config["zero_optimization"]["stage"] = 3
            
        # --- 动态设置 buffer_size ---
        model_size_gb = get_model_size_gb(args.model_path)
        if model_size_gb is not None:
            if model_size_gb <= 3:
                nvme_buffer_size = 4e8 # 3B 及以下
                print(f"检测到模型大小 <= 3B，设置 NVMe buffer_size 为 {nvme_buffer_size}")
            elif model_size_gb <= 7:
                nvme_buffer_size = 6e8 # 7B
                print(f"检测到模型大小 > 3B 且 <= 7B，设置 NVMe buffer_size 为 {nvme_buffer_size}")
            elif model_size_gb <= 14:
                nvme_buffer_size = 8e8 # 14B (估算)
                print(f"检测到模型大小 > 7B 且 <= 14B，设置 NVMe buffer_size 为 {nvme_buffer_size} (估算)")
            elif model_size_gb <= 32:
                nvme_buffer_size = 1e9 # 32B (估算)
                print(f"检测到模型大小 > 14B 且 <= 32B，设置 NVMe buffer_size 为 {nvme_buffer_size} (估算)")
            else: # 70B, 72B 及更大
                nvme_buffer_size = 1.5e9 # (估算)
                print(f"检测到模型大小 > 32B，设置 NVMe buffer_size 为 {nvme_buffer_size} (估算)")
        else:
            nvme_buffer_size = 5e8 # 默认值
            print(f"警告: 无法确定模型大小，使用默认 NVMe buffer_size: {nvme_buffer_size}")
        # --- 动态设置结束 ---

        # 配置优化器NVMe offload
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "nvme",
            "pin_memory": True,
            "nvme_path": args.nvme_path,
            "buffer_count": 8,
            "fast_init": False
        }

        # 配置参数NVMe offload - 4090优化版
        config["zero_optimization"]["offload_param"] = {
            "device": "nvme",
            "pin_memory": True,
            "nvme_path": args.nvme_path,
            "buffer_count": 32,
            "buffer_size": nvme_buffer_size, # 使用动态计算的值
        }

        # 添加aio配置以提高NVMe性能
        config["aio"] = {
            "block_size": 1048576,  # 1MB
            "queue_depth": 8,
            "thread_count": 4,
            "single_submit": False,
            "overlap_events": True
        }

        # 优化ZeRO-3的内存使用 (NVMe Offload)
        # 更积极地换出参数以节省内存
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 0
        config["zero_optimization"]["stage3_max_live_parameters"] = 5e7 # 显著减少活跃参数数量，强制更多参数卸载到NVMe
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = "auto"


    return config

def append_result_to_csv(result_file, result_dict):
    """将结果追加到CSV文件中"""
    # 检查文件是否存在，如果不存在则创建并写入头部
    file_exists = os.path.isfile(result_file)
    
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Sequence_Length', 'Batch_Size', 'Precision', 'Use_Liger',
                      'Zero_Stage', 'Offload', 'Offload_NVMe',
                      'Avg_Time', 'Avg_Tokens_Per_Second', 'Avg_TFLOPS',
                      'Max_Memory_MB', 'Max_CUDA_Memory_Allocated_MB', 'Max_CUDA_Memory_Reserved_MB']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Model': result_dict['model'],
            'Sequence_Length': result_dict['seq_len'],
            'Batch_Size': result_dict['batch_size'],
            'Precision': result_dict['precision'],
            'Use_Liger': result_dict['use_liger'],
            'Zero_Stage': result_dict['zero_stage'],
            'Offload': result_dict['offload'],
            'Offload_NVMe': result_dict['offload_nvme'],
            'Avg_Time': f"{result_dict['avg_time']:.4f}",
            'Avg_Tokens_Per_Second': f"{result_dict['avg_tokens_per_sec']:.1f}",
            'Avg_TFLOPS': f"{result_dict['avg_tflops']:.2f}",
            'Max_Memory_MB': f"{result_dict['max_memory_mb']:.1f}",
            'Max_CUDA_Memory_Allocated_MB': f"{result_dict['max_cuda_memory_allocated_mb']:.1f}",
            'Max_CUDA_Memory_Reserved_MB': f"{result_dict['max_cuda_memory_reserved_mb']:.1f}"
        })

def benchmark_model_ds(args):
    # 在开始前强制清理 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()

    # 内存跟踪变量初始化
    max_memory_mb = 0
    max_cuda_memory_allocated_mb = 0
    max_cuda_memory_reserved_mb = 0

    # 选择要使用的数据类型
    dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    print(f"Use {dtype} for benchmarking.")
    print(f"Model: {args.model_path}")
    print(f"Use Liger Kernel: {args.use_liger}")
    print(f"Seq_len: {args.seq_len}, batch_size: {args.batch_size}")
    print(f"DeepSpeed设置: ZeRO-{args.zero_stage}, Offload: {args.offload}, NVMe Offload: {args.offload_nvme}")
    print(f"Warm-up steps: {args.warm_step}, test steps: {args.test_step}")

    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载基础模型
    if args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            use_cache=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            use_cache=False
        )

    # 启用梯度检查点以提高内存效率
    if hasattr(model, "gradient_checkpointing_enable"):
        print("gradient_checkpointing_enable")
        model.gradient_checkpointing_enable()
    
    # 3. 创建DeepSpeed配置
    ds_config = create_ds_config(args)
    # 可选: 保存配置到文件查看
    with open('ds_config_temp.json', 'w') as f:
        json.dump(ds_config, f, indent=4)
    
    # 4. 准备数据集
    dataset = DummyDataset(
        size=(args.warm_step + args.test_step) * args.batch_size,
        tokenizer=tokenizer,
        max_length=args.seq_len
    )

    # 5. 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    # 6. 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    # 训练模式
    model_engine.train()

    # 计算FLOPS
    total_flops = calculate_flops_per_batch(model, args.batch_size, args.seq_len)
    tokens_per_batch = args.batch_size * args.seq_len

    # 统计数据初始化
    tokens_per_sec_list = []
    tflops_list = []
    batch_time_list = []

    print(f"Bench Begin: {args.warm_step} warmup steps and {args.test_step} test steps")
    
    # 开始循环
    step = 0
    for batch in dataloader:
        step_start = time.perf_counter()
        
        input_ids = batch['input_ids'].to(model_engine.device)
        attention_mask = batch['attention_mask'].to(model_engine.device)
        labels = batch['labels'].to(model_engine.device)

        # 前向和后向传播
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        
        print(f"Step {step+1}/{args.warm_step + args.test_step}, Loss: {loss.item():.6f}")
        
        # DeepSpeed后向传播
        model_engine.backward(loss)
        model_engine.step()
        
        # 更新内存统计
        current_memory_mb, current_cuda_memory_allocated_mb, current_cuda_memory_reserved_mb = track_memory_usage()
        max_memory_mb = max(max_memory_mb, current_memory_mb)
        max_cuda_memory_allocated_mb = max(max_cuda_memory_allocated_mb, current_cuda_memory_allocated_mb)
        max_cuda_memory_reserved_mb = max(max_cuda_memory_reserved_mb, current_cuda_memory_reserved_mb)
        
        # 计算统计数据
        step_time = time.perf_counter() - step_start
        tokens_per_sec = tokens_per_batch / step_time
        tflops = total_flops / (10**12 * step_time)
        
        # 记录测试阶段的数据
        if step >= args.warm_step:
            tokens_per_sec_list.append(tokens_per_sec)
            tflops_list.append(tflops)
            batch_time_list.append(step_time)
            
            print(f"Step {step-args.warm_step+1}/{args.test_step}, Time: {step_time:.4f}s, "
                  f"Speed: {tokens_per_sec:.1f} tokens/s, "
                  f"TFLOPS: {tflops:.2f}")
        else:
            print(f"Warmup {step+1}/{args.warm_step}")
        
        step += 1
        if step >= args.warm_step + args.test_step:
            break
    
    # 计算平均值
    avg_tokens_per_sec = np.mean(tokens_per_sec_list)
    avg_tflops = np.mean(tflops_list)
    avg_batch_time = np.mean(batch_time_list)
    
    # 打印结果摘要
    print("\n========== DeepSpeed Benchmark Result ==========")
    print(f"Model: {os.path.basename(args.model_path.rstrip('/'))}")
    print(f"Seq_len: {args.seq_len}")
    print(f"Batch_size: {args.batch_size}")
    print(f"Precision: {'BF16' if args.use_bf16 else 'FP16'}")
    print(f"Use Liger: {args.use_liger}")
    print(f"ZeRO Stage: {args.zero_stage}")
    print(f"CPU Offload: {args.offload}")
    print(f"NVMe Offload: {args.offload_nvme}")
    print(f"Average batch time: {avg_batch_time:.4f} s")
    print(f"Average speed: {avg_tokens_per_sec:.1f} tokens/s")
    print(f"Average TFLOPS: {avg_tflops:.2f}")
    print(f"Max Memory: {max_memory_mb:.1f} MB")
    print(f"Max CUDA Memory Allocated: {max_cuda_memory_allocated_mb:.1f} MB")
    print(f"Max CUDA Memory Reserved: {max_cuda_memory_reserved_mb:.1f} MB")
    print("=============================================\n")
    
    # 准备结果
    result = {
        "model": os.path.basename(args.model_path.rstrip("/")),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "precision": "BF16" if args.use_bf16 else "FP16",
        "use_liger": "Yes" if args.use_liger else "No",
        "zero_stage": args.zero_stage,
        "offload": "Yes" if args.offload else "No",
        "offload_nvme": "Yes" if args.offload_nvme else "No",
        "avg_time": avg_batch_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "avg_tflops": avg_tflops,
        "max_memory_mb": max_memory_mb,
        "max_cuda_memory_allocated_mb": max_cuda_memory_allocated_mb,
        "max_cuda_memory_reserved_mb": max_cuda_memory_reserved_mb
    }
    
    # 如果提供了结果文件，将结果追加到CSV
    if args.result_file:
        append_result_to_csv(args.result_file, result)
        print(f"结果已追加到文件: {args.result_file}")
    
    # 返回结果，用于脚本收集
    return result

if __name__ == "__main__":
    args = parse_args()
    benchmark_model_ds(args)
