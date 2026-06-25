# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""pytorch allocator config"""
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync" # 峰值-2G，其余无变化
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "garbage_collection_threshold:0.6" # 无明显变化
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import math  # 添加math库用于cosine学习率计算
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils.metric import calculate_flops_per_batch
from utils.datasets import MathFusionQADataset
from offload_transformer import SlideFormerOffloader

try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
except ModuleNotFoundError:
    AutoLigerKernelForCausalLM = None



def load_mathfusionqa_datasets(tokenizer, max_length=4096):
    """加载MathFusionQA的8个split数据集并合并"""
    
    datasets = []
    
    # 直接加载整个数据集字典
    dataset_dict = load_dataset("QizhiPei/MathFusionQA")
    
    print(f"数据集结构: {dataset_dict}")
    
    # 遍历所有split并转换为自定义数据集格式
    for split_name, dataset in dataset_dict.items():
        try:
            processed_dataset = MathFusionQADataset(dataset, tokenizer, max_length)
            datasets.append(processed_dataset)
            print(f"加载数据集 {split_name} 成功，大小: {len(processed_dataset)}")
        except Exception as e:
            print(f"处理数据集 {split_name} 失败: {e}")
    
    # 合并所有数据集
    if datasets:
        combined_dataset = ConcatDataset(datasets)
        print(f"成功合并所有数据集，总大小: {len(combined_dataset)}")
        return combined_dataset
    else:
        raise ValueError("没有成功加载任何数据集")


def train_model(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/",
    max_seq_length=4096,
    train_batch_size=4,
    num_epochs=1,
    output_dir="./mathfusionqa-ft-results",
    use_bf16=True,
    attn_implementation: str = "flash_attention_2",
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.03,
):
    # 在开始前强制清理 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 选择要使用的数据类型
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 确保在 CPU 上加载基础模型 (mem: 8->23.5G)
    model_cls = AutoLigerKernelForCausalLM or AutoModelForCausalLM
    base_model = model_cls.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="flash_attention_2", # 使用 Flash Attention, 注释掉即为 sdpa
    #     torch_dtype=torch.float16,
    #     device_map='cpu'  # 显式指定 device_map
    # )
    
    # print(list(base_model.named_parameters()))
        
    # 3. 创建 SlideFormerOffloader (mem: 23.5->69.3G gpu: 0 -> 4.9G)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer_kwargs = {
        "lr": learning_rate,  # 使用传入的学习率
        "bias_correction": True,
        "weight_decay": weight_decay,  # 使用传入的weight_decay
        "eps": 1e-8,
        "fp32_optimizer_state": True,
        "num_layer": len(base_model.get_decoder().layers) + 2,  # 包括嵌入层和输出层
        "nvme_offload_fraction": 0.0, # 0为关闭，目前是0/0.5/1三档
        "offload_dir": None,
        "prefetch": True
    }
    
    model = SlideFormerOffloader(
        model=base_model,
        device=device,
        dtype=dtype,  #
        enable_timing=False,
        enable_memory_stats=False,
        optimizer_kwargs=optimizer_kwargs,
    )
    
    # 3. 准备数据集 - 使用MathFusionQA数据集
    train_dataset = load_mathfusionqa_datasets(tokenizer, max_length=max_seq_length)
    
    # 打印一些样本以检查数据集格式
    print(f"训练集大小: {len(train_dataset)}")
    sample = train_dataset[0]
    print("样本数据形状:")
    for k, v in sample.items():
        print(f"{k}: {v.shape}")
    
    # 4. 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, pin_memory=True, drop_last=True) # , shuffle=True
    
    # 移除学习率调度器创建代码
    
    # 5. 训练循环
    model.train()
    
    # 计算总步数
    epoch_steps = len(train_dataloader)
    total_steps = epoch_steps * num_epochs
    total_flops = calculate_flops_per_batch(base_model, train_batch_size, max_seq_length)
    tokens_per_batch = train_batch_size * max_seq_length
    
    # 计算warmup步数
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # 打印日志的步数
    log_step = 10  # 设置为5，匹配原脚本的logging_steps
    # 全局步数计数器
    global_step = 0
    total_loss_sum = 0
    # total_batches = 0
    # loss
    losses = []
    steps = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        running_loss = 0
        for step, batch in enumerate(train_dataloader):
            # 计算当前步的学习率
            if global_step < warmup_steps:
                # 线性warmup
                current_lr = learning_rate * (global_step / max(1, warmup_steps))
            else:
                # cosine decay
                progress = (global_step - warmup_steps) / max(1, (total_steps - warmup_steps))
                current_lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # 更新学习率
            model.update_learning_rate(current_lr)
            
            if step == 0:
                step_start = time.perf_counter()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 使用 OffloadTransformerModel 进行前向和后向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 计算性能指标
            running_loss += loss.item()
            # losses.append(loss.item())
            # steps.append(global_step)
            total_loss_sum += loss.item()
            # total_batches += 1
            # print(loss.item())
            
            # 立即删除不需要的输出
            del outputs
            
            # 输出统计信息
            if step % log_step == 0:
                # 计算统计信息
                step_time = time.perf_counter() - step_start
                avg_loss = running_loss / log_step
                if step != 0:
                    losses.append(avg_loss)
                    steps.append(global_step + 1)
                avg_tokens_per_sec = tokens_per_batch*log_step / (step_time)
                avg_tflops = total_flops*log_step / (10**12 * (step_time))
                avg_iter_per_sec = log_step / (step_time)
                avg_batch_time = step_time / log_step  # Time per batch in seconds
                global_avg_loss = total_loss_sum / (global_step + 1)
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {step}/{epoch_steps}, "
                      f"Loss: {avg_loss:.4f}, Avg Loss: {global_avg_loss:.4f}, LR: {current_lr:.2e}, "
                      f"Time: {avg_batch_time:.2f}s, "
                      f"Speed: {avg_tokens_per_sec:.1f} tokens/s, "
                      f"Iter/s: {avg_iter_per_sec:.2f}, "
                      f"TFLOPS: {avg_tflops:.2f}")
                
                running_loss = 0
                step_start = time.perf_counter()
            
            # 更新全局步数
            global_step += 1
        
        # 打印每个epoch的统计信息
        epoch_time = time.perf_counter() - epoch_start_time
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s, "
              f"Average tokens/s: {tokens_per_batch*epoch_steps/epoch_time:.1f}")

    
    # 8. 保存最终模型
    print("\n正在保存最终训练模型...")
    # 使用SlideFormerOffloader中的save_pretrained方法保存底层模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # print(f"模型和tokenizer已保存到 {output_dir}")
    
    # 绘制loss曲线
    print("正在生成loss曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存loss图
    loss_figure_path = os.path.join(output_dir, 'loss_curve.pdf')
    plt.savefig(loss_figure_path)
    print(f"Loss曲线已保存到 {loss_figure_path}")

    # 保存loss数据为CSV
    loss_data = pd.DataFrame({'step': steps, 'loss': losses})
    loss_csv_path = os.path.join(output_dir, 'loss_data.csv')
    loss_data.to_csv(loss_csv_path, index=False)
    print(f"Loss数据已保存到 {loss_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SlideFormer real fine-tuning (MathFusionQA).")
    parser.add_argument("--model_path", type=str, default="/home/scc/models/Llama-3.1-8B-Instruct/")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--output_dir", type=str, default="./mathfusion-ft-results")
    parser.add_argument(
        "--attn_implementation", type=str,
        default="flash_attention_2", choices=["flash_attention_2", "sdpa"]
    )
    args = parser.parse_args()

    train_model(
        model_path=args.model_path,
        max_seq_length=args.seq_len,
        train_batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        use_bf16=args.use_bf16,
        attn_implementation=args.attn_implementation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )
