import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

"""pytorch allocator config"""
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync" # 峰值-2G，其余无变化
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "garbage_collection_threshold:0.6" # 无明显变化
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import math
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import psutil
import deepspeed
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils.metric import calculate_flops_per_batch
from utils.datasets import MathFusionQADataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM

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

def create_ds_config(
    batch_size,
    use_bf16=True,
    learning_rate=5e-6,
    weight_decay=0.1,
    zero_stage=2,
    offload=True,
    offload_nvme=False,
    nvme_path="/RAID0"
):
    """创建DeepSpeed配置"""
    config = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": weight_decay
            }
        },
        "fp16": {
            "enabled": not use_bf16,
        },
        "bf16": {
            "enabled": use_bf16,
        },
        "zero_optimization": {
            "stage": zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
        }
    }
    
    config["activation_checkpointing"] = {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": True,
    }

    # 设置ZeRO-Offload (CPU)
    if offload:
        if zero_stage < 2:
            print("警告: ZeRO-Offload需要至少ZeRO-2阶段，自动设置为ZeRO-2")
            config["zero_optimization"]["stage"] = 2
        
        # CPU offload参数
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            # "buffer_count": 4,
            # "fast_init": True
        }
        
        # 对于stage 3，启用完全参数分片和offload
        if zero_stage >= 3:
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }
            
            config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
            
            # 优化ZeRO-3的内存使用
            # config["zero_optimization"]["stage3_param_persistence_threshold"] = 0
            # config["zero_optimization"]["stage3_max_live_parameters"] = 5e7
            # config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e7

    # 设置ZeRO-Infinity (NVMe)
    if offload_nvme:
        if zero_stage < 3:
            print("警告: ZeRO-Infinity需要ZeRO-3阶段，自动设置为ZeRO-3")
            config["zero_optimization"]["stage"] = 3
            
        nvme_buffer_size = 5e8  # 默认值
            
        # 配置优化器NVMe offload
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "nvme",
            "pin_memory": True,
            "nvme_path": nvme_path,
            "buffer_count": 8,
            "fast_init": False
        }

        # 配置参数NVMe offload
        config["zero_optimization"]["offload_param"] = {
            "device": "nvme",
            "pin_memory": True,
            "nvme_path": nvme_path,
            "buffer_count": 32,
            "buffer_size": nvme_buffer_size,
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
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 0
        config["zero_optimization"]["stage3_max_live_parameters"] = 5e7
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = "auto"

    return config

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

def train_model_ds(
    max_seq_length=4096,
    train_batch_size=4,
    num_epochs=1,
    save_steps=500,
    save_checkpoints=False,
    output_dir="./mathfusionqa-ft-ds-results",
    use_bf16=True,
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.03,
    zero_stage=3,
    offload=True,
    offload_nvme=False,
    nvme_path="/RAID0",
    use_liger=True
):
    # 在开始前强制清理 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 选择要使用的数据类型
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # 1. 加载tokenizer
    # llama
    # model_path = "/home/scc/models/Llama-3.3-70B-Instruct/"
    model_path = "/home/scc/models/Llama-3.1-8B-Instruct/"
    # qwen2
    # model_path =  "/home/scc/models/Qwen2.5-14B-Instruct/"
    # mistral
    # model_path =  "/home/scc/models/Mistral-7B-Instruct/"

    print(f"Model: {model_path}")
    print(f"Precision: {'BF16' if use_bf16 else 'FP16'}")
    print(f"Use Liger: {use_liger}")
    print(f"ZeRO Stage: {zero_stage}, CPU Offload: {offload}, NVMe Offload: {offload_nvme}")
    print(f"Seq_len: {max_seq_length}, Batch_size: {train_batch_size}")
    print(f"学习率: {learning_rate}, Weight Decay: {weight_decay}, Warmup比例: {warmup_ratio}")
    print(f"Epochs: {num_epochs}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载基础模型
    if use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            use_cache=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            use_cache=False
        )
    
    # 启用梯度检查点以提高内存效率
    if hasattr(model, "gradient_checkpointing_enable"):
        print("启用梯度检查点以提高内存效率")
        model.gradient_checkpointing_enable()
    
    # 3. 创建DeepSpeed配置
    ds_config = create_ds_config(
        batch_size=train_batch_size,
        use_bf16=use_bf16,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        zero_stage=zero_stage,
        offload=offload,
        offload_nvme=offload_nvme,
        nvme_path=nvme_path
    )
    
    # 可选: 保存配置到文件查看
    with open('ds_config_real.json', 'w') as f:
        json.dump(ds_config, f, indent=4)
    
    # 4. 准备数据集 - 使用MathFusionQA数据集
    train_dataset = load_mathfusionqa_datasets(tokenizer, max_length=max_seq_length)
    
    # 打印样本以检查数据集格式
    print(f"训练集大小: {len(train_dataset)}")
    sample = train_dataset[0]
    print("样本数据形状:")
    for k, v in sample.items():
        print(f"{k}: {v.shape}")
    
    # 5. 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, pin_memory=True, drop_last=True)
    
    # 6. 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    # 训练模式
    model_engine.train()
    
    # 计算总步数
    epoch_steps = len(train_dataloader)
    total_steps = epoch_steps * num_epochs
    total_flops = calculate_flops_per_batch(model, train_batch_size, max_seq_length)
    tokens_per_batch = train_batch_size * max_seq_length
    
    # 计算warmup步数
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"总训练步数: {total_steps}, Warmup步数: {warmup_steps}")
    
    # 记录变量初始化
    log_step = 10
    global_step = 0
    total_loss_sum = 0
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
            
            # 更新学习率 - 通过DeepSpeed的方式
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            if step == 0:
                step_start = time.perf_counter()
            
            input_ids = batch['input_ids'].to(model_engine.device)
            attention_mask = batch['attention_mask'].to(model_engine.device)
            labels = batch['labels'].to(model_engine.device)
            
            # 前向传播
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # DeepSpeed backward和优化器步进
            model_engine.backward(loss)
            model_engine.step()
            
            # 计算性能指标
            running_loss += loss.item()
            total_loss_sum += loss.item()
            print(loss.item())
            
            # 输出统计信息
            if step % log_step == 0:
                # 计算统计信息
                step_time = time.perf_counter() - step_start
                avg_loss = running_loss / log_step
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
                
                # 跟踪内存使用
                # memory_mb, cuda_allocated_mb, cuda_reserved_mb = track_memory_usage()
                # print(f"Memory: {memory_mb:.1f}MB, CUDA Allocated: {cuda_allocated_mb:.1f}MB, CUDA Reserved: {cuda_reserved_mb:.1f}MB")
            
            # 更新全局步数
            global_step += 1
            
            # 检查点保存
            if save_checkpoints and global_step % save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                model_engine.save_checkpoint(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"保存检查点到 {checkpoint_path}")
        
        # 打印每个epoch的统计信息
        epoch_time = time.perf_counter() - epoch_start_time
        print(f"\nEpoch {epoch + 1} 完成，用时 {epoch_time:.2f}s, "
              f"平均 tokens/s: {tokens_per_batch*epoch_steps/epoch_time:.1f}")

    # 保存最终模型
    print("\n正在保存最终训练模型...")
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # model_engine.get_state_dict()
    model_engine.save_16bit_model(output_dir, tokenizer)
    # if model_engine.global_rank == 0:
    #     model_engine.zero_collect_model()
    #     print("Rank 0 正在合并并保存最终模型...")
    #     model_to_save =  model_engine.module 
    #     model_to_save.save_pretrained(output_dir, safe_serialization=False)
    #     tokenizer.save_pretrained(output_dir)
    #     print(f"模型和tokenizer已成功保存到 {output_dir}")
    #     print("保存的内容不包含优化器状态，可以直接用于推理。")
    
    # model_engine.save_checkpoint(output_dir)
    # tokenizer.save_pretrained(output_dir)
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
    train_model_ds(
        max_seq_length=4096,
        train_batch_size=32,
        num_epochs=1,
        save_checkpoints=False,
        output_dir="./mathfusion-ft-ds-results",
        use_bf16=True,
        learning_rate=5e-7,
        weight_decay=0.1,  
        warmup_ratio=0.03,
        zero_stage=3,  # 使用ZeRO-2阶段
        offload=True,  # 启用CPU Offload
        offload_nvme=False,  # 不使用NVMe Offload
        use_liger=True  # 使用Liger kernel优化
    )
