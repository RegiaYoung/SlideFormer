import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import gc
import time
import torch
import argparse
import deepspeed
import numpy as np
import json
import psutil
from utils.log_mem import log_memory_stats
# 替换自定义模型导入，改为transformers的GPT2模型
from transformers import GPT2Config, GPT2LMHeadModel
# 导入LigerFusedLinearCrossEntropyLoss
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 创建自定义GPT2模型，使用LigerFusedLinearCrossEntropyLoss进行优化
class LigerGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化融合损失函数
        self.lce = LigerFusedLinearCrossEntropyLoss(reduction='mean')
        
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用transformer获取隐藏状态
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # 设置设备，用于模型并行
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # 使用融合损失计算方式替代原始实现
        loss = None
        lm_logits = None
        
        if labels is not None:
            # 为融合算子准备输入：移位隐藏状态和标签
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 扁平化tokens
            shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # 使用融合算子计算损失
            loss = self.lce(
                self.lm_head.weight,  # 直接传递权重
                shift_hidden_states,
                shift_labels
            )
            # 不需要计算logits，可以节省内存
            lm_logits = None
        else:
            # 如果没有labels，则正常计算logits
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回输出，符合原始API
        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Zero3-Offload Evaluation for GPT Models")
    parser.add_argument("--seq_len", type=int, default=1024, help="序列长度")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--hidden_dim", type=int, default=5120, help="隐藏层维度")
    parser.add_argument("--num_heads", type=int, default=80, help="注意力头数量")
    parser.add_argument("--num_layers", type=int, default=40, help="Transformer层数")
    parser.add_argument("--vocab_size", type=int, default=50257, help="词表大小")
    parser.add_argument("--use_bf16", action="store_true", help="使用bf16精度")
    parser.add_argument("--warm_step", type=int, default=1, help="预热步数")
    parser.add_argument("--test_step", type=int, default=3, help="测试步数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--result_file", type=str, default=None, help="结果CSV文件路径")
    
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

def create_ds_config(args):
    """创建优化的DeepSpeed配置，默认使用ZeRO-3和CPU卸载"""
    config = {
        "train_batch_size": args.batch_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": not args.use_bf16,
        },
        "bf16": {
            "enabled": args.use_bf16,
        },
        "zero_optimization": {
            "stage": 3,  # 默认使用ZeRO-Stage3
            # 增加零优化内存效率的参数
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            
            # 默认开启参数卸载到CPU
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            
            # 默认开启优化器状态卸载到CPU
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": True
            },
            
            # 优化ZeRO-3内存使用的关键参数
            "stage3_param_persistence_threshold": 10000,  # 小参数保留在GPU上的阈值
            "stage3_max_live_parameters": 1e8,           # GPU上最大活跃参数量
            "stage3_prefetch_bucket_size": 5e8,          # 预取参数的桶大小
        }
    }
    
    # 激活检查点配置，进一步降低内存使用
    config["activation_checkpointing"] = {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": True,
    }

    return config

def append_result_to_csv(result_file, result_dict):
    """将结果追加到CSV文件中"""
    import csv
    # 检查文件是否存在，如果不存在则创建并写入头部
    file_exists = os.path.isfile(result_file)
    
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['Hidden_Dim', 'Num_Heads', 'Num_Layers', 'Sequence_Length', 'Batch_Size', 
                     'Precision', 'Offload_Param', 'Offload_Optimizer', 'Stage3_Param_Threshold',
                     'Stage3_Max_Live_Params', 'Stage3_Prefetch_Size',
                     'Avg_Time', 'Avg_Tokens_Per_Second', 'Avg_TFLOPS',
                     'Max_Memory_MB', 'Max_CUDA_Memory_Allocated_MB', 'Max_CUDA_Memory_Reserved_MB']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Hidden_Dim': result_dict['hidden_dim'],
            'Num_Heads': result_dict['num_heads'],
            'Num_Layers': result_dict['num_layers'],
            'Sequence_Length': result_dict['seq_len'],
            'Batch_Size': result_dict['batch_size'],
            'Precision': result_dict['precision'],
            'Offload_Param': result_dict['offload_param'],
            'Offload_Optimizer': result_dict['offload_optimizer'],
            'Stage3_Param_Threshold': result_dict['stage3_param_threshold'],
            'Stage3_Max_Live_Params': result_dict['stage3_max_live_params'],
            'Stage3_Prefetch_Size': result_dict['stage3_prefetch_size'],
            'Avg_Time': f"{result_dict['avg_time']:.4f}",
            'Avg_Tokens_Per_Second': f"{result_dict['avg_tokens_per_sec']:.1f}",
            'Avg_TFLOPS': f"{result_dict['avg_tflops']:.2f}",
            'Max_Memory_MB': f"{result_dict['max_memory_mb']:.1f}",
            'Max_CUDA_Memory_Allocated_MB': f"{result_dict['max_cuda_memory_allocated_mb']:.1f}",
            'Max_CUDA_Memory_Reserved_MB': f"{result_dict['max_cuda_memory_reserved_mb']:.1f}"
        })

def benchmark_gpt_model_ds(args):
    # 在开始前强制清理 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()

    # 内存跟踪变量初始化
    max_memory_mb = 0
    max_cuda_memory_allocated_mb = 0
    max_cuda_memory_reserved_mb = 0

    # 选择要使用的数据类型
    dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    print(f"使用 {dtype} 进行测试")
    print(f"模型配置: 隐藏层维度={args.hidden_dim}, 头数={args.num_heads}, 层数={args.num_layers}")
    print(f"序列长度: {args.seq_len}, 批次大小: {args.batch_size}")
    print(f"DeepSpeed设置: ZeRO-3 (默认启用参数和优化器卸载到CPU)")
    print(f"预热步数: {args.warm_step}, 测试步数: {args.test_step}")

    # 1. 创建GPT2配置和模型
    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.seq_len,
        n_embd=args.hidden_dim,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        embd_pdrop=0.1
    )
    
    # 创建GPT2模型实例，使用我们的自定义模型替代
    print("使用 LigerFusedLinearCrossEntropyLoss 融合算子...")
    model = LigerGPT2LMHeadModel(config)
    
    # 启用梯度检查点以提高内存效率
    if hasattr(model, "gradient_checkpointing_enable"):
        print("启用梯度检查点...")
        model.gradient_checkpointing_enable()
    
    # 2. 创建DeepSpeed配置
    ds_config = create_ds_config(args)
    # 可选: 保存配置到文件查看
    with open('ds_gpt_config_temp.json', 'w') as f:
        json.dump(ds_config, f, indent=4)
    
    # 3. 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    # 训练模式
    model_engine.train()

    # 计算FLOPS
    seq_len = args.seq_len
    h = args.hidden_dim
    L = args.num_layers
    b = args.batch_size
    
    # 估计每个token的浮点运算次数
    flops_per_token = 6 * L * h * h  # 简化的估计
    total_flops = flops_per_token * seq_len * b
    
    tokens_per_batch = args.batch_size * args.seq_len

    # 统计数据初始化
    tokens_per_sec_list = []
    tflops_list = []
    batch_time_list = []
    
    # 直接生成随机输入数据
    device = model_engine.device
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    # 为GPT2模型准备attention_mask
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    print(f"测试开始: {args.warm_step} 步预热和 {args.test_step} 步测试")
    
    # 开始循环，使用简单的for循环而不是DataLoader
    total_steps = args.warm_step + args.test_step
    for step in range(total_steps):
        step_start = time.perf_counter()

        # 前向和后向传播，使用GPT2模型的标准输入格式
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        
        # DeepSpeed后向传播
        model_engine.backward(loss)
        model_engine.step()
        
        # 释放输出以节省内存
        del outputs
        
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
            
            print(f"步骤 {step+1}/{total_steps}, 损失: {loss:.4f}, 时间: {step_time:.4f}s, "
                  f"速度: {tokens_per_sec:.1f} tokens/s, "
                  f"TFLOPS: {tflops:.2f}")
        else:
            print(f"预热 {step+1}/{args.warm_step}")
    
    # 计算平均值
    avg_tokens_per_sec = np.mean(tokens_per_sec_list)
    avg_tflops = np.mean(tflops_list)
    avg_batch_time = np.mean(batch_time_list)
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("性能统计摘要:")
    print("="*50)
    
    # 打印配置信息
    print(f"模型配置:")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 序列长度: {args.seq_len}")
    print(f"  - 隐藏层维度: {args.hidden_dim}")
    print(f"  - 注意力头数: {args.num_heads}")
    print(f"  - 层数: {args.num_layers}")
    print(f"  - 精度: {'BF16' if args.use_bf16 else 'FP16'}")
    print(f"  - DeepSpeed ZeRO-3: 已启用")
    print(f"  - 参数卸载到CPU: 已启用")
    print(f"  - 优化器卸载到CPU: 已启用")
    
    # 打印性能结果
    print(f"\n{args.test_step}步的平均性能:")
    print(f"  - 平均批次时间: {avg_batch_time:.4f} s")
    print(f"  - 平均速度: {avg_tokens_per_sec:.1f} tokens/s")
    print(f"  - 平均TFLOPS: {avg_tflops:.2f}")
    
    # 内存使用统计
    print(f"\n内存使用:")
    print(f"  - 最大CPU内存: {max_memory_mb / 1024:.2f} GB")
    print(f"  - 最大CUDA内存分配: {max_cuda_memory_allocated_mb:.2f} MB")
    print(f"  - 最大CUDA内存预留: {max_cuda_memory_reserved_mb:.2f} MB")
    print("="*50)

    # 准备结果
    result = {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "precision": "BF16" if args.use_bf16 else "FP16",
        "offload_param": "Yes",  # 默认启用
        "offload_optimizer": "Yes",  # 默认启用
        "stage3_param_threshold": 10000,
        "stage3_max_live_params": int(1e8),
        "stage3_prefetch_size": int(5e8),
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
    benchmark_gpt_model_ds(args)
