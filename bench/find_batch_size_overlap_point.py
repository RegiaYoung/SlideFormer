import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import gc
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.datasets import DummyDataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from offload_transformer import SlideFormerOffloader
import concurrent.futures
from transformer_layer import DecoderWrapper
from optimizer import LayerAdam

def measure_grad_d2h_time(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/", 
    layer_idx: int = 1,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_bf16: bool = True
):
    """精确测量梯度从GPU到CPU的传输时间"""
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设置数据类型
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"\n正在测量梯度从GPU到CPU的传输时间...")
    print(f"使用数据类型: {dtype}")
    
    # 加载模型获取目标层参数量
    device = torch.device("cuda")
    print(f"使用设备: {device}")
    
    # 使用CPU加载模型以获取实际参数结构（而非meta张量）
    print("正在加载模型配置...")
    base_model = AutoLigerKernelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',  # 先加载到CPU而非meta
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    # 获取decoder层
    decoder = base_model.get_decoder()
    layers = decoder.layers if hasattr(decoder, 'layers') else decoder.block
    
    # 确保指定的layer_idx有效
    if layer_idx <= 0 or layer_idx > len(layers):
        layer_idx = 1
        print(f"调整为有效的layer_idx: {layer_idx}")
    
    # 选取目标层并计算参数量
    target_layer = layers[layer_idx-1]  # 索引从0开始
    param_count = sum(p.numel() for p in target_layer.parameters())
    print(f"层 {layer_idx} 的参数数量: {param_count}")
    
    # 释放模型内存
    del base_model, decoder, layers, target_layer
    torch.cuda.empty_cache()
    gc.collect()
    
    # 创建所需的张量
    cpu_grad_tensor = torch.empty(param_count, dtype=dtype, device="cpu", pin_memory=True)
    gpu_grad_tensor = torch.empty(param_count, dtype=dtype, device=device)
    
    # 填充随机数据
    gpu_grad_tensor.normal_()
    
    # 创建CUDA流和事件
    stream = torch.cuda.Stream()
    
    # 预热
    print("预热中...")
    for _ in range(warmup_iterations):
        with torch.cuda.stream(stream):
            cpu_grad_tensor.copy_(gpu_grad_tensor, non_blocking=True)
        stream.synchronize()
    
    # 正式测量
    print(f"开始测量，执行 {num_iterations} 次迭代...")
    transfer_times = []
    
    for i in range(num_iterations):
        stream.synchronize()  # 确保之前的操作完成
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.cuda.stream(stream):
            cpu_grad_tensor.copy_(gpu_grad_tensor, non_blocking=True)
        
        stream.synchronize()  # 等待传输完成
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        transfer_times.append(end_time - start_time)
    
    # 计算统计数据
    avg_time = np.mean(transfer_times)
    std_time = np.std(transfer_times)
    min_time = np.min(transfer_times)
    max_time = np.max(transfer_times)
    
    print(f"梯度D2H传输时间统计 (参数量: {param_count}):")
    print(f"  平均时间: {avg_time*1000:.4f}ms")
    print(f"  标准差: {std_time*1000:.4f}ms")
    print(f"  最小/最大: {min_time*1000:.4f}ms / {max_time*1000:.4f}ms")
    
    return avg_time

def measure_update_time(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/",
    layer_idx: int = 1,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_bf16: bool = True,
    optimizer_kwargs: dict = None
):
    """精确测量参数更新时间"""
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设置数据类型
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"\n正在测量参数更新时间...")
    print(f"使用数据类型: {dtype}")
    
    # 使用CPU加载模型以获取实际参数结构（而非meta张量）
    print("正在加载模型配置...")
    base_model = AutoLigerKernelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',  # 先加载到CPU而非meta
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    # 获取decoder层
    decoder = base_model.get_decoder()
    layers = decoder.layers if hasattr(decoder, 'layers') else decoder.block
    
    # 确保指定的layer_idx有效
    if layer_idx <= 0 or layer_idx > len(layers):
        layer_idx = 1
        print(f"调整为有效的layer_idx: {layer_idx}")
    
    # 选取目标层并加载到CPU
    target_layer = layers[layer_idx-1]  # 确保在CPU上
    param_count = sum(p.numel() for p in target_layer.parameters())
    print(f"层 {layer_idx} 的参数数量: {param_count}")
    
    # 初始化LayerAdam优化器
    default_optimizer_kwargs = {
        "lr": 1e-5,
        "bias_correction": True,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "fp32_optimizer_state": True,
        "num_layer": 1,
        "nvme_offload_fraction": 0.0,
        "offload_dir": "/tmp",
        "prefetch": False
    }
    
    optimizer_kwargs = optimizer_kwargs or default_optimizer_kwargs
    optimizer = LayerAdam(**optimizer_kwargs)
    
    # 将参数注册到优化器
    param_to_grad_views = {}
    optimizer.add_layer_params(0, target_layer.parameters())
    
    # 创建梯度视图
    cpu_grad_tensor = torch.empty(param_count, dtype=dtype, device="cpu", pin_memory=True)
    cpu_grad_tensor.normal_()  # 填充随机数据
    
    # 设置梯度视图映射
    offset = 0
    for param in target_layer.parameters():
        size = param.numel()
        shape = param.shape
        param_to_grad_views[param] = cpu_grad_tensor[offset:offset + size].view(shape)
        offset += size
    
    # 预热
    print("预热中...")
    for _ in range(warmup_iterations):
        optimizer.step_with_grad_views(0, param_to_grad_views)
    
    # 正式测量
    print(f"开始测量，执行 {num_iterations} 次迭代...")
    update_times = []
    
    for i in range(num_iterations):
        # 重新填充随机数据，模拟不同的梯度
        cpu_grad_tensor.normal_()
        
        start_time = time.perf_counter()
        
        # 执行更新
        optimizer.step_with_grad_views(0, param_to_grad_views)
        
        end_time = time.perf_counter()
        update_times.append(end_time - start_time)
    
    # 释放模型内存
    del base_model, decoder, layers, target_layer
    torch.cuda.empty_cache()
    gc.collect()
    
    # 计算统计数据
    avg_time = np.mean(update_times)
    std_time = np.std(update_times)
    min_time = np.min(update_times)
    max_time = np.max(update_times)
    
    print(f"参数更新时间统计 (参数量: {param_count}):")
    print(f"  平均时间: {avg_time*1000:.4f}ms")
    print(f"  标准差: {std_time*1000:.4f}ms")
    print(f"  最小/最大: {min_time*1000:.4f}ms / {max_time*1000:.4f}ms")
    
    return avg_time

def measure_backward_time(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/",
    layer_idx: int = 1,
    batch_size: int = 1,
    seq_length: int = 512,
    num_iterations: int = 10,
    warmup_iterations: int = 2,
    use_bf16: bool = True
):
    """精确测量不同批大小下的反向传播时间"""
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设置数据类型
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    device = torch.device("cuda")
    print(f"\n正在测量批大小 {batch_size} 的反向传播时间...")
    print(f"使用数据类型: {dtype}")
    
    # 加载模型
    print("正在加载模型...")
    base_model = AutoLigerKernelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        device_map='cpu'  # 改为加载到CPU以避免meta张量问题
    )
    
    # 获取decoder层
    decoder = base_model.get_decoder()
    layers = decoder.layers if hasattr(decoder, 'layers') else decoder.block
    
    # 确保指定的layer_idx有效
    if layer_idx <= 0 or layer_idx > len(layers):
        layer_idx = 1
        print(f"调整为有效的layer_idx: {layer_idx}")
    
    # 选取目标层并移至GPU
    print(f"正在将目标层 {layer_idx} 加载到GPU...")
    target_layer = layers[layer_idx-1].to(device)
    hidden_size = base_model.config.hidden_size
    
    # 获取rotary_emb和update_causal_mask
    rotary_emb = decoder.rotary_emb.to(device)
    update_causal_mask = decoder._update_causal_mask
    
    # 释放基础模型内存，仅保留需要的组件
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # 创建输入数据
    hidden_states = torch.randn(batch_size, seq_length, hidden_size, 
                               device=device, dtype=dtype, requires_grad=True)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    
    # 创建position embeddings
    cache_position = torch.arange(0, seq_length, device=device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = rotary_emb(hidden_states, position_ids)
    
    # 创建attention mask
    causal_mask = update_causal_mask(attention_mask, hidden_states, cache_position, None, None)
    
    # 定义forward函数
    def _forward(hidden_states, causal_mask, position_embeddings):
        return target_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            output_attentions=False,
            use_cache=False
        )[0]
    
    # 预热
    print("预热中...")
    for _ in range(warmup_iterations):
        # 前向传播
        output = checkpoint(_forward, hidden_states, causal_mask, position_embeddings, use_reentrant=False)
        # 创建随机梯度
        grad = torch.randn_like(output)
        # 反向传播
        output.backward(grad, retain_graph=True)
    
    torch.cuda.empty_cache()
    
    # 正式测量
    print(f"开始测量，执行 {num_iterations} 次迭代...")
    backward_times = []
    
    for i in range(num_iterations):
        # 前向传播
        output = checkpoint(_forward, hidden_states, causal_mask, position_embeddings, use_reentrant=False)
        # 创建随机梯度
        grad = torch.randn_like(output)
        
        # 同步GPU并开始计时
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # 反向传播
        output.backward(grad, retain_graph=True)
        
        # 同步GPU并结束计时
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        backward_times.append(end_time - start_time)
    
    # 清理
    hidden_states.grad = None
    target_layer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    
    # 计算统计数据
    avg_time = np.mean(backward_times)
    std_time = np.std(backward_times)
    min_time = np.min(backward_times)
    max_time = np.max(backward_times)
    
    print(f"反向传播时间统计 (批大小: {batch_size}, 序列长度: {seq_length}):")
    print(f"  平均时间: {avg_time*1000:.4f}ms")
    print(f"  标准差: {std_time*1000:.4f}ms")
    print(f"  最小/最大: {min_time*1000:.4f}ms / {max_time*1000:.4f}ms")
    
    # 释放GPU内存
    del target_layer, rotary_emb, decoder
    torch.cuda.empty_cache()
    
    return avg_time

def find_overlap_batch_size(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/",
    layer_idx: int = 1,
    seq_length: int = 512,
    max_batch_size: int = 64,
    num_iterations: int = 10,
    warmup_iterations: int = 2,
    use_bf16: bool = True
):
    """找出T_backward > T_grad_d2h + T_update的批大小转折点"""
    print("\n" + "="*80)
    print(f"开始测试批大小转折点")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"测试层: {layer_idx}")
    print(f"序列长度: {seq_length}")
    
    # 1. 测量T_grad_d2h
    print("\n1. 测量梯度从GPU到CPU的传输时间")
    t_grad_d2h = measure_grad_d2h_time(
        model_path=model_path,
        layer_idx=layer_idx,
        num_iterations=100,
        warmup_iterations=10,
        use_bf16=use_bf16
    )
    
    # 2. 测量T_update
    print("\n2. 测量参数更新时间")
    t_update = measure_update_time(
        model_path=model_path,
        layer_idx=layer_idx,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        use_bf16=use_bf16
    )
    
    # 计算总更新开销
    total_update_overhead = t_grad_d2h + t_update
    print(f"\n总更新开销 (T_grad_d2h + T_update): {total_update_overhead*1000:.4f}ms")
    
    # 3. 使用二分查找寻找转折点批大小
    print("\n3. 开始寻找转折点批大小")
    left, right = 1, max_batch_size
    found = False
    last_smaller = None
    first_larger = None
    left_time = None
    right_time = None
    
    # 先测试最小批大小
    t_backward_min = measure_backward_time(
        model_path=model_path,
        layer_idx=layer_idx,
        batch_size=left,
        seq_length=seq_length,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        use_bf16=use_bf16
    )
    
    if t_backward_min > total_update_overhead:
        print(f"\n最小批大小 {left} 的反向传播时间 ({t_backward_min*1000:.4f}ms) 已经大于更新开销 ({total_update_overhead*1000:.4f}ms)")
        return left
    
    last_smaller = left
    left_time = t_backward_min
    
    # 测试最大批大小
    t_backward_max = measure_backward_time(
        model_path=model_path,
        layer_idx=layer_idx,
        batch_size=right,
        seq_length=seq_length,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        use_bf16=use_bf16
    )
    
    if t_backward_max <= total_update_overhead:
        print(f"\n最大批大小 {right} 的反向传播时间 ({t_backward_max*1000:.4f}ms) 仍小于或等于更新开销 ({total_update_overhead*1000:.4f}ms)")
        print(f"需要更大的批大小来找到转折点。")
        return None
    
    first_larger = right
    right_time = t_backward_max
    
    # 使用二分查找寻找转折点
    while right - left > 1:
        mid = (left + right) // 2
        
        t_backward_mid = measure_backward_time(
            model_path=model_path,
            layer_idx=layer_idx,
            batch_size=mid,
            seq_length=seq_length,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            use_bf16=use_bf16
        )
        
        if t_backward_mid > total_update_overhead:
            first_larger = mid
            right = mid
            right_time = t_backward_mid
        else:
            last_smaller = mid
            left = mid
            left_time = t_backward_mid
    
    crossover_bs = first_larger
    
    # 总结结果
    print("\n" + "="*80)
    print("批大小转折点实验结果")
    print("="*80)
    print(f"梯度D2H传输时间: {t_grad_d2h*1000:.4f}ms")
    print(f"参数更新时间: {t_update*1000:.4f}ms")
    print(f"总更新开销: {total_update_overhead*1000:.4f}ms")
    print(f"\n批大小转折点: {crossover_bs}")
    
    if last_smaller is not None and left_time is not None:
        print(f"批大小 {last_smaller} 的反向传播时间: {left_time*1000:.4f}ms < 更新开销: {total_update_overhead*1000:.4f}ms")
    
    if first_larger is not None and right_time is not None:
        print(f"批大小 {first_larger} 的反向传播时间: {right_time*1000:.4f}ms > 更新开销: {total_update_overhead*1000:.4f}ms")
    
    return crossover_bs

def test_multiple_layers(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/",
    layers_to_test: list = None,
    seq_length: int = 512,
    max_batch_size: int = 64,
    num_iterations: int = 10,
    warmup_iterations: int = 2,
    use_bf16: bool = True
):
    """测试多个层的转折点批大小"""
    # 如果未指定要测试的层，则测试第一层、中间层
    if layers_to_test is None:
        # 获取模型层数（使用轻量级加载）
        print("正在加载模型配置以获取层数...")
        config = AutoLigerKernelForCausalLM.from_pretrained(
            model_path, 
            device_map=None, 
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
        ).config
        
        num_layers = config.num_hidden_layers
        print(f"模型共有 {num_layers} 层")
        
        # 选择第一层、中间层
        layers_to_test = [1, num_layers // 2]
    
    results = []
    
    for layer_idx in layers_to_test:
        print(f"\n\n{'#'*100}")
        print(f"测试层 {layer_idx}")
        print(f"{'#'*100}")
        
        crossover_bs = find_overlap_batch_size(
            model_path=model_path,
            layer_idx=layer_idx,
            seq_length=seq_length,
            max_batch_size=max_batch_size,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            use_bf16=use_bf16
        )
        
        results.append((layer_idx, crossover_bs))
    
    # 打印总结报告
    print("\n\n" + "="*100)
    print("批大小转折点测试总结报告")
    print("="*100)
    
    for layer_idx, crossover_bs in results:
        status = "找到" if crossover_bs is not None else "超出最大批大小"
        print(f"层 {layer_idx}: 批大小转折点 = {crossover_bs if crossover_bs is not None else '>'+str(max_batch_size)} ({status})")
    
    return results

if __name__ == "__main__":
    # 执行测试
    test_multiple_layers(
        # model_path="/home/scc/models/Llama-3.1-8B-Instruct/",
        model_path="/home/scc/models/Qwen2.5-3B-Instruct/",
        # model_path="/home/scc/models/Qwen2.5-7B-Instruct/",
        # model_path="/home/scc/models/Qwen2.5-14B-Instruct/",
        # model_path="/home/scc/models/Qwen2.5-32B-Instruct/",
        # model_path="/home/scc/models/Qwen2.5-72B-Instruct/",
        seq_length=1024,
        max_batch_size=32,
        num_iterations=10,
        warmup_iterations=2
    )
