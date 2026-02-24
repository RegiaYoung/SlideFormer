import os
import time
import torch
import gc
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.metric import calculate_flops_per_batch
from utils.datasets import DummyDataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from torch.utils.checkpoint import checkpoint

def test_peak_tflops(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/",
    max_seq_length: int = 1024,
    batch_size: int = 96,
    num_iterations: int = 32,
    use_bf16: bool = True,
):
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设置数据类型
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"使用 {dtype} 进行测试")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型到CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    base_model = AutoLigerKernelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        device_map='cpu'  # 先加载到CPU
    )
    
    # 创建测试数据
    test_dataset = DummyDataset(
        size=batch_size * 2,  # 多创建一些数据
        tokenizer=tokenizer,
        max_length=max_seq_length
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 获取一个batch的数据
    batch = next(iter(test_dataloader))
    
    # 准备输入数据
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # 获取模型层数
    decoder = base_model.get_decoder()
    layers = decoder.layers if hasattr(decoder, 'layers') else decoder.block
    num_layers = len(layers)
    print(f"模型有 {num_layers} 个decoder层")
    hidden_size = base_model.config.hidden_size
    print(f"隐藏层大小: {hidden_size}")
    
    # 计算每个batch的总FLOPS
    total_flops = calculate_flops_per_batch(base_model, batch_size, max_seq_length)
    print(f"每批次FLOPS: {total_flops / 1e12:.2f} TFLOPS")
    print(f"循环次数: {num_iterations}")
    
    # 测试嵌入层
    print("\n------------------ 测试embedding层 ------------------")
    embed_layer = base_model.get_input_embeddings().to(device)
    embed_times = []
    
    # 预热
    _ = embed_layer(input_ids)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    # 测量时间
    for i in range(num_iterations):
        # # 清理内存
        # torch.cuda.empty_cache()
        # # 正向传播
        # torch.cuda.synchronize()
        embed_output = embed_layer(input_ids)
        # 反向传播
        grad = torch.randn_like(embed_output)
        embed_output.backward(grad)

        # 清理梯度
        embed_layer.zero_grad(set_to_none=True)
        # # 清理中间变量
        # del embed_output
        # del grad
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    embed_time =  (end_time - start_time)/num_iterations
    print(f"嵌入层平均时间: {embed_time:.6f} 秒")
    # 清理嵌入层内存
    embed_layer.to('cpu')
    torch.cuda.empty_cache()
    
    # 测试单个decoder层
    print("\n------------------ 测试decoder层 ------------------")
    # 准备必要的组件
    rotary_emb = decoder.rotary_emb.to(device)
    update_causal_mask = decoder._update_causal_mask
    
    # 选择第一个decoder层进行测试
    decoder_layer = layers[0].to(device)
    decoder_times = []
    
    # 使用checkpoint包装forward函数以减少内存使用
    def _decoder_forward(hidden_states, attention_mask, position_embeddings):
        return decoder_layer(
            hidden_states, 
            attention_mask=attention_mask, 
            position_embeddings=position_embeddings,
            output_attentions=False, 
            use_cache=False
        )[0]
    
    # 创建position embeddings
    cache_position = torch.arange(0, max_seq_length, device=device)
    position_ids = cache_position.unsqueeze(0)
    
    hidden_states = torch.randn(batch_size, max_seq_length, hidden_size, 
                                device=device, dtype=dtype)
    # 创建attention mask
    causal_mask = update_causal_mask(attention_mask, hidden_states, cache_position, None, None)
    # 获取position embeddings
    position_embeddings = rotary_emb(hidden_states, position_ids)

    start_time = time.perf_counter()
    # 测量时间
    for i in range(num_iterations):
        # 清理GPU内存
        # torch.cuda.empty_cache()
        
        # # 创建新的输入，需要计算梯度
        # hidden_states = torch.randn(batch_size, max_seq_length, hidden_size, 
        #                          device=device, dtype=dtype, requires_grad=True)
        
        # # 更新position embeddings和attention mask
        # position_embeddings = rotary_emb(hidden_states, position_ids)
        # causal_mask = update_causal_mask(attention_mask, hidden_states, cache_position, None, None)
        
        # 正向传播 - 使用checkpoint减少内存占用
        # torch.cuda.synchronize()
        
        # 使用checkpoint来减少内存占用
        decoder_output = checkpoint(
            _decoder_forward, 
            hidden_states, 
            causal_mask, 
            position_embeddings,
            use_reentrant=False  # 避免reentrant问题
        )
        
        # 反向传播
        grad = torch.randn_like(decoder_output)
        decoder_output.backward(grad) 
        # 清理中间变量和梯度
        # decoder_layer.zero_grad(set_to_none=True)
        # del decoder_output
        # del grad
        # del hidden_states
        # del causal_mask


    torch.cuda.synchronize()
    decoder_time = (time.perf_counter() - start_time)/num_iterations
    print(f"Decoder层平均时间: {decoder_time:.6f} 秒")
    
    # 清理decoder层内存
    decoder_layer.to('cpu')
    rotary_emb.to('cpu')
    torch.cuda.empty_cache()
    
    # 测试输出层
    print("\n------------------ 测试输出层 ------------------")
    # 组合norm和lm_head以及loss
    norm_layer = decoder.norm.to(device)
    lm_head = base_model.get_output_embeddings().to(device)
    lce = LigerFusedLinearCrossEntropyLoss(reduction="mean")

    torch.cuda.empty_cache()
    # 创建新的输入
    hidden_states = torch.randn(batch_size, max_seq_length, hidden_size, 
                                device=device, dtype=dtype, requires_grad=True)
    # 测量时间
    for i in range(num_iterations):
        # 清理GPU内存

        start_time = time.perf_counter()
        
        normed = norm_layer(hidden_states)
        shift_hidden_states = normed[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_hidden_states = shift_hidden_states.view(-1, hidden_size)
        shift_labels = shift_labels.view(-1)
        loss = lce(lm_head.weight, shift_hidden_states, shift_labels)
        
        # 反向传播
        loss.backward()
        # 清理梯度和中间变量
        # norm_layer.zero_grad(set_to_none=True)
        # lm_head.zero_grad(set_to_none=True)
        # del hidden_states
        # del loss

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    output_time = (end_time - start_time)/num_iterations
    print(f"输出层平均时间: {output_time:.6f} 秒")
    
    # 清理输出层内存
    norm_layer.to('cpu')
    lm_head.to('cpu')
    torch.cuda.empty_cache()
    
    # 计算总时间和TFLOPS
    total_time = embed_time + (num_layers * decoder_time) + output_time
    print(f"\n预估总前向+反向时间: {total_time:.6f} 秒")
    
    tflops = total_flops / (total_time * 1e12)
    print(f"峰值 TFLOPS: {tflops:.2f}")
    print(f"时间分布: 嵌入层 ({embed_time:.6f}s) + {num_layers} 个Decoder层 ({num_layers * decoder_time:.6f}s) + 输出层 ({output_time:.6f}s)")
    
    return tflops

if __name__ == "__main__":
    test_peak_tflops()
