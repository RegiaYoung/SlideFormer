from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def low_precision_init(target_dtype: torch.dtype = torch.float16):
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(target_dtype)
        yield
    finally:
        torch.set_default_dtype(dtype)


def get_model_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"
    
    
# 计算模型的参数量和理论FLOPS
def calculate_flops_per_batch(config, batch_size, seq_length):
    """基于Megatron-LM的FLOPS计算方法, 支持GQA和SwiGLU"""
    
    # 基本参数
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads  # GQA中的KV head数量
    vocab_size = config.vocab_size
    intermediate_size = config.intermediate_size
    
    # 注意力相关参数
    head_dim = hidden_size // num_attention_heads
    kv_channels = head_dim
    query_projection_size = kv_channels * num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_size
    
    # GQA相关
    num_query_groups = num_key_value_heads
    
    # 计算因子
    expansion_factor = 3 * 2 * 2
    
    # SwiGLU相关
    gated_linear_multiplier = 3/2  # SwiGLU activation multiplier
    
    # MLP部分的计算需要考虑SwiGLU
    flops = (
        expansion_factor
        * batch_size
        * seq_length
        * num_layers
        * hidden_size
        * hidden_size
        * (
            # 注意力机制
            (
                1  # Q投影
                + (num_query_groups / num_attention_heads)  # K,V投影合并计算
                + (seq_length / hidden_size)  # QK注意力计算
            ) * query_projection_to_hidden_size_ratio
            # MLP部分，加入SwiGLU因子
            + (intermediate_size / hidden_size) * gated_linear_multiplier
            # 词表投影
            + (vocab_size / (2 * num_layers * hidden_size))
        )
    )
    
    return flops
