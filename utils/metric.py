import torch

# 计算模型的参数量和理论FLOPS
def calculate_flops_per_batch(model, batch_size, seq_length):
    """基于Megatron-LM的FLOPS计算方法, 支持GQA和SwiGLU"""
    config = model.config
    
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



# def calculate_flops_per_batch(model, batch_size, seq_length):
#     """基于Megatron-LM的FLOPS计算方法, 支持GQA和SwiGLU（修正版）"""
#     config = model.config
    
#     # 基本参数
#     hidden_size = config.hidden_size
#     num_layers = config.num_hidden_layers
#     num_attention_heads = config.num_attention_heads
#     num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
#     vocab_size = config.vocab_size
#     intermediate_size = config.intermediate_size
#     swiglu = 'silu' in getattr(config, 'hidden_act', '')  # 更稳健的SwiGLU检测
    
#     # 验证GQA参数
#     assert hidden_size % num_attention_heads == 0, "hidden_size必须能被num_attn_heads整除"
#     assert num_attention_heads % num_key_value_heads == 0, "num_attn_heads必须能被num_kv_heads整除"
    
#     # 注意力层FLOPs（前向）
#     qkv_ratio = 1 + 2 * (num_key_value_heads / num_attention_heads)  # Q + K + V投影
#     attn_proj_flops = 2 * batch_size * seq_length * hidden_size**2 * qkv_ratio
#     attn_core_flops = 4 * batch_size * num_attention_heads * seq_length**2 * (hidden_size // num_attention_heads)
#     attn_flops = attn_proj_flops + attn_core_flops
    
#     # MLP层FLOPs（前向）
#     swiglu_factor = 3.0 if swiglu else 2.0  # 修正SwiGLU系数
#     mlp_flops = 2 * swiglu_factor * batch_size * seq_length * hidden_size * intermediate_size
    
#     # 总层FLOPs（前向）
#     total_layers_flops = num_layers * (attn_flops + mlp_flops)
    
#     # Logits FLOPs（前向）
#     logits_flops = 2 * batch_size * seq_length * hidden_size * vocab_size
    
#     # 总训练FLOPs = (前向+反向) = 前向*3
#     total_flops = (total_layers_flops + logits_flops) * 3
    
#     return total_flops

# def calculate_flops_per_batch(model, batch_size, seq_length):
#     """基于Megatron-LM的FLOPS计算方法, 支持GQA和SwiGLU"""
#     config = model.config
    
#     # 基本参数
#     hidden_size = config.hidden_size
#     num_layers = config.num_hidden_layers
#     num_attention_heads = config.num_attention_heads
#     num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)  # GQA中的KV头数量
#     vocab_size = config.vocab_size
#     intermediate_size = config.intermediate_size
#     swiglu = getattr(config, 'hidden_act', '') == 'silu'  # LLaMA使用silu激活表示SwiGLU
    
#     # 计算每层FLOPs
#     # 注意力层FLOPs（前向）
#     attn_flops = (
#         4 * batch_size * seq_length * hidden_size**2 * (1 + num_key_value_heads/num_attention_heads) 
#         + 4 * batch_size * seq_length**2 * hidden_size
#     )
    
#     # MLP层FLOPs（前向）
#     mlp_expansion = intermediate_size / hidden_size
#     scale_factor = 3.0/2.0 if swiglu else 1.0
#     mlp_flops = 4 * mlp_expansion * scale_factor * batch_size * seq_length * hidden_size**2
    
#     # 总层FLOPs（前向）
#     total_layers_flops = num_layers * (attn_flops + mlp_flops)
    
#     # Logits FLOPs（前向）
#     logits_flops = 2 * batch_size * seq_length * hidden_size * vocab_size
    
#     # 总FLOPs（前向+反向 = 前向*3）
#     total_flops = (total_layers_flops + logits_flops) * 3
    
#     # 转换为TFLOPS
#     return total_flops
