import torch

def get_rotary_embedding(seq_length, head_dim, batch_size, base=10000):
    """计算RoPE位置编码
    
    遵循RoPE论文和LLaMA实现的标准方法:
    - 使用exp(iθm) = cos(θm) + i*sin(θm)
    - θm = m/(10000^(2k/d))，其中m是位置，k是维度索引
    
    Args:
        seq_length: 序列长度
        head_dim: attention头的维度大小
        batch_size: 批次大小
        base: 用于计算频率的基数，默认10000
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: cos和sin张量，形状为(batch_size, seq_len, head_dim)
    """
    # 1. 计算维度索引，只需要一半的维度，因为我们对复数的实部和虚部使用相同的角度
    dim_t = torch.arange(0, head_dim, 2).float()
    
    # 2. 计算频率因子 θ = 1/10000^(2k/d)
    inv_freq = 1.0 / (base ** (dim_t / head_dim))
    
    # 3. 计算位置索引 m
    t = torch.arange(seq_length, dtype=torch.float)
    
    # 4. 计算 mθ
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    
    # 5. 计算 cos(mθ) 和 sin(mθ)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()  # [seq_len, head_dim]
    sin = emb.sin()  # [seq_len, head_dim]
    
    # 6. 扩展到batch维度
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, head_dim]
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, head_dim]
    
    return cos, sin
