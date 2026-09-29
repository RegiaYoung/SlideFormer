import torch

# 添加内存监控
def log_memory_stats():
    gpu_allocated = torch.cuda.memory_allocated() / 1024**2
    gpu_reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    max_reserved = torch.cuda.max_memory_reserved() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    return f"GPU Memory: {gpu_allocated:.1f}MB allocated, {max_allocated:.1f}MB Max allocated, {gpu_reserved:.1f}MB reserved, {max_reserved:.1f}MB Max Reserved."
