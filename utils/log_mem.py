import torch
# from logger import logger

# 添加内存监控
def log_memory_stats():
    gpu_allocated = torch.cuda.memory_allocated() / 1024**2
    gpu_reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    max_reserved = torch.cuda.max_memory_reserved() / 1024**2
    # used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    torch.cuda.reset_peak_memory_stats()
    return f"GPU Memory: {gpu_allocated:.1f}MB allocated, {max_allocated:.1f}MB Max allocated, {gpu_reserved:.1f}MB reserved, {max_reserved:.1f}MB Max Reserved."



# def log_memory_stats(message, force=True):

#     if not force:
#         return

#     # 手动垃圾回收
#     gc.collect()
#     # 输出GPU信息
#     logger.info(message)
#     logger.info(f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
#         Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
#         CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
#         Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

#     # 输出CPU信息
#     vm_stats = psutil.virtual_memory()
#     used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
#     logger.info(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

#     # get the peak memory to report correct data, so reset the counter for the next call
#     torch.cuda.reset_peak_memory_stats()