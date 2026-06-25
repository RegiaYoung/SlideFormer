import torch
from logger import logger
import gc
import psutil

torch_memory_reserved = torch.cuda.memory_reserved
torch_max_memory_reserved = torch.cuda.max_memory_reserved

def see_memory_usage(message, force=True):

    if not force:
        return

    # 手动垃圾回收
    gc.collect()
    # 输出GPU信息
    logger.info(message)
    logger.info(f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    # 输出CPU信息
    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()