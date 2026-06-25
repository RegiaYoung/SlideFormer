#!/bin/bash

# 主要参数分类
# 1. 模型参数
# --hidden_dim：隐藏层维度，例如5120表示使用5120维的隐藏层
# --num_heads：注意力头数量，例如80表示使用80个注意力头
# --num_layers：模型层数，例如40表示模型有40层
# --batch_size：训练批次大小
# --max_seq_len：最大序列长度，默认1024
# --vocab_size：词汇表大小，默认50257
# 2. 激活值交换和重计算参数
# --is_swap_and_recompute：是否启用激活值交换和重计算机制（0关闭，1开启）
# --is_swap_prior：是否使用优先级排序策略进行激活值交换（1表示使用）
# --is_fully_swap：是否交换所有激活值（0部分交换，1全部交换）
# --swap_ratio：激活值交换比例（如0.3表示交换30%的激活值）
# 3. 异步和存储配置
# --is_new_param_async：参数是否异步传输（1开启）
# --is_grad_async：梯度是否异步传输（1开启）
# --is_mp：是否使用多进程（1开启）
# --is_nvme：是否使用NVMe存储（1开启）
# --is_nvme_async：NVMe操作是否异步（1开启）
# --is_nvme_rearrange：是否重新安排NVMe通信（1开启）
# --sb_config：配置文件路径，包含详细的NVMe存储设置

# {
#   "zero_config": {
#     "offload_optimizer": {...},  // 优化器状态卸载设置
#     "offload_param": {...},      // 参数卸载设置
#     "offload_act": {...}         // 激活值卸载设置
#   },
#   "aio_config": {
#     "block_size": 1048576,      // IO块大小
#     "queue_depth": 32,          // IO队列深度
#     "thread_count": 2           // 处理IO的线程数
#     // 其他异步IO参数
#   }
# }

## for single GPU
python main.py \
    --hidden_dim 5120 \
    --num_heads 80 \
    --num_layers  40 \
    --batch_size  32\
    --is_swap_and_recompute  1 \
    --is_swap_prior  1 \
    --is_fully_swap  0 \
    --swap_ratio 0.0 \
    --is_new_param_async  1 \
    --is_grad_async  1 \
    --is_mp  1 \
    --is_nvme  0 \
    --is_nvme_async  0 \
    --is_nvme_rearrange  0 \
    --sb_config ./config.json

## for multi-GPU
## use CUDA_VISIBLE_DEVICES to set the GPU to be used
# CUDA_VISIBLE_DEVICES=0,1 \
#     torchrun main.py \
#     --hidden_dim 5120 \
#     --num_heads 80 \
#     --num_layers  40 \
#     --batch_size  64 \
#     --is_swap_and_recompute  0 \
#     --is_swap_prior  1 \
#     --is_fully_swap  0 \
#     --swap_ratio 0.8 \
#     --is_new_param_async  1 \
#     --is_grad_async  1 \
#     --is_mp  1 \
#     --is_nvme  1 \
#     --is_nvme_async  1 \
#     --is_nvme_rearrange  1 \
#     --sb_config /home/xiejun/Ratel_Private/config.json