from ratel_init import Init, SB_hook
from ratel_optimizer import SB_optimizer
from see_mem import see_memory_usage
from nvtx import nvtx_wrap
import argparse
import torch
import torch.nn as nn
from op_ds.ops.CPUAdam import DeepSpeedCPUAdam
import torch.multiprocessing as mp
from time import time
from gpt_model import GPT2Model, GPT2Config, act_stream, set_training
from utils import priority_sort, get_act_swap_list
import psutil  # 添加psutil库用于监控CPU内存
import math
import os  # 添加os模块

def test_async(mp_queue_fp32, mp_queue_fp32_grad, mp_queue_signal, mp_queue_fp32_state_step, mp_queue_fp32_state_m, mp_queue_fp32_state_v, mp_model_parameters, mp_queue_fp32_state_id, mp_grad_event, mp_finish):
    model = mp_model_parameters.get()
    model_parameters = model.parameters()
    
    optimizer_parameters = {}
    optimizer = DeepSpeedCPUAdam(model_parameters, **optimizer_parameters, adamw_mode=False)
    
    @nvtx_wrap
    def cpu_step():
        optimizer.step()
    count = 0
    while(1):
        if not mp_finish.empty():
            if mp_finish.get() == 'finish':
                break
        if not mp_queue_signal.empty():
            temp_signal = mp_queue_signal.get()
            if temp_signal == 555:
                print('next step')

            # print(f'sub process get single {temp_signal}')

            if mp_queue_fp32_state_id.qsize():
                temp_event = mp_grad_event.get()
                # print('bef sync', temp_event.query())
                temp_event.synchronize()
                # print('aft sync',temp_event.query())
                fp32_param = mp_queue_fp32.get()
                fp32_param.grad = mp_queue_fp32_grad.get()
                optimizer.state[fp32_param]['step'] = mp_queue_fp32_state_step.get()
                optimizer.state[fp32_param]['exp_avg'] = mp_queue_fp32_state_m.get()
                optimizer.state[fp32_param]['exp_avg_sq'] = mp_queue_fp32_state_v.get()

                optimizer.param_groups[0]['params'] = [fp32_param] 

                cpu_step()

                optimizer.param_groups[0]['params'] = [] 

                count += 1
                # print('finish')

def update_memory_stats():
    """更新内存统计信息"""
    current_cpu_memory_gb = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
    current_gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    current_gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    
    return current_cpu_memory_gb, current_gpu_allocated_mb, current_gpu_reserved_mb

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

if __name__ == '__main__':
    # 多进程初始化
    mp.set_start_method('spawn', force=True)
    mp_queue_fp32 = mp.Queue()
    mp_queue_fp32_grad = mp.Queue()
    mp_queue_signal = mp.Queue()
    mp_queue_fp32_state_step = mp.Queue()
    mp_queue_fp32_state_m = mp.Queue()
    mp_queue_fp32_state_v = mp.Queue()
    mp_queue_fp32_state_id = mp.Queue()
    mp_model_parameters = mp.Queue()
    mp_grad_event = mp.Queue()
    mp_finish = mp.Queue()
    mp_list = []
    mp_list.append(mp_queue_fp32)
    mp_list.append(mp_queue_fp32_grad)
    mp_list.append(mp_queue_signal)
    mp_list.append(mp_queue_fp32_state_step)
    mp_list.append(mp_queue_fp32_state_m)
    mp_list.append(mp_queue_fp32_state_v)
    mp_list.append(mp_queue_fp32_state_id)
    mp_list.append(mp_grad_event)

    ## 解析参数
    # 解析模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=5120, help="hidden dimension of transformer model")
    parser.add_argument("--num_heads", type=int, default=80, help="number of attention heads in transformer model")
    parser.add_argument("--num_layers", type=int, default=40, help="number of layers in transformer model")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="max sequence length")
    parser.add_argument("--vocab_size", type=int, default=50257, help="vocabulary size")
    
    # 解析swap和重计算配置
    parser.add_argument("--is_swap_and_recompute", type=int, default=0, help="whether to use swap and recompute")
    parser.add_argument("--is_swap_prior", type=int, default=1, help="whether to consider swap prioritization")
    parser.add_argument("--is_fully_swap", type=int, default=0, help="whether to fully swap")
    parser.add_argument("--swap_ratio", type=float, default=0.2, help="swap ratio")

    # 解析异步和nvme配置
    parser.add_argument("--is_new_param_async", type=int, default=1, help="whether parameters are transmitted asynchronously")
    parser.add_argument("--is_grad_async", type=int, default=1, help="whether gradient are transmitted asynchronously")
    parser.add_argument("--is_mp", type=int, default=1, help="whether to use multiprocessing")
    parser.add_argument("--is_nvme", type=int, default=1, help="whether to offload to nvme")
    parser.add_argument("--is_nvme_async", type=int, default=1, help="whether to offload to nvme asynchronously")
    parser.add_argument("--is_nvme_rearrange", type=int, default=1, help="whether to reprogram nvme communications")

    parser.add_argument("--sb_config", type=str, default='/home/lcy/flush/Ratel_Private/config.json', help="config path")
    args = parser.parse_args()

    assert args.hidden_dim % args.num_heads == 0
    args.dim_head = args.hidden_dim // args.num_heads

    # 初始化模型config
    config = GPT2Config(
        dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_head=args.dim_head,
        max_seq_len=args.max_seq_len,
        attn_pdrop=0.1,
        dropout=0.1,
        vocab_size=args.vocab_size,
        layer_norm_epsilon=1e-5,
    )

    set_training(args)

    # 初始化矩阵乘激活值的优先级
    act_list = [i for i in range(4 * config.num_layers)]
    if args.is_swap_prior:
        act_priority = priority_sort(act_list)
    else:
        act_priority = act_list
    act_pack = {}
    print(act_priority)

    # 初始化模型，SSD-CPU-GPU三级存储初始化，参数属性改造
    see_memory_usage("before act ini")
    fw_time = []
    swap_list = []
    see_memory_usage("before model init")
    with Init(is_nvme=args.is_nvme, is_nvme_async=args.is_nvme_async, config=args.sb_config):
        model = GPT2Model(config).half()

    # 多进程初始化
    if args.is_mp:
        model.share_memory()
        mp_model_parameters.put(model)
        p1 = mp.Process(target = test_async, args=(mp_queue_fp32, mp_queue_fp32_grad, mp_queue_signal, mp_queue_fp32_state_step, mp_queue_fp32_state_m, mp_queue_fp32_state_v, mp_model_parameters, mp_queue_fp32_state_id, mp_grad_event, mp_finish))
        p1.start()

    # Hook逻辑，实现参数异步预取和释放
    SB_hook(model, args.is_new_param_async, fw_time=fw_time, is_swap_and_recompute=args.is_swap_and_recompute)

    # 初始化输入和target, loss_fn
    # input_data = torch.randint(0, args.vocab_size, (args.max_seq_len, args.batch_size))
    input_data = torch.randint(0, args.vocab_size, (args.batch_size, args.max_seq_len))
    input_data = input_data.to('cuda')
    # target = torch.randn(args.max_seq_len, args.batch_size, args.hidden_dim, dtype=torch.float16)
    target = torch.roll(input_data, shifts=-1, dims=1)
    target[:, -1] = -100
    target = target.to('cuda')
    # loss_fn = nn.MSELoss()
    # For fair
    loss_fn = nn.CrossEntropyLoss()

    # 初始化CPU Adam，和优化器相关
    model_parameters = model.parameters()
    optimizer_parameters = {}
    optimizer = DeepSpeedCPUAdam(model_parameters,
                                        **optimizer_parameters,
                                        adamw_mode=False)
    
    # 改造优化器，实现异步梯度卸载和异步优化器更新
    optimizer = SB_optimizer(optimizer, args.is_mp, mp_list = mp_list, is_nvme=args.is_nvme, is_grad_async=args.is_grad_async, is_nvme_async=args.is_nvme_async, is_nvme_rearrange=args.is_nvme_rearrange, config=args.sb_config)

    # 添加性能统计变量
    iter_times = []
    tokens_per_sec = []
    max_cpu_memory_mb = 0
    max_gpu_allocated_mb = 0
    max_gpu_reserved_mb = 0
    
    # 重置CUDA内存峰值统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    event_list = []
    for i in range(4):
        iter_start = time()
        print(f'-----------------------Iter {i}-----------------------')
        print('---begin forward---')
        torch.cuda.nvtx.range_push("iteration")
        
        torch.cuda.nvtx.range_push("forward")
        output = model(input_data, swap_list, act_pack)
        torch.cuda.nvtx.range_pop()

        # 自动调度swap和重计算
        if i == 0 and args.is_swap_and_recompute:
            get_act_swap_list(fw_time, args, swap_list, act_pack, act_priority)

        torch.cuda.current_stream().synchronize()
        act_stream.synchronize()
        forward_end = time()
        print('forward time', forward_end - iter_start)
        # loss = loss_fn(output, target)
        loss = loss_fn(output.view(-1, args.vocab_size), target.view(-1))
        
        # 更新内存统计
        cpu_mem, gpu_alloc, gpu_reserved = track_memory_usage()
        max_cpu_memory_mb = max(max_cpu_memory_mb, cpu_mem)
        max_gpu_allocated_mb = max(max_gpu_allocated_mb, gpu_alloc)
        max_gpu_reserved_mb = max(max_gpu_reserved_mb, gpu_reserved)
        
        print('---begin backward---')
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        optimizer.independent_gradient_partition_epilogue()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("optimizer")

        if not args.is_mp and not args.is_nvme_async:
            optimizer.step()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        global_back_id = 0
        event_list = []
        global_flag_id = 0
        torch.cuda.current_stream().synchronize()
        print('back_and_opt time', time() - forward_end)
        
        # 优化器步骤后更新内存统计
        cpu_mem, gpu_alloc, gpu_reserved = track_memory_usage()
        max_cpu_memory_mb = max(max_cpu_memory_mb, cpu_mem)
        max_gpu_allocated_mb = max(max_gpu_allocated_mb, gpu_alloc)
        max_gpu_reserved_mb = max(max_gpu_reserved_mb, gpu_reserved)  
        
        # 计算迭代时间并记录
        iter_time = time() - iter_start
        iter_times.append(iter_time)
        
        # 计算tokens per second
        # 每次迭代处理的tokens数量 = batch_size * sequence_length
        total_tokens = args.batch_size * args.max_seq_len
        tokens_per_sec.append(total_tokens / iter_time)
        
        print(f'Iteration {i} time: {iter_time:.4f} s, tokens/s: {total_tokens / iter_time:.2f}')
    
    torch.cuda.current_stream().synchronize()
    mp_finish.put('finish')
    if args.is_mp:
        p1.join()
        
    # 打印性能统计信息
    print("\n" + "="*50)
    print("Performance Summary:")
    print("="*50)
    
    # 基本配置信息
    print(f"Model Configuration:")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Sequence Length: {args.max_seq_len}")
    print(f"  - Hidden Dimension: {args.hidden_dim}")
    print(f"  - Number of Heads: {args.num_heads}")
    print(f"  - Number of Layers: {args.num_layers}")
    
    # 计算后三个迭代的平均性能（如果有足够的迭代）
    if len(iter_times) >= 3:
        last_three_iter_times = iter_times[-3:]
        last_three_tokens_per_sec = tokens_per_sec[-3:]
        avg_iter_time = sum(last_three_iter_times) / len(last_three_iter_times)
        avg_tokens_per_sec = sum(last_three_tokens_per_sec) / len(last_three_tokens_per_sec)
        
        print(f"\nLast 3 Iterations Average Performance:")
        print(f"  - Average Iteration Time: {avg_iter_time:.4f} s")
        print(f"  - Average Tokens/s: {avg_tokens_per_sec:.2f}")
    else:
        print("\nNot enough iterations for calculating 3-iteration average")
    
    # 内存使用统计
    print(f"\nMemory Usage:")
    print(f"  - Max CPU Memory: {max_cpu_memory_mb / 1024:.2f} GB")
    print(f"  - Max CUDA Memory Allocated: {max_gpu_allocated_mb:.2f} MB")
    print(f"  - Max CUDA Memory Reserved: {max_gpu_reserved_mb:.2f} MB")
    print("="*50)



