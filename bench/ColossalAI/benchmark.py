import argparse
import resource
import time
import warnings
from contextlib import nullcontext
import numpy as np
import os
import csv
import torch
import psutil
import torch.distributed as dist
from data_utils import RandomDataset
from model_utils import format_numel_str, get_model_numel, calculate_flops_per_batch
from performance_evaluator import PerformanceEvaluator, get_profile_context
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, TorchFSDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.pipeline.schedule.v_schedule import PipelineGraph
from colossalai.shardformer import PipelineGradientCheckpointConfig

warnings.filterwarnings("ignore")

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

def append_result_to_csv(result_file, result_dict):
    """将结果追加到CSV文件中"""
    # 检查文件是否存在，如果不存在则创建并写入头部
    file_exists = os.path.isfile(result_file)
    
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Sequence_Length', 'Batch_Size',
                      'Avg_Time', 'Avg_Tokens_Per_Second', 'Avg_TFLOPS', 
                      'Max_Memory_MB', 'Max_CUDA_Memory_Allocated_MB', 'Max_CUDA_Memory_Reserved_MB']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Model': result_dict['model'],
            'Sequence_Length': result_dict['seq_len'],
            'Batch_Size': result_dict['batch_size'],
            'Avg_Time': f"{result_dict['avg_time']:.4f}",
            'Avg_Tokens_Per_Second': f"{result_dict['avg_tokens_per_sec']:.1f}",
            'Avg_TFLOPS': f"{result_dict['avg_tflops']:.2f}",
            'Max_Memory_MB': f"{result_dict['max_memory_mb']:.1f}",
            'Max_CUDA_Memory_Allocated_MB': f"{result_dict['max_cuda_memory_allocated_mb']:.1f}",
            'Max_CUDA_Memory_Reserved_MB': f"{result_dict['max_cuda_memory_reserved_mb']:.1f}"
        })

# ==============================
# Constants
# ==============================

# We have lots of llamas for your choice!
# MODEL_CONFIGS = {
#     "100m": LlamaConfig(
#         max_position_embeddings=4096,
#         num_hidden_layers=4,
#         num_attention_heads=32,
#         intermediate_size=2048,
#         hidden_size=1024,
#     ),
#     "5b": LlamaConfig(max_position_embeddings=4096, num_key_value_heads=8),
#     "7b": LlamaConfig(max_position_embeddings=4096),
#     # "7b": LlamaConfig(num_hidden_layers=4, max_position_embeddings=4096),
#     "13b": LlamaConfig(
#         hidden_size=5120,
#         intermediate_size=13824,
#         num_hidden_layers=40,
#         num_attention_heads=40,
#         max_position_embeddings=4096,
#     ),
#     "70b": LlamaConfig(
#         hidden_size=8192,
#         intermediate_size=28672,
#         num_hidden_layers=80,
#         num_attention_heads=64,
#         max_position_embeddings=4096,
#         num_key_value_heads=8,
#     ),
# }

MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=False,  # 设为False以兼容gradient checkpointing
        bos_token_id=128000,
        eos_token_id=128001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling={"type": "linear", "factor": 8.0},
        attention_bias=False,
        attention_dropout=0.0,
    ),
    "Mistral-Small-24B-Instruct": LlamaConfig(
        vocab_size=131072,
        hidden_size=5120,
        intermediate_size=32768,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=False,  # 设为False以兼容gradient checkpointing
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=100000000.0,
        attention_bias=False,
        attention_dropout=0.0,
    ),
    "Qwen2.5-3B-Instruct": LlamaConfig(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,  # 设为False以兼容gradient checkpointing
        bos_token_id=151643,
        eos_token_id=151645,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        rope_scaling={"type": "linear", "factor": 4.0},
        attention_bias=False,
        attention_dropout=0.0,
    ),
    "Qwen2.5-7B-Instruct": LlamaConfig(
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,  # 设为False以兼容gradient checkpointing
        bos_token_id=151643,
        eos_token_id=151645,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling={"type": "linear", "factor": 4.0},
        attention_bias=False,
        attention_dropout=0.0,
    ),
    "Qwen2.5-14B-Instruct": LlamaConfig(
        vocab_size=152064,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,  # 设为False以兼容gradient checkpointing
        bos_token_id=151643,
        eos_token_id=151645,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling={"type": "linear", "factor": 4.0},
        attention_bias=False,
        attention_dropout=0.0,
    ),
    "Qwen2.5-32B-Instruct": LlamaConfig(
        vocab_size=152064,
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,  # 设为False以兼容gradient checkpointing
        bos_token_id=151643,
        eos_token_id=151645,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling={"type": "linear", "factor": 4.0},
        attention_bias=False,
        attention_dropout=0.0,
    ),
    "Qwen2.5-72B-Instruct": LlamaConfig(
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,  # 设为False以兼容gradient checkpointing
        bos_token_id=151643,
        eos_token_id=151645,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling={"type": "linear", "factor": 4.0},
        attention_bias=False,
        attention_dropout=0.0,
    ),
}


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="7b", help="Model configuration")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["gemini", "gemini_auto", "fsdp", "fsdp_cpu", "3d", "3d_cpu"],
        default="gemini",
        help="Choose which plugin to use",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-s", "--num_steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("-i", "--ignore_steps", type=int, default=2, help="Number of steps to ignore")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument(
        "-w", "--warmup_ratio", type=float, default=0.8, help="warm up ratio of non-model data. Only for gemini-auto"
    )
    parser.add_argument("-m", "--memory_limit", type=int, help="Gemini memory limit in mb")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use xformers")
    parser.add_argument("--shard_param_frac", type=float, default=1.0, help="Shard param fraction. Only for gemini")
    parser.add_argument("--offload_optim_frac", type=float, default=0.0, help="Offload optim fraction. Only for gemini")
    parser.add_argument("--offload_param_frac", type=float, default=0.0, help="Offload param fraction. Only for gemini")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--sp", type=int, default=1, help="Sequence parallel size")
    parser.add_argument("--extra_dp", type=int, default=1, help="Extra data parallel size, used for Gemini")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size of pipeline parallel")
    parser.add_argument("--zero", type=int, default=0, help="Zero Stage when hybrid plugin is enabled")
    parser.add_argument("--custom-ckpt", action="store_true", help="Customize checkpoint", default=False)

    parser.add_argument("--pp_style", default="1f1b", choices=["1f1b", "interleaved", "zbv"])
    parser.add_argument("--n_chunks", default=1, help="number of model chunks", type=eval)
    parser.add_argument("--profile", action="store_true", help="Profile the code")
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Use nsys for profiling. \
        You should put something like this before colossalai launch: \
        nsys profile -w true -t cuda,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --python-backtrace=cuda -o prof_out",
    )
    parser.add_argument("--disable-async-reduce", action="store_true", help="Disable the asynchronous reduce operation")
    parser.add_argument("--prefetch_num", type=int, default=0, help="chunk prefetch max number")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--use_fp8_comm", action="store_true", default=False, help="for using fp8 during communication")
    parser.add_argument("--use_fp8", action="store_true", default=False, help="for using fp8 linear")
    parser.add_argument("--overlap_p2p", action="store_true", default=True, help="for using overlap p2p")
    parser.add_argument("--overlap_allgather", action="store_true")
    parser.add_argument(
        "--sp_mode",
        default="all_to_all",
        choices=["all_to_all", "ring_attn", "ring", "split_gather"],
        help="Sequence parallelism mode",
    )
    parser.add_argument("--result_file", type=str, default=None, help="CSV file to save the results")
    args = parser.parse_args()

    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    def empty_init():
        pass

    # ckpt config for LLaMA3-70B on 64 H100 GPUs
    hybrid_kwargs = (
        {
            "gradient_checkpoint_config": PipelineGradientCheckpointConfig(
                num_ckpt_layers_per_stage=[19, 19, 19, 13],
            ),
            "num_layers_per_stage": [19, 20, 20, 21],
            "pp_style": "interleaved",
        }
        if args.custom_ckpt
        else {}
    )

    # ==============================
    # Initialize Booster
    # ==============================
    if args.config in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.config]
    else:
        config = AutoConfig.from_pretrained(args.config, trust_remote_code=True)

    use_empty_init = True
    if args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision="bf16",
            shard_param_frac=args.shard_param_frac,
            offload_optim_frac=args.offload_optim_frac,
            offload_param_frac=args.offload_param_frac,
            tp_size=args.tp,
            extra_dp_size=args.extra_dp,
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=args.xformers,
            max_prefetch=args.prefetch_num,
            enable_async_reduce=not args.disable_async_reduce,
            use_fp8=args.use_fp8,
            fp8_communication=args.use_fp8_comm,
            pin_memory=True
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            placement_policy="auto",
            precision="bf16",
            warmup_non_model_data_ratio=args.warmup_ratio,
            tp_size=args.tp,
            extra_dp_size=args.extra_dp,
            enable_fused_normalization=get_accelerator().is_available(),
            max_prefetch=args.prefetch_num,
            enable_async_reduce=not args.disable_async_reduce,
            enable_flash_attention=args.xformers,
            use_fp8=args.use_fp8,
            fp8_communication=args.use_fp8_comm,
            pin_memory=True
        )
    elif args.plugin == "fsdp":
        if use_empty_init:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                ),
                param_init_fn=empty_init(),
                fp8_communication=args.use_fp8_comm,
            )
        else:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                ),
                fp8_communication=args.use_fp8_comm,
            )
    elif args.plugin == "fsdp_cpu":
        if use_empty_init:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                ),
                cpu_offload=CPUOffload(offload_params=True),
                param_init_fn=empty_init(),
                fp8_communication=args.use_fp8_comm,
            )
        else:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                ),
                cpu_offload=CPUOffload(offload_params=True),
                fp8_communication=args.use_fp8_comm,
            )
    elif args.plugin == "3d":
        if args.pp_style == "zbv":
            mem_f = 34 * config.hidden_size + 5 * config.num_attention_heads * args.max_length
            mem_w = -32 * config.hidden_size
            mem_b = -mem_w - mem_f
            scheduler_nodes = PipelineGraph(
                n_stage=args.pp,
                n_micro=args.batch_size // args.mbs,
                f_cost=1000,
                b_cost=1000,
                w_cost=1000,
                c_cost=1,
                f_mem=mem_f * 1.5,
                b_mem=mem_b * 1.5,
                w_mem=mem_w * 1.5,
            ).get_v_schedule()
        else:
            scheduler_nodes = None

        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            pp_style=args.pp_style,
            num_model_chunks=args.n_chunks,
            zero_stage=args.zero,
            sp_size=args.sp,
            sequence_parallelism_mode=args.sp_mode,
            enable_sequence_parallelism=args.sp > 1,
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=args.xformers,
            microbatch_size=args.mbs,
            precision="bf16",
            enable_metadata_cache=not args.no_cache,
            overlap_allgather=args.overlap_allgather,
            use_fp8=args.use_fp8,
            fp8_communication=args.use_fp8_comm,
            scheduler_nodes=scheduler_nodes,
            **hybrid_kwargs,
        )
    elif args.plugin == "3d_cpu":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            pp_style=args.pp_style,
            num_model_chunks=args.n_chunks,
            zero_stage=args.zero,
            cpu_offload=True,
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=args.xformers,
            microbatch_size=args.mbs,
            initial_scale=2**8,
            precision="bf16",
            overlap_p2p=args.overlap_p2p,
            use_fp8=args.use_fp8,
            fp8_communication=args.use_fp8_comm,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Dataset and Dataloader
    # ==============================
    dp_size = getattr(plugin, "dp_size", coordinator.world_size)

    if args.config in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.config]
    else:
        config = AutoConfig.from_pretrained(args.config, trust_remote_code=True)
    get_accelerator().manual_seed(42)

    dataset = RandomDataset(
        num_samples=args.batch_size * args.num_steps * dp_size, max_length=args.max_length, vocab_size=config.vocab_size
    )
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, seed=42)

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    init_ctx = (
        LazyInitContext(default_device=get_accelerator().get_current_device())
        if isinstance(plugin, (GeminiPlugin, HybridParallelPlugin))
        else nullcontext()
    )
    init_kwargs = {}
    if config.model_type == "chatglm":
        init_kwargs["empty_init"] = False

    with init_ctx:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            **init_kwargs,
            torch_dtype=torch.bfloat16,
        )
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        if config.model_type == "chatglm":
            model.transformer.encoder.gradient_checkpointing = True

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")
    if config.model_type == "chatglm":
        num_layers = model.config.num_layers
    else:
        num_layers = model.config.num_hidden_layers
    # performance_evaluator = PerformanceEvaluator(
    #     model_numel,
    #     num_layers,
    #     model.config.hidden_size,
    #     model.config.vocab_size,
    #     args.grad_checkpoint,
    #     args.ignore_steps,
    #     dp_world_size=dp_size,
    # )

    optimizer = HybridAdam(model.parameters(), weight_decay=0.1)
    torch.set_default_dtype(torch.bfloat16)
    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)

    torch.set_default_dtype(torch.float)
    coordinator.print_on_master(
        f"Booster init max device memory: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB"
    )
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )
    
    # 计算FLOPS
    total_flops = calculate_flops_per_batch(config, args.batch_size, args.max_length)
    tokens_per_batch = args.batch_size * args.max_length
    
    # 内存跟踪变量初始化
    max_memory_mb = 0
    max_cuda_memory_allocated_mb = 0
    max_cuda_memory_reserved_mb = 0

    # 统计数据初始化
    tokens_per_sec_list = []
    tflops_list = []
    batch_time_list = []
    warm_steps = args.ignore_steps
    test_steps = args.num_steps - warm_steps
    print(f"Bench Begin: {warm_steps} warmup steps and {test_steps} test steps")
    
    with get_profile_context(
        args.profile,
        args.ignore_steps,
        1,  # avoid creating massive log files
        save_dir=f"./profile/{time.strftime('%H:%M', time.localtime())}-{args.plugin}-llama-{args.config}",
        nsys=args.nsys,
    ) as prof:
        if isinstance(plugin, HybridParallelPlugin) and args.pp > 1:
            data_iter = iter(dataloader)
            for step in tqdm(range(len(dataloader)), desc="Step", disable=not coordinator.is_master()):
                # performance_evaluator.on_step_start(step)
                outputs = booster.execute_pipeline(
                    data_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs[0],
                    optimizer=optimizer,
                    return_loss=True,
                )
                loss = outputs["loss"]
                if args.pp_style == "zbv":
                    if coordinator.is_master():
                        print(f"Step {step} loss: {loss}")
                else:
                    if coordinator.is_last_process():
                        print(f"Step {step} loss: {loss}")
                optimizer.step()
                optimizer.zero_grad()

                # performance_evaluator.on_step_end(input_ids=torch.empty(args.batch_size, args.max_length))
                prof.step()
        else:
            for step, batch in enumerate(tqdm(dataloader, desc="Step", disable=not coordinator.is_master())):
                # performance_evaluator.on_step_start(step)
                step_start = time.perf_counter()
                
                outputs = model(**batch)
                loss = outputs[0]
                del outputs  # free memory

                if dist.get_rank() == dist.get_world_size() - 1:
                    print(f"Step {step} loss: {loss}")
                booster.backward(loss, optimizer)
                optimizer.step()
                optimizer.zero_grad()
                
                current_memory_mb, current_cuda_memory_allocated_mb, current_cuda_memory_reserved_mb = track_memory_usage()
                max_memory_mb = max(max_memory_mb, current_memory_mb)
                max_cuda_memory_allocated_mb = max(max_cuda_memory_allocated_mb, current_cuda_memory_allocated_mb)
                max_cuda_memory_reserved_mb = max(max_cuda_memory_reserved_mb, current_cuda_memory_reserved_mb)

                # 计算统计数据
                step_time = time.perf_counter() - step_start
                tokens_per_sec = tokens_per_batch / step_time
                tflops = total_flops / (10**12 * step_time)
                
                # 记录测试阶段的数据
                if step >= warm_steps:
                    tokens_per_sec_list.append(tokens_per_sec)
                    tflops_list.append(tflops)
                    batch_time_list.append(step_time)
                    
                    print(f"Step {step+1}/{test_steps}, Time: {step_time:.4f}s, "
                        f"Speed: {tokens_per_sec:.1f} tokens/s, "
                        f"TFLOPS: {tflops:.2f}")
                else:
                    print(f"Warmup {step+1}/{warm_steps}")
                
                step += 1
                if step >= warm_steps + test_steps:
                    break

                # performance_evaluator.on_step_end(**batch)
                prof.step()
                
    # 计算平均值
    avg_tokens_per_sec = np.mean(tokens_per_sec_list)
    avg_tflops = np.mean(tflops_list)
    avg_batch_time = np.mean(batch_time_list)
    
    # 打印结果摘要
    print("\n========== Benchmark Result ==========")
    print(f"Model: {args.config}")
    print(f"Seq_len: {args.max_length}")
    print(f"Batch_size: {args.batch_size}")
    print(f"Average batch time: {avg_batch_time:.4f} s")
    print(f"Average speed: {avg_tokens_per_sec:.1f} tokens/s")
    print(f"Average TFLOPS: {avg_tflops:.2f}")
    print(f"Max Memory: {max_memory_mb:.1f} MB")
    print(f"Max CUDA Memory Allocated: {max_cuda_memory_allocated_mb:.1f} MB")
    print(f"Max CUDA Memory Reserved: {max_cuda_memory_reserved_mb:.1f} MB")
    print("===================================\n")
    
    # 准备结果
    result = {
        "model": args.config,
        "seq_len": args.max_length,
        "batch_size": args.batch_size,
        "avg_time": avg_batch_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "avg_tflops": avg_tflops,
        "max_memory_mb": max_memory_mb,
        "max_cuda_memory_allocated_mb": max_cuda_memory_allocated_mb,
        "max_cuda_memory_reserved_mb": max_cuda_memory_reserved_mb
    }
    
    # 如果提供了结果文件，将结果追加到CSV
    if args.result_file:
        append_result_to_csv(args.result_file, result)
        print(f"结果已追加到文件: {args.result_file}")
        
    # performance_evaluator.on_fit_end()
    coordinator.print_on_master(f"Max device memory usage: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB")

if __name__ == "__main__":
    main()
