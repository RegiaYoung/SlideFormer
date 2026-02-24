
import torch
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

'''
Model  | Part     | Mem Type | Size(MB) | H2D (ms)   | D2H (ms)   | BW H2D(GB/s) | BW D2H(GB/s)
----------------------------------------------------------------------------------------------------
3B     | Embed    | Pinned   | 593.5    | 24.449     | 23.574     | 23.71        | 24.59
3B     | Embed    | Pageable | 593.5    | 27.993     | 73.071     | 20.70        | 7.93
3B     | Decoder  | Pinned   | 147.0    | 6.060      | 5.843      | 23.69        | 24.57
3B     | Decoder  | Pageable | 147.0    | 6.974      | 18.170     | 20.58        | 7.90
7B     | Embed    | Pinned   | 1039.5   | 42.814     | 41.285     | 23.71        | 24.59
7B     | Embed    | Pageable | 1039.5   | 48.979     | 127.909    | 20.73        | 7.94
7B     | Decoder  | Pinned   | 444.5    | 18.312     | 17.657     | 23.71        | 24.58
7B     | Decoder  | Pageable | 444.5    | 20.966     | 54.749     | 20.70        | 7.93
8B     | Embed    | Pinned   | 1002.0   | 41.269     | 39.795     | 23.71        | 24.59
8B     | Embed    | Pageable | 1002.0   | 47.220     | 123.696    | 20.72        | 7.91
8B     | Decoder  | Pinned   | 416.0    | 17.138     | 16.525     | 23.71        | 24.58
8B     | Decoder  | Pageable | 416.0    | 19.631     | 51.389     | 20.69        | 7.91
14B    | Embed    | Pinned   | 1485.0   | 61.157     | 58.975     | 23.71        | 24.59
14B    | Embed    | Pageable | 1485.0   | 69.948     | 183.246    | 20.73        | 7.91
14B    | Decoder  | Pinned   | 525.0    | 21.626     | 20.854     | 23.71        | 24.59
14B    | Decoder  | Pageable | 525.0    | 24.756     | 64.633     | 20.71        | 7.93
----------------------------------------------------------------------------------------------------
Activation Checkpointing (Hidden State) Transfer Benchmark (Seq Len = 1024)
----------------------------------------------------------------------------------------------------
3B     | Act-B32  | Pinned   | 128.0    | 5.278      | 5.088      | 23.68        | 24.57
3B     | Act-B32  | Pageable | 128.0    | 6.056      | 15.839     | 20.64        | 7.89
3B     | Act-B64  | Pinned   | 256.0    | 10.547     | 10.171     | 23.70        | 24.58
3B     | Act-B64  | Pageable | 256.0    | 12.080     | 31.584     | 20.69        | 7.92
3B     | Act-B128 | Pinned   | 512.0    | 21.089     | 20.337     | 23.71        | 24.59
3B     | Act-B128 | Pageable | 512.0    | 24.148     | 63.059     | 20.71        | 7.93
7B     | Act-B32  | Pinned   | 224.0    | 9.230      | 8.900      | 23.70        | 24.58
7B     | Act-B32  | Pageable | 224.0    | 10.573     | 27.638     | 20.69        | 7.91
7B     | Act-B64  | Pinned   | 448.0    | 18.453     | 17.795     | 23.71        | 24.59
7B     | Act-B64  | Pageable | 448.0    | 21.129     | 55.225     | 20.71        | 7.92
8B     | Act-B32  | Pinned   | 256.0    | 10.548     | 10.171     | 23.70        | 24.58
8B     | Act-B32  | Pageable | 256.0    | 12.083     | 31.587     | 20.69        | 7.91
8B     | Act-B64  | Pinned   | 512.0    | 21.088     | 20.337     | 23.71        | 24.59
8B     | Act-B64  | Pageable | 512.0    | 24.148     | 63.112     | 20.71        | 7.92
14B    | Act-B32  | Pinned   | 320.0    | 13.183     | 12.713     | 23.70        | 24.58
14B    | Act-B32  | Pageable | 320.0    | 15.094     | 39.484     | 20.70        | 7.91
14B    | Act-B64  | Pinned   | 640.0    | 26.359     | 25.420     | 23.71        | 24.59
14B    | Act-B64  | Pageable | 640.0    | 30.164     | 78.894     | 20.72        | 7.92
'''

# Configuration for different model sizes
# Values extracted from config.json files in /data/home/scc/models/
MODELS_CONFIG = {
    "3B":  {"h": 2048,  "i": 11008, "v": 151936, "heads": 16, "kv_heads": 2},  # Qwen2.5-3B
    "7B":  {"h": 3584,  "i": 18944, "v": 152064, "heads": 28, "kv_heads": 4},  # Qwen2.5-7B
    "8B":  {"h": 4096,  "i": 14336, "v": 128256, "heads": 32, "kv_heads": 8},  # Llama-3.1-8B
    "14B": {"h": 5120,  "i": 13824, "v": 152064, "heads": 40, "kv_heads": 8},  # Qwen2.5-14B
}


def get_layer_size_mb_and_shape(config: Dict, layer_type: str) -> Tuple[float, int]:
    """Calculates the total number of elements for a layer and returns size in MB and numel."""
    h = config["h"]
    dtype_size = 2 # bfloat16 = 2 bytes

    total_elements = 0
    
    if layer_type == "Embed":
        v = config["v"]
        total_elements = v * h
    else: # Decoder
        i = config["i"]
        heads = config["heads"]
        kv_heads = config["kv_heads"]
        head_dim = h // heads
        
        # Self Attention: Q, K, V, O
        total_elements += h * h # Q
        total_elements += h * (kv_heads * head_dim) # K
        total_elements += h * (kv_heads * head_dim) # V
        total_elements += h * h # O
        
        # MLP: Gate, Up, Down
        total_elements += i * h # Gate
        total_elements += i * h # Up
        total_elements += h * i # Down
        
        # Layer Norms: 2 * h
        total_elements += 2 * h
        
    size_mb = (total_elements * dtype_size) / 1024 / 1024
    return size_mb, total_elements

def measure_transfer(
    size_mb: float,
    numel: int,
    direction: str, 
    use_pin_memory: bool, 
    stream: torch.cuda.Stream, 
    warmup: int = 10, 
    iters: int = 20,
    dtype = torch.bfloat16
) -> float:
    """
    Measures transfer time using pre-allocated Flat Buffers.
    direction: 'h2d' or 'd2h'
    """
    
    # 1. Allocate Buffers (Simulating Flat Buffer Architecture)
    
    # CPU Buffer
    # If use_pin_memory=False, it's standard pageable memory (OS managed)
    cpu_tensor = torch.empty(numel, dtype=dtype, device='cpu', pin_memory=use_pin_memory)
    
    # Fill with some data to avoid lazy allocation optimization issues (though empty is usually fine)
    # cpu_tensor.normal_() # Skip initialization to save time, doesn't affect transfer speed
    
    # GPU Buffer
    gpu_tensor = torch.empty(numel, dtype=dtype, device='cuda')
    
    # Determine Source and Dest based on direction
    if direction == 'h2d':
        src = cpu_tensor
        dst = gpu_tensor
    else:
        src = gpu_tensor
        dst = cpu_tensor
    
    # Warmup
    for _ in range(warmup):
        with torch.cuda.stream(stream):
             dst.copy_(src, non_blocking=True)
    
    stream.synchronize()
    torch.cuda.synchronize()
    
    # Force GC
    import gc
    gc.collect()
    
    # Measurement
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record(stream)
    
    for _ in range(iters):
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
    
    end_event.record(stream)
    end_event.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / iters
    
    # Clean up immediately to free memory for next run
    del src
    del dst
    del cpu_tensor
    del gpu_tensor
    
    return avg_time_ms

def run_benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        return
        
    print(f"Benchmarking Flat Buffer Transfer (Simulating custom framework behavior)...")
    print(f"{'Model':<6} | {'Part':<8} | {'Mem Type':<8} | {'Size(MB)':<8} | {'H2D (ms)':<10} | {'D2H (ms)':<10} | {'BW H2D(GB/s)':<12} | {'BW D2H(GB/s)':<12}")
    print("-" * 100)
    
    stream = torch.cuda.Stream()
    
    for model_name, config in MODELS_CONFIG.items():
        for part in ["Embed", "Decoder"]:
            
            # 1. Calculate Size
            size_mb, numel = get_layer_size_mb_and_shape(config, part)
            
            # 2. Test for Pin and Pageable
            for mem_type in ["Pinned", "Pageable"]:
                use_pin = (mem_type == "Pinned")
                
                try:
                    # Measure H2D
                    h2d_ms = measure_transfer(size_mb, numel, 'h2d', use_pin, stream)
                    
                    # Measure D2H
                    d2h_ms = measure_transfer(size_mb, numel, 'd2h', use_pin, stream)
                    
                    bw_h2d = (size_mb / 1024) / (h2d_ms / 1000) if h2d_ms > 0 else 0
                    bw_d2h = (size_mb / 1024) / (d2h_ms / 1000) if d2h_ms > 0 else 0
                    
                    print(f"{model_name:<6} | {part:<8} | {mem_type:<8} | {size_mb:<8.1f} | {h2d_ms:<10.3f} | {d2h_ms:<10.3f} | {bw_h2d:<12.2f} | {bw_d2h:<12.2f}")
                    
                except RuntimeError as e:
                    print(f"{model_name:<6} | {part:<8} | {mem_type:<8} | {size_mb:<8.1f} | {'OOM/Err':<10} | {'-':<10} | {'-':<12} | {'-':<12}")
                    # print(e) # Debug
                    torch.cuda.empty_cache()
                
            torch.cuda.empty_cache()

    # 3. Activation Checkpointing Benchmark
    print("-" * 100)
    print("Activation Checkpointing (Hidden State) Transfer Benchmark (Seq Len = 1024)")
    print("-" * 100)
    
    seq_len = 1024
    batch_sizes = [32, 64, 128]
    
    for model_name, config in MODELS_CONFIG.items():
        h = config["h"]
        for bs in batch_sizes:
            # Activation: [BS, Seq, Hidden] * 2 bytes (bfloat16)
            numel = bs * seq_len * h
            size_mb = (numel * 2) / 1024 / 1024
            part_label = f"Act-B{bs}"
            
            for mem_type in ["Pinned", "Pageable"]:
                use_pin = (mem_type == "Pinned")
                
                try:
                    # Measure H2D (Load/Unpack)
                    h2d_ms = measure_transfer(size_mb, numel, 'h2d', use_pin, stream)
                    
                    # Measure D2H (Offload/Pack)
                    d2h_ms = measure_transfer(size_mb, numel, 'd2h', use_pin, stream)
                    
                    bw_h2d = (size_mb / 1024) / (h2d_ms / 1000) if h2d_ms > 0 else 0
                    bw_d2h = (size_mb / 1024) / (d2h_ms / 1000) if d2h_ms > 0 else 0
                    
                    print(f"{model_name:<6} | {part_label:<8} | {mem_type:<8} | {size_mb:<8.1f} | {h2d_ms:<10.3f} | {d2h_ms:<10.3f} | {bw_h2d:<12.2f} | {bw_d2h:<12.2f}")
                    
                except RuntimeError as e:
                    print(f"{model_name:<6} | {part_label:<8} | {mem_type:<8} | {size_mb:<8.1f} | {'OOM/Err':<10} | {'-':<10} | {'-':<12} | {'-':<12}")
                    torch.cuda.empty_cache()
            
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_benchmark()
