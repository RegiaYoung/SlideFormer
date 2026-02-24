#!/usr/bin/env python3
"""
CPU Adam 性能基准测试程序

测试 LayerAdam (CPU Adam) 在不同配置下的性能：
- 不同模型大小: 3B, 8B, 14B, 32B, 72B
- 不同线程数/核数
- NUMA 拓扑感知 (内存和CPU不在同一NUMA的情况)

使用方法:
    # 基础测试
    python cpu_adam_benchmark.py
    
    # 指定线程数范围
    python cpu_adam_benchmark.py --threads 1,2,4,8,16,32
    
    # 指定模型
    python cpu_adam_benchmark.py --models 8B,14B
    
    # NUMA感知测试
    numactl --cpunodebind=0 --membind=0 python cpu_adam_benchmark.py  # 同一NUMA
    numactl --cpunodebind=0 --membind=1 python cpu_adam_benchmark.py  # 跨NUMA
    
    # 启用详细profiling
    python cpu_adam_benchmark.py --profile
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

import torch

# 添加项目根目录到路径
# 自动寻找项目根目录（通过查找 optimizer 模块）
current_dir = Path(__file__).resolve().parent
project_root = current_dir
found = False
while project_root != project_root.parent:
    if (project_root / 'optimizer').exists() or (project_root / 'optimizer.py').exists():
        found = True
        break
    project_root = project_root.parent

if not found:
    # Fallback to default relative path if not found (e.g. if run from a weird content)
    project_root = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(project_root))

from optimizer import LayerAdam


# ============================================================================
# 模型配置定义
# ============================================================================

@dataclass
class ModelConfig:
    """模型配置，用于模拟不同大小的LLM"""
    name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int = 128256  # Llama 3 default
    
    @property
    def layer_params(self) -> int:
        """计算每层的参数量 (decoder layer)"""
        # Self-attention: Q, K, V, O projections
        # Q: hidden_size * hidden_size
        # K: hidden_size * (hidden_size // num_attention_heads * num_kv_heads)
        # V: hidden_size * (hidden_size // num_attention_heads * num_kv_heads)
        # O: hidden_size * hidden_size
        head_dim = self.hidden_size // self.num_attention_heads
        q_proj = self.hidden_size * self.hidden_size
        k_proj = self.hidden_size * (head_dim * self.num_kv_heads)
        v_proj = self.hidden_size * (head_dim * self.num_kv_heads)
        o_proj = self.hidden_size * self.hidden_size
        attention_params = q_proj + k_proj + v_proj + o_proj
        
        # MLP: gate_proj, up_proj, down_proj (SwiGLU)
        gate_proj = self.hidden_size * self.intermediate_size
        up_proj = self.hidden_size * self.intermediate_size
        down_proj = self.intermediate_size * self.hidden_size
        mlp_params = gate_proj + up_proj + down_proj
        
        # LayerNorm: input_layernorm, post_attention_layernorm
        # 每个 LayerNorm 有 hidden_size 个参数 (只有 weight, 无 bias)
        layernorm_params = self.hidden_size * 2
        
        return attention_params + mlp_params + layernorm_params
    
    @property
    def total_params(self) -> int:
        """计算总参数量"""
        # Embedding: vocab_size * hidden_size
        embed_params = self.vocab_size * self.hidden_size
        # Decoder layers
        decoder_params = self.num_layers * self.layer_params
        # Final LayerNorm
        final_norm = self.hidden_size
        # LM head (通常与 embedding 共享，但我们单独计算)
        lm_head = self.vocab_size * self.hidden_size
        
        return embed_params + decoder_params + final_norm + lm_head
    
    @property
    def total_params_b(self) -> float:
        """总参数量（单位：B）"""
        return self.total_params / 1e9
    
    def estimate_optimizer_memory_gb(self) -> float:
        """估算优化器状态所需内存（GB）
        
        Adam 优化器状态包括：
        - 参数本身 (FP32): total_params * 4 bytes
        - exp_avg (FP32): total_params * 4 bytes  
        - exp_avg_sq (FP32): total_params * 4 bytes
        总计: total_params * 12 bytes
        """
        return (self.total_params * 12) / (1024 ** 3)  # GB


# 常见模型配置
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "3B": ModelConfig(
        name="3B (Llama-3.2-3B style)",
        num_layers=28,
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=24,
        num_kv_heads=8,
    ),
    "8B": ModelConfig(
        name="8B (Llama-3.1-8B style)",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_kv_heads=8,
    ),
    "14B": ModelConfig(
        name="14B (Qwen2.5-14B style)",
        num_layers=40,
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_kv_heads=8,
    ),
    "32B": ModelConfig(
        name="32B (Qwen2.5-32B style)",
        num_layers=64,
        hidden_size=5120,
        intermediate_size=27648,
        num_attention_heads=40,
        num_kv_heads=8,
    ),
    "72B": ModelConfig(
        name="72B (Qwen2.5-72B style)",
        num_layers=80,
        hidden_size=8192,
        intermediate_size=29568,
        num_attention_heads=64,
        num_kv_heads=8,
    ),
}


# ============================================================================
# 系统信息获取
# ============================================================================

def get_cpu_info() -> Dict:
    """获取 CPU 信息"""
    info = {
        "physical_cores": os.cpu_count(),
        "avx512": False,
        "avx2": False,
        "numa_nodes": 1,
    }
    
    try:
        # 获取 lscpu 输出
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            output = result.stdout
            
            # 检查 AVX 支持
            if "avx512" in output.lower():
                info["avx512"] = True
            if "avx2" in output.lower():
                info["avx2"] = True
            
            # 解析更多信息
            for line in output.split("\n"):
                if "NUMA node(s):" in line:
                    info["numa_nodes"] = int(line.split(":")[1].strip())
                elif "Model name:" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                elif "CPU(s):" in line and "NUMA" not in line and "On-line" not in line:
                    info["total_cpus"] = int(line.split(":")[1].strip())
                elif "Thread(s) per core:" in line:
                    info["threads_per_core"] = int(line.split(":")[1].strip())
                elif "Core(s) per socket:" in line:
                    info["cores_per_socket"] = int(line.split(":")[1].strip())
                elif "Socket(s):" in line:
                    info["sockets"] = int(line.split(":")[1].strip())
    except Exception as e:
        print(f"Warning: Failed to get detailed CPU info: {e}")
    
    return info


def get_numa_info() -> Dict:
    """获取 NUMA 拓扑信息"""
    info = {
        "available": False,
        "nodes": [],
        "current_cpu_node": None,
        "current_mem_node": None,
    }
    
    try:
        # 检查 numactl 是否可用
        result = subprocess.run(
            ["numactl", "--hardware"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["available"] = True
            output = result.stdout
            
            # 解析 NUMA 节点信息
            for line in output.split("\n"):
                if line.startswith("node") and "cpus:" in line:
                    node_id = int(line.split()[1])
                    cpus = line.split("cpus:")[1].strip()
                    info["nodes"].append({"id": node_id, "cpus": cpus})
        
        # 获取当前运行的 NUMA 绑定
        result = subprocess.run(
            ["numactl", "--show"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            output = result.stdout
            for line in output.split("\n"):
                if "nodebind:" in line:
                    info["current_cpu_node"] = line.split(":")[1].strip()
                elif "membind:" in line:
                    info["current_mem_node"] = line.split(":")[1].strip()
                    
    except FileNotFoundError:
        print("Warning: numactl not installed. NUMA testing not available.")
    except Exception as e:
        print(f"Warning: Failed to get NUMA info: {e}")
    
    return info


def get_memory_info() -> Dict:
    """获取内存信息"""
    info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal:" in line:
                    info["total_gb"] = int(line.split()[1]) / (1024 * 1024)
                elif "MemAvailable:" in line:
                    info["available_gb"] = int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return info


# ============================================================================
# 基准测试核心
# ============================================================================

@dataclass
class BenchmarkResult:
    """单次基准测试结果"""
    model_name: str
    num_threads: int
    layer_params: int
    update_time_ms: float
    throughput_gparams_per_sec: float
    memory_bandwidth_gb_per_sec: float
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "threads": self.num_threads,
            "layer_params_m": self.layer_params / 1e6,
            "update_time_ms": round(self.update_time_ms, 3),
            "throughput_gparams_s": round(self.throughput_gparams_per_sec, 3),
            "memory_bw_gb_s": round(self.memory_bandwidth_gb_per_sec, 2),
        }


class CPUAdamBenchmark:
    """CPU Adam 基准测试类"""
    
    def __init__(
        self,
        models: List[str] = None,
        thread_counts: List[int] = None,
        warmup_iters: int = 5,
        bench_iters: int = 20,
        dtype: torch.dtype = torch.float32,
    ):
        self.models = models or list(MODEL_CONFIGS.keys())
        self.thread_counts = thread_counts or self._get_default_thread_counts()
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.dtype = dtype
        self.results: List[BenchmarkResult] = []
        
        # 系统信息
        self.cpu_info = get_cpu_info()
        self.numa_info = get_numa_info()
        self.memory_info = get_memory_info()
    
    def _get_default_thread_counts(self) -> List[int]:
        """获取默认的线程数测试范围"""
        max_threads = os.cpu_count() or 8
        threads = [1]
        t = 2
        while t <= max_threads:
            threads.append(t)
            t *= 2
        if max_threads not in threads:
            threads.append(max_threads)
        return sorted(threads)
    
    def _create_fake_layer_params(
        self, num_params: int
    ) -> Tuple[List[torch.nn.Parameter], torch.Tensor]:
        """创建模拟层参数和梯度"""
        # 创建单个大 tensor 来模拟一层的参数
        param_tensor = torch.randn(num_params, dtype=self.dtype, device="cpu")
        param = torch.nn.Parameter(param_tensor)
        
        # 创建对应的梯度
        grad = torch.randn(num_params, dtype=torch.bfloat16, device="cpu")
        
        return [param], grad
    
    def _benchmark_single(
        self,
        model_config: ModelConfig,
        num_threads: int,
    ) -> BenchmarkResult:
        """执行单次基准测试"""
        # 设置 OpenMP 线程数
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)
        
        layer_params = model_config.layer_params
        
        # 创建优化器
        optimizer = LayerAdam(
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            adamw_mode=True,
            fp32_optimizer_state=True,
            num_layer=1,
        )
        
        # 创建模拟参数
        params, grad = self._create_fake_layer_params(layer_params)
        optimizer.add_layer_params(0, params)
        
        # 设置梯度
        # params[0].grad = grad
        
        # Warmup
        for _ in range(self.warmup_iters):
             # Ensure grad is bfloat16
            if grad.dtype != torch.bfloat16:
                 grad = grad.to(torch.bfloat16)
            optimizer.step_with_grad_views(layer_idx=0, grad_views={params[0]: grad})
        
        # 同步确保 warmup 完成
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # 基准测试
        times = []
        for _ in range(self.bench_iters):
            # 重新设置梯度（模拟真实场景）
            # params[0].grad = torch.randn_like(params[0])
            # grad = torch.randn_like(grad) # Keep using bfloat16
            
            # Simulate new gradient (bfloat16)
            grad.normal_() 
            
            start = time.perf_counter()
            optimizer.step_with_grad_views(layer_idx=0, grad_views={params[0]: grad})
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # ms
        
        # 计算统计
        avg_time_ms = sum(times) / len(times)
        
        # 计算吞吐量 (GParams/s)
        throughput = (layer_params / 1e9) / (avg_time_ms / 1000)
        
        # 计算内存带宽 (GB/s)
        # Adam 需要读: param, grad, exp_avg, exp_avg_sq (4个)
        # Adam 需要写: param, exp_avg, exp_avg_sq (3个)
        # 总共 7 次内存访问，每次 4 bytes (fp32)
        bytes_accessed = layer_params * 7 * 4  # bytes
        memory_bw = (bytes_accessed / 1e9) / (avg_time_ms / 1000)
        
        # 清理
        del optimizer, params, grad
        gc.collect()
        
        return BenchmarkResult(
            model_name=model_config.name,
            num_threads=num_threads,
            layer_params=layer_params,
            update_time_ms=avg_time_ms,
            throughput_gparams_per_sec=throughput,
            memory_bandwidth_gb_per_sec=memory_bw,
        )
    
    def run(self, verbose: bool = True) -> List[BenchmarkResult]:
        """运行完整基准测试"""
        if verbose:
            self._print_system_info()
        
        total_tests = len(self.models) * len(self.thread_counts)
        current_test = 0
        
        for model_name in self.models:
            if model_name not in MODEL_CONFIGS:
                print(f"Warning: Unknown model {model_name}, skipping")
                continue
            
            model_config = MODEL_CONFIGS[model_name]
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Testing: {model_config.name}")
                print(f"Layer params: {model_config.layer_params / 1e6:.2f}M")
                print(f"Total params: {model_config.total_params_b:.2f}B")
                print(f"{'='*60}")
            
            for num_threads in self.thread_counts:
                current_test += 1
                if verbose:
                    print(f"[{current_test}/{total_tests}] Threads: {num_threads}...", end=" ", flush=True)
                
                try:
                    result = self._benchmark_single(model_config, num_threads)
                    self.results.append(result)
                    
                    if verbose:
                        print(
                            f"Time: {result.update_time_ms:.2f}ms, "
                            f"Throughput: {result.throughput_gparams_per_sec:.2f} GParams/s, "
                            f"MemBW: {result.memory_bandwidth_gb_per_sec:.1f} GB/s"
                        )
                except Exception as e:
                    print(f"ERROR: {e}")
        
        return self.results
    
    def _print_system_info(self):
        """打印系统信息"""
        print("\n" + "="*70)
        print("CPU Adam Benchmark - System Information")
        print("="*70)
        
        # CPU 信息
        print("\n[CPU]")
        if "cpu_model" in self.cpu_info:
            print(f"  Model: {self.cpu_info['cpu_model']}")
        print(f"  Total CPUs: {self.cpu_info.get('total_cpus', 'N/A')}")
        print(f"  Sockets: {self.cpu_info.get('sockets', 'N/A')}")
        print(f"  Cores/Socket: {self.cpu_info.get('cores_per_socket', 'N/A')}")
        print(f"  Threads/Core: {self.cpu_info.get('threads_per_core', 'N/A')}")
        print(f"  AVX512: {'Yes' if self.cpu_info.get('avx512') else 'No'}")
        print(f"  AVX2: {'Yes' if self.cpu_info.get('avx2') else 'No'}")
        
        # NUMA 信息
        print("\n[NUMA]")
        print(f"  Nodes: {self.cpu_info.get('numa_nodes', 1)}")
        if self.numa_info.get("available"):
            print(f"  Current CPU bind: {self.numa_info.get('current_cpu_node', 'all')}")
            print(f"  Current MEM bind: {self.numa_info.get('current_mem_node', 'all')}")
            for node in self.numa_info.get("nodes", []):
                print(f"  Node {node['id']} CPUs: {node['cpus']}")
        
        # 内存信息
        print("\n[Memory]")
        if self.memory_info:
            print(f"  Total: {self.memory_info.get('total_gb', 0):.1f} GB")
            print(f"  Available: {self.memory_info.get('available_gb', 0):.1f} GB")
        
        # 测试配置
        print("\n[Benchmark Config]")
        print(f"  Models: {', '.join(self.models)}")
        print(f"  Thread counts: {self.thread_counts}")
        print(f"  Warmup iters: {self.warmup_iters}")
        print(f"  Bench iters: {self.bench_iters}")
        print(f"  Dtype: {self.dtype}")
        print("="*70)
    
    def print_summary(self):
        """打印结果摘要"""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*90)
        print("Benchmark Results Summary")
        print("="*90)
        
        # 按模型分组
        from collections import defaultdict
        by_model = defaultdict(list)
        for r in self.results:
            by_model[r.model_name].append(r)
        
        for model_name, results in by_model.items():
            print(f"\n{model_name}")
            print(f"  Layer params: {results[0].layer_params / 1e6:.2f}M")
            print("-" * 80)
            print(f"  {'Threads':>8} | {'Time (ms)':>12} | {'Throughput (GParams/s)':>22} | {'MemBW (GB/s)':>14} | {'Scaling':>8}")
            print("-" * 80)
            
            base_time = results[0].update_time_ms
            for r in results:
                scaling = base_time / r.update_time_ms
                print(
                    f"  {r.num_threads:>8} | {r.update_time_ms:>12.3f} | "
                    f"{r.throughput_gparams_per_sec:>22.3f} | "
                    f"{r.memory_bandwidth_gb_per_sec:>14.1f} | "
                    f"{scaling:>8.2f}x"
                )
    
    def save_results(self, output_path: str):
        """保存结果到 JSON 文件"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu": self.cpu_info,
                "numa": self.numa_info,
                "memory": self.memory_info,
            },
            "config": {
                "models": self.models,
                "thread_counts": self.thread_counts,
                "warmup_iters": self.warmup_iters,
                "bench_iters": self.bench_iters,
                "dtype": str(self.dtype),
            },
            "model_configs": {
                name: {
                    "num_layers": cfg.num_layers,
                    "hidden_size": cfg.hidden_size,
                    "intermediate_size": cfg.intermediate_size,
                    "layer_params": cfg.layer_params,
                    "total_params_b": cfg.total_params_b,
                }
                for name, cfg in MODEL_CONFIGS.items()
                if name in self.models
            },
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


# ============================================================================
# 高级 Profiling
# ============================================================================

class CPUAdamProfiler:
    """CPU Adam 详细性能分析器"""
    
    def __init__(self, model_name: str = "8B", num_threads: int = None):
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        self.num_threads = num_threads or os.cpu_count()
    
    def profile_detailed(self, num_iters: int = 50) -> Dict:
        """执行详细的性能分析"""
        import cProfile
        import pstats
        import io
        
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        torch.set_num_threads(self.num_threads)
        
        layer_params = self.model_config.layer_params
        
        # 创建优化器
        optimizer = LayerAdam(
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            adamw_mode=True,
            fp32_optimizer_state=True,
            num_layer=1,
        )
        
        # 创建参数
        param = torch.nn.Parameter(
            torch.randn(layer_params, dtype=torch.float32, device="cpu")
        )
        optimizer.add_layer_params(0, [param])
        # param.grad = torch.randn_like(param)
        grad = torch.randn(layer_params, dtype=torch.bfloat16, device="cpu")
        
        # Warmup
        for _ in range(5):
            optimizer.step_with_grad_views(layer_idx=0, grad_views={param: grad})
        
        # Profile
        profiler = cProfile.Profile()
        profiler.enable()
        
        times = []
        for _ in range(num_iters):
            # param.grad = torch.randn_like(param)
            grad.normal_()
            start = time.perf_counter()
            grad = grad.float()
            optimizer.step_with_grad_views(layer_idx=0, grad_views={param: grad})
            times.append(time.perf_counter() - start)
        
        profiler.disable()
        
        # 获取 profile 结果
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        profile_output = s.getvalue()
        
        # 统计分析
        times_ms = [t * 1000 for t in times]
        
        results = {
            "model": self.model_name,
            "layer_params_m": layer_params / 1e6,
            "num_threads": self.num_threads,
            "num_iters": num_iters,
            "timing": {
                "mean_ms": sum(times_ms) / len(times_ms),
                "min_ms": min(times_ms),
                "max_ms": max(times_ms),
                "std_ms": (sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
            },
            "profile_output": profile_output,
        }
        
        return results
    
    def profile_with_perf(self, num_iters: int = 100) -> str:
        """使用 perf 进行硬件性能计数器分析"""
        print(f"\nRunning perf analysis for {self.model_name}...")
        print("(This requires 'perf' tool and appropriate permissions)")
        
        # 创建临时脚本
        script = f'''
import os
import torch
import sys
sys.path.insert(0, "{project_root}")
from optimizer import LayerAdam

os.environ["OMP_NUM_THREADS"] = "{self.num_threads}"
torch.set_num_threads({self.num_threads})

layer_params = {self.model_config.layer_params}

optimizer = LayerAdam(
    lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0.01, adamw_mode=True,
    fp32_optimizer_state=True, num_layer=1,
)

param = torch.nn.Parameter(torch.randn(layer_params, dtype=torch.float32, device="cpu"))
optimizer.add_layer_params(0, [param])
grad = torch.randn(layer_params, dtype=torch.bfloat16, device="cpu")

# Warmup
for _ in range(10):
    grad.normal_()
    optimizer.step_with_grad_views(layer_idx=0, grad_views={param: grad})

# Benchmark
for _ in range({num_iters}):
    grad.normal_()
    optimizer.step_with_grad_views(layer_idx=0, grad_views={param: grad})
'''
        
        script_path = "/tmp/cpu_adam_perf_test.py"
        with open(script_path, "w") as f:
            f.write(script)
        
        # 运行 perf
        try:
            result = subprocess.run(
                [
                    "perf", "stat",
                    "-e", "cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses",
                    "python", script_path
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.stderr  # perf outputs to stderr
        except FileNotFoundError:
            return "perf tool not found. Install linux-tools-common."
        except subprocess.TimeoutExpired:
            return "Timeout during perf analysis"
        except Exception as e:
            return f"Error: {e}"


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU Adam Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to test (e.g., '3B,8B,14B'). Default: all",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default=None,
        help="Comma-separated list of thread counts (e.g., '1,2,4,8'). Default: auto",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed profiling (cProfile)",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Enable perf hardware counters analysis",
    )
    parser.add_argument(
        "--profile-model",
        type=str,
        default="8B",
        help="Model to use for detailed profiling",
    )
    parser.add_argument(
        "--profile-threads",
        type=int,
        default=None,
        help="Thread count for detailed profiling",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 解析模型列表
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    
    # 解析线程列表
    threads = None
    if args.threads:
        threads = [int(t.strip()) for t in args.threads.split(",")]
    
    # 详细 profiling 模式
    if args.profile:
        profiler = CPUAdamProfiler(
            model_name=args.profile_model,
            num_threads=args.profile_threads,
        )
        results = profiler.profile_detailed(num_iters=args.iters)
        
        print("\n" + "="*70)
        print("Detailed Profiling Results")
        print("="*70)
        print(f"Model: {results['model']}")
        print(f"Layer params: {results['layer_params_m']:.2f}M")
        print(f"Threads: {results['num_threads']}")
        print(f"\nTiming Statistics:")
        print(f"  Mean: {results['timing']['mean_ms']:.3f} ms")
        print(f"  Min:  {results['timing']['min_ms']:.3f} ms")
        print(f"  Max:  {results['timing']['max_ms']:.3f} ms")
        print(f"  Std:  {results['timing']['std_ms']:.3f} ms")
        print(f"\nProfile Output (top functions):")
        print(results['profile_output'])
        return
    
    # perf 分析模式
    if args.perf:
        profiler = CPUAdamProfiler(
            model_name=args.profile_model,
            num_threads=args.profile_threads,
        )
        output = profiler.profile_with_perf(num_iters=args.iters)
        print("\n" + "="*70)
        print("Perf Hardware Counters Analysis")
        print("="*70)
        print(output)
        return
    
    # 标准基准测试
    benchmark = CPUAdamBenchmark(
        models=models,
        thread_counts=threads,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )
    
    benchmark.run()
    benchmark.print_summary()
    
    # 保存结果
    if args.output:
        benchmark.save_results(args.output)
    else:
        # 默认输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "bench" / "cpu_adam_profile" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"cpu_adam_bench_{timestamp}.json"
        benchmark.save_results(str(output_path))


if __name__ == "__main__":
    main()

