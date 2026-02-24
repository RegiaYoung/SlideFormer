# CPU Adam Performance Benchmark Suite

本工具用于测试 LayerAdam (CPU Adam) 优化器的性能，包括：
- 不同模型大小（3B/8B/14B/32B/72B）
- 不同线程数/核数的 scaling 性能
- NUMA 拓扑感知测试（跨 NUMA 节点访问的影响）
- 详细的性能 profiling

## 快速开始

### 1. 基础测试

```bash
cd /home/scc/SSDP/benchmark

# 运行基础测试（所有模型，自动检测线程数）
python cpu_adam_benchmark.py

# 指定特定模型和线程数
python cpu_adam_benchmark.py --models 8B,14B --threads 1,4,8,16,32

# 快速测试（较少迭代）
python cpu_adam_benchmark.py --models 8B --threads 1,8 --warmup 3 --iters 10
```

### 2. NUMA 测试

测试 CPU 和内存在不同 NUMA 节点时的性能影响：

```bash
# 同一 NUMA 节点（最佳情况）
numactl --cpunodebind=0 --membind=0 python cpu_adam_benchmark.py

# 跨 NUMA 节点（模拟远程内存访问）
numactl --cpunodebind=0 --membind=1 python cpu_adam_benchmark.py

# 或使用自动化脚本
chmod +x numa_benchmark.sh
./numa_benchmark.sh
```

### 3. 详细 Profiling

```bash
# 使用 cProfile 分析
python cpu_adam_benchmark.py --profile --profile-model 8B --profile-threads 16

# 使用 perf 硬件计数器（需要 root 或 perf 权限）
python cpu_adam_benchmark.py --perf --profile-model 8B
```

### 4. 完整基准测试

```bash
chmod +x run_full_benchmark.sh

# 完整测试（所有模型 + NUMA）
./run_full_benchmark.sh

# 快速模式
./run_full_benchmark.sh --quick

# 跳过 NUMA 测试
./run_full_benchmark.sh --no-numa

# 包含 profiling
./run_full_benchmark.sh --profile
```

### 5. 分析结果

```bash
# 分析单个结果
python analyze_results.py results/xxx.json

# 生成可视化图表
python analyze_results.py results/xxx.json --plot

# 对比多个 NUMA 配置
python analyze_results.py results/numa/*.json --compare-numa --plot
```

## 输出说明

### 结果 JSON 格式

```json
{
    "timestamp": "2025-01-07T...",
    "system_info": {
        "cpu": {"cpu_model": "...", "avx512": true, ...},
        "numa": {"nodes": [...], "current_cpu_node": "0", ...},
        "memory": {"total_gb": 256, ...}
    },
    "results": [
        {
            "model": "8B (Llama-3.1-8B style)",
            "threads": 16,
            "layer_params_m": 262.14,
            "update_time_ms": 12.345,
            "throughput_gparams_s": 21.23,
            "memory_bw_gb_s": 148.6
        },
        ...
    ]
}
```

### 关键指标

| 指标 | 说明 |
|-----|------|
| `update_time_ms` | 单次 Adam 更新耗时（毫秒）|
| `throughput_gparams_s` | 吞吐量（每秒处理的参数量，单位 GParams/s）|
| `memory_bw_gb_s` | 内存带宽（GB/s），基于 Adam 的内存访问模式计算 |
| `speedup` | 相对单线程的加速比 |
| `efficiency` | 并行效率 = speedup / threads × 100% |

### Scaling 评估标准

- **效率 > 80%**: Good scaling ✓
- **效率 50-80%**: Moderate scaling △
- **效率 < 50%**: Poor scaling ✗

## 模型配置

预定义的模型配置基于主流 LLM 架构：

| 模型 | 参考架构 | 层数 | Hidden Size | 每层参数量 |
|-----|---------|------|-------------|----------|
| 3B  | Llama-3.2-3B | 28 | 3072 | ~100M |
| 8B  | Llama-3.1-8B | 32 | 4096 | ~260M |
| 14B | Qwen2.5-14B | 40 | 5120 | ~350M |
| 32B | Qwen2.5-32B | 64 | 5120 | ~530M |
| 72B | Qwen2.5-72B | 80 | 8192 | ~900M |

## NUMA 测试说明

### 背景

在多 socket 服务器上，每个 CPU socket 有自己的本地内存（NUMA node）。访问远程 NUMA node 的内存会有额外延迟，这可能显著影响内存密集型操作（如 CPU Adam）的性能。

### 测试配置

1. **Same NUMA**: CPU 和内存在同一节点 → 最佳性能
2. **Cross NUMA**: CPU 在节点 0，内存在节点 1 → 远程内存访问
3. **Baseline**: 无 NUMA 绑定 → 系统自动分配

### 预期结果

- 跨 NUMA 访问通常会导致 10-30% 的性能下降
- 线程数越多，NUMA 影响可能越明显（竞争带宽）

## 性能瓶颈诊断

### 1. 内存带宽瓶颈

特征：
- 增加线程数后，内存带宽饱和
- 更多线程无法提升吞吐量
- efficiency 随线程增加急剧下降

解决方案：
- 确保在同一 NUMA 节点运行
- 考虑使用 NVMe 卸载优化器状态

### 2. CPU 计算瓶颈

特征：
- 内存带宽未饱和
- 单线程性能较低
- AVX 向量化未生效

检查：
```bash
# 确认 AVX 支持
lscpu | grep avx

# 检查编译时是否启用 AVX
python -c "from optimizer import LayerAdam; print('OK')"
# 查看输出中的 "AVX512" 或 "AVX2"
```

### 3. NUMA 瓶颈

特征：
- 跨 NUMA 测试性能显著下降
- 远程内存访问延迟高

解决方案：
```bash
# 绑定到单一 NUMA 节点
numactl --cpunodebind=0 --membind=0 python train.py

# 或设置 interleave
numactl --interleave=all python train.py
```

## 依赖

```
torch>=2.0.0
matplotlib (可选，用于可视化)
numactl (可选，用于 NUMA 测试)
perf (可选，用于硬件计数器分析)
```

## 文件结构

```
benchmark/
├── cpu_adam_benchmark.py   # 主基准测试脚本
├── analyze_results.py      # 结果分析和可视化
├── numa_benchmark.sh       # NUMA 自动化测试
├── run_full_benchmark.sh   # 完整测试套件
├── README.md               # 本文档
└── results/                # 测试结果目录
    ├── *.json              # 原始数据
    └── *.png               # 可视化图表
```

## 常见问题

### Q: 测试时报错 "numactl not found"

```bash
sudo apt-get install numactl
```

### Q: 如何确认 AVX512 是否启用？

创建优化器时会打印：
```
Adam Optimizer #0 is created with AVX512 arithmetic capability.
```

如果显示 "AVX2" 或 "scalar"，说明未使用 AVX512。

### Q: 为什么 scaling 效率很低？

1. 检查是否是内存带宽瓶颈（查看 memory_bw_gb_s 是否饱和）
2. 检查 NUMA 配置（使用 `numactl --show`）
3. 尝试减少线程数，找到最佳配置

### Q: 如何在训练中使用最佳配置？

```bash
# 设置 OMP 线程数
export OMP_NUM_THREADS=16

# 绑定 NUMA
numactl --cpunodebind=0 --membind=0 torchrun --nproc_per_node=4 train.py
```

