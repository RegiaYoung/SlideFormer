#!/usr/bin/env python3
"""
CPU Adam 基准测试结果分析和可视化

使用方法:
    # 分析单个结果文件
    python analyze_results.py results/cpu_adam_bench_xxx.json
    
    # 分析多个结果文件（NUMA对比）
    python analyze_results.py results/numa/*.json
    
    # 生成可视化图表
    python analyze_results.py results/*.json --plot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互模式
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization disabled.")


# Preferred display order for models (fallbacks to append unknowns at end)
MODEL_ORDER = ["3B", "8B", "14B", "32B", "72B"]


def sort_models(models: List[str]) -> List[str]:
    """Sort models by predefined size order, append unknowns at end."""
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: order.get(m, len(order)))


def load_results(file_path: str) -> Dict:
    """加载单个结果文件"""
    with open(file_path, "r") as f:
        return json.load(f)


def analyze_single_result(data: Dict, verbose: bool = True) -> Dict:
    """分析单个基准测试结果"""
    results = data.get("results", [])
    
    # 按模型分组
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    
    analysis = {}
    
    for model, model_results in by_model.items():
        # 排序按线程数
        model_results.sort(key=lambda x: x["threads"])
        
        # 获取基线（单线程）
        baseline = model_results[0]
        
        # 计算 scaling 效率
        scalings = []
        for r in model_results:
            speedup = baseline["update_time_ms"] / r["update_time_ms"]
            ideal_speedup = r["threads"]
            efficiency = speedup / ideal_speedup * 100
            scalings.append({
                "threads": r["threads"],
                "time_ms": r["update_time_ms"],
                "speedup": speedup,
                "ideal_speedup": ideal_speedup,
                "efficiency": efficiency,
                "throughput": r["throughput_gparams_s"],
                "memory_bw": r["memory_bw_gb_s"],
            })
        
        # 找到最佳线程数
        best = min(model_results, key=lambda x: x["update_time_ms"])
        
        analysis[model] = {
            "layer_params_m": baseline["layer_params_m"],
            "baseline_time_ms": baseline["update_time_ms"],
            "best_time_ms": best["update_time_ms"],
            "best_threads": best["threads"],
            "max_speedup": baseline["update_time_ms"] / best["update_time_ms"],
            "scalings": scalings,
        }
    
    if verbose:
        print_analysis(analysis, data.get("system_info", {}))
    
    return analysis


def print_analysis(analysis: Dict, system_info: Dict):
    """打印分析结果"""
    print("\n" + "="*90)
    print("Performance Analysis")
    print("="*90)
    
    # 系统信息
    cpu_info = system_info.get("cpu", {})
    numa_info = system_info.get("numa", {})
    
    if cpu_info:
        print(f"\nCPU: {cpu_info.get('cpu_model', 'Unknown')}")
        print(f"AVX512: {'Yes' if cpu_info.get('avx512') else 'No'}, AVX2: {'Yes' if cpu_info.get('avx2') else 'No'}")
    
    if numa_info.get("available"):
        print(f"NUMA: CPU bind={numa_info.get('current_cpu_node', 'all')}, "
              f"Mem bind={numa_info.get('current_mem_node', 'all')}")
    
    # 每个模型的分析
    for model, data in analysis.items():
        print(f"\n{'='*80}")
        print(f"Model: {model}")
        print(f"Layer params: {data['layer_params_m']:.2f}M")
        print(f"Baseline (1 thread): {data['baseline_time_ms']:.3f} ms")
        print(f"Best: {data['best_time_ms']:.3f} ms @ {data['best_threads']} threads")
        print(f"Max speedup: {data['max_speedup']:.2f}x")
        print(f"{'='*80}")
        
        # Scaling 表格
        print(f"\n{'Threads':>8} | {'Time(ms)':>10} | {'Speedup':>8} | {'Ideal':>8} | {'Efficiency':>10} | {'Throughput':>12} | {'MemBW(GB/s)':>12}")
        print("-"*88)
        
        for s in data["scalings"]:
            print(f"{s['threads']:>8} | {s['time_ms']:>10.3f} | {s['speedup']:>8.2f}x | "
                  f"{s['ideal_speedup']:>7.1f}x | {s['efficiency']:>9.1f}% | "
                  f"{s['throughput']:>10.3f} GP/s | {s['memory_bw']:>10.1f}")
        
        # Scaling 评估
        best_efficiency = max(s["efficiency"] for s in data["scalings"] if s["threads"] > 1)
        worst_efficiency = min(s["efficiency"] for s in data["scalings"] if s["threads"] > 1)
        
        print(f"\nScaling Assessment:")
        if best_efficiency > 80:
            print(f"  ✓ Good scaling (best efficiency: {best_efficiency:.1f}%)")
        elif best_efficiency > 50:
            print(f"  △ Moderate scaling (best efficiency: {best_efficiency:.1f}%)")
        else:
            print(f"  ✗ Poor scaling (best efficiency: {best_efficiency:.1f}%)")
        
        # 内存带宽分析
        max_bw = max(s["memory_bw"] for s in data["scalings"])
        print(f"  Peak memory bandwidth: {max_bw:.1f} GB/s")
        
        # 如果带宽随线程增加不多，说明是内存带宽瓶颈
        bw_at_max_threads = data["scalings"][-1]["memory_bw"]
        bw_at_1_thread = data["scalings"][0]["memory_bw"]
        bw_scaling = bw_at_max_threads / bw_at_1_thread if bw_at_1_thread > 0 else 1
        
        threads_ratio = data["scalings"][-1]["threads"]
        if bw_scaling < threads_ratio * 0.3:
            print(f"  ⚠ Likely memory bandwidth bound (BW scaling: {bw_scaling:.2f}x vs threads: {threads_ratio}x)")


def compare_numa_results(results_list: List[Dict], verbose: bool = True) -> Dict:
    """比较多个 NUMA 配置的结果"""
    comparisons = {}
    
    for data in results_list:
        # 获取 NUMA 配置标识
        numa_info = data.get("system_info", {}).get("numa", {})
        cpu_bind = numa_info.get("current_cpu_node", "all")
        mem_bind = numa_info.get("current_mem_node", "all")
        config_name = f"cpu{cpu_bind}_mem{mem_bind}"
        
        # 分析结果
        analysis = analyze_single_result(data, verbose=False)
        comparisons[config_name] = analysis
    
    if verbose:
        print_numa_comparison(comparisons)
    
    return comparisons


def print_numa_comparison(comparisons: Dict):
    """打印 NUMA 对比结果"""
    print("\n" + "="*100)
    print("NUMA Configuration Comparison")
    print("="*100)
    
    # 获取所有模型
    all_models = set()
    for config, analysis in comparisons.items():
        all_models.update(analysis.keys())
    
    for model in sort_models(list(all_models)):
        print(f"\n{'-'*80}")
        print(f"Model: {model}")
        print(f"{'-'*80}")
        
        # 打印表头
        configs = list(comparisons.keys())
        print(f"\n{'Config':>20} | {'Best Time(ms)':>12} | {'Best Threads':>12} | {'Max Speedup':>12} | {'vs Baseline':>12}")
        print("-"*80)
        
        baseline_time = None
        for config in configs:
            if model not in comparisons[config]:
                continue
            
            data = comparisons[config][model]
            
            if baseline_time is None:
                baseline_time = data["best_time_ms"]
                vs_baseline = "1.00x (baseline)"
            else:
                ratio = baseline_time / data["best_time_ms"]
                vs_baseline = f"{ratio:.2f}x"
            
            print(f"{config:>20} | {data['best_time_ms']:>12.3f} | {data['best_threads']:>12} | "
                  f"{data['max_speedup']:>11.2f}x | {vs_baseline:>12}")


def plot_results(analysis: Dict, output_path: str = None):
    """生成可视化图表"""
    if not HAS_MATPLOTLIB:
        print("Cannot generate plots: matplotlib not installed")
        return

    models = sort_models(list(analysis.keys()))
    num_models = len(models)
    fig, axes = plt.subplots(2, num_models, figsize=(6*num_models, 10))

    if num_models == 1:
        axes = axes.reshape(2, 1)

    for idx, model in enumerate(models):
        data = analysis[model]
        threads = [s["threads"] for s in data["scalings"]]
        times = [s["time_ms"] for s in data["scalings"]]
        efficiencies = [s["efficiency"] for s in data["scalings"]]

        # 上图: 时间 vs 线程数
        ax1 = axes[0, idx]
        ax1.plot(threads, times, 'b-o', label='Actual', linewidth=2, markersize=8)

        ideal_times = [times[0] / t for t in threads]
        ax1.plot(threads, ideal_times, 'g--', label='Ideal', linewidth=1.5, alpha=0.7)

        ax1.set_xlabel('Threads')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'{model}\nUpdate Time vs Threads')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')

        # 下图: 效率 vs 线程数
        ax2 = axes[1, idx]
        bars = ax2.bar(range(len(threads)), efficiencies, color='steelblue', alpha=0.7)

        for bar, eff in zip(bars, efficiencies):
            if eff >= 80:
                bar.set_color('green')
            elif eff >= 50:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        ax2.set_xticks(range(len(threads)))
        ax2.set_xticklabels([str(t) for t in threads])
        ax2.set_xlabel('Threads')
        ax2.set_ylabel('Parallel Efficiency (%)')
        ax2.set_title(f'{model}\nScaling Efficiency')
        ax2.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% (Good)')
        ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% (Moderate)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 110)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.savefig('cpu_adam_scaling.png', dpi=150, bbox_inches='tight')
        print("Plot saved to: cpu_adam_scaling.png")

    plt.close()


def plot_numa_comparison(comparisons: Dict, output_path: str = None):
    """生成 NUMA 对比图表"""
    if not HAS_MATPLOTLIB:
        print("Cannot generate plots: matplotlib not installed")
        return

    all_models = set()
    for config, analysis in comparisons.items():
        all_models.update(analysis.keys())
    all_models = sort_models(list(all_models))

    num_models = len(all_models)
    num_configs = len(comparisons)

    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
    if num_models == 1:
        axes = [axes]

    configs = list(comparisons.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, num_configs))

    for idx, model in enumerate(all_models):
        ax = axes[idx]

        for config_idx, config in enumerate(configs):
            if model not in comparisons[config]:
                continue

            data = comparisons[config][model]
            threads = [s["threads"] for s in data["scalings"]]
            times = [s["time_ms"] for s in data["scalings"]]

            ax.plot(threads, times, '-o', color=colors[config_idx],
                    label=config, linewidth=2, markersize=6)

        ax.set_xlabel('Threads')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{model}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

    plt.suptitle('NUMA Configuration Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"NUMA comparison plot saved to: {output_path}")
    else:
        plt.savefig('cpu_adam_numa_comparison.png', dpi=150, bbox_inches='tight')
        print("NUMA comparison plot saved to: cpu_adam_numa_comparison.png")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze CPU Adam benchmark results")
    parser.add_argument("files", nargs="+", help="Result JSON file(s)")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument("--output", type=str, help="Output path for plots")
    parser.add_argument("--compare-numa", action="store_true", 
                       help="Compare results as different NUMA configurations")
    
    args = parser.parse_args()
    
    # 加载所有结果
    results_list = []
    for file_path in args.files:
        try:
            data = load_results(file_path)
            data["_source_file"] = file_path
            results_list.append(data)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not results_list:
        print("No results to analyze")
        sys.exit(1)
    
    # 分析
    if args.compare_numa and len(results_list) > 1:
        # NUMA 对比模式
        comparisons = compare_numa_results(results_list)
        
        if args.plot:
            plot_numa_comparison(comparisons, args.output)
    else:
        # 单结果或多结果聚合分析
        combined_data = {"results": [], "system_info": results_list[0].get("system_info", {})}

        for data in results_list:
            print(f"\n{'#'*80}")
            print(f"# File: {data.get('_source_file', 'Unknown')}")
            print(f"{'#'*80}")
            
            analysis = analyze_single_result(data)
            combined_data["results"].extend(data.get("results", []))

        if args.plot:
            combined_analysis = analyze_single_result(combined_data, verbose=False)
            plot_path = args.output or "cpu_adam_scaling.png"
            plot_results(combined_analysis, plot_path)


if __name__ == "__main__":
    main()

