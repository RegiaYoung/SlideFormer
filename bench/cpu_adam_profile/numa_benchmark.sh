#!/bin/bash
#
# NUMA-aware CPU Adam Benchmark Script
#
# 该脚本自动化测试 CPU Adam 在不同 NUMA 配置下的性能：
# 1. CPU 和内存在同一 NUMA 节点
# 2. CPU 和内存在不同 NUMA 节点 (跨 NUMA 访问)
#
# 使用方法:
#     chmod +x numa_benchmark.sh
#     ./numa_benchmark.sh
#
# 要求:
#     - numactl 已安装
#     - 系统有至少 2 个 NUMA 节点
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/cpu_adam_benchmark.py"

# 默认配置
MODELS="${MODELS:-3B,8B,14B}"
THREADS="${THREADS:-1,2,4,8,16,32,64}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/numa}"

# 打印标题
echo -e "${BLUE}"
echo "=================================================================="
echo "         CPU Adam NUMA Performance Benchmark"
echo "=================================================================="
echo -e "${NC}"

# 检查 numactl
if ! command -v numactl &> /dev/null; then
    echo -e "${RED}Error: numactl is not installed${NC}"
    echo "Install it with: sudo apt-get install numactl"
    exit 1
fi

# 获取 NUMA 信息
NUMA_NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
echo -e "${GREEN}System has ${NUMA_NODES} NUMA node(s)${NC}"

if [ "$NUMA_NODES" -lt 2 ]; then
    echo -e "${YELLOW}Warning: System has less than 2 NUMA nodes.${NC}"
    echo "Cross-NUMA testing will be skipped."
fi

# 显示 NUMA 拓扑
echo ""
echo "NUMA Topology:"
numactl --hardware | grep -E "node [0-9]+ cpus:|node [0-9]+ size:"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 函数: 运行基准测试
run_benchmark() {
    local cpu_node=$1
    local mem_node=$2
    local test_name=$3
    local output_file="${OUTPUT_DIR}/${TIMESTAMP}_${test_name}.json"
    
    echo -e "${BLUE}Running: ${test_name}${NC}"
    echo "  CPU Node: ${cpu_node}, Memory Node: ${mem_node}"
    echo "  Output: ${output_file}"
    
    # 构建 numactl 命令
    local numa_cmd=""
    if [ "$cpu_node" != "all" ] && [ "$mem_node" != "all" ]; then
        numa_cmd="numactl --cpunodebind=${cpu_node} --membind=${mem_node}"
    fi
    
    # 运行基准测试
    ${numa_cmd} python "${BENCHMARK_SCRIPT}" \
        --models "${MODELS}" \
        --threads "${THREADS}" \
        --warmup "${WARMUP}" \
        --iters "${ITERS}" \
        --output "${output_file}"
    
    echo -e "${GREEN}✓ Completed: ${test_name}${NC}"
    echo ""
}

# 测试 1: 无 NUMA 绑定 (基线)
echo "=================================================================="
echo "Test 1: Baseline (No NUMA binding)"
echo "=================================================================="
run_benchmark "all" "all" "baseline"

# 测试 2: 同一 NUMA 节点 (CPU 和内存在同一节点)
if [ "$NUMA_NODES" -ge 1 ]; then
    echo "=================================================================="
    echo "Test 2: Same NUMA node (Node 0)"
    echo "=================================================================="
    run_benchmark "0" "0" "same_numa_node0"
fi

# 测试 3: 跨 NUMA 节点 (CPU 和内存在不同节点)
if [ "$NUMA_NODES" -ge 2 ]; then
    echo "=================================================================="
    echo "Test 3: Cross NUMA (CPU on Node 0, Memory on Node 1)"
    echo "=================================================================="
    run_benchmark "0" "1" "cross_numa_cpu0_mem1"
    
    echo "=================================================================="
    echo "Test 4: Cross NUMA (CPU on Node 1, Memory on Node 0)"
    echo "=================================================================="
    run_benchmark "1" "0" "cross_numa_cpu1_mem0"
    
    # 可选: 第二个节点的本地测试
    echo "=================================================================="
    echo "Test 5: Same NUMA node (Node 1)"
    echo "=================================================================="
    run_benchmark "1" "1" "same_numa_node1"
fi

# 生成汇总报告
echo "=================================================================="
echo "Generating Summary Report"
echo "=================================================================="

SUMMARY_FILE="${OUTPUT_DIR}/${TIMESTAMP}_summary.txt"

{
    echo "CPU Adam NUMA Benchmark Summary"
    echo "================================"
    echo "Timestamp: $(date)"
    echo "Models: ${MODELS}"
    echo "Thread counts: ${THREADS}"
    echo ""
    echo "System Info:"
    echo "------------"
    lscpu | grep -E "Model name|Socket|Core|Thread|NUMA"
    echo ""
    echo "Memory Info:"
    numactl --hardware | grep -E "size|free"
    echo ""
    echo "Results Files:"
    ls -la "${OUTPUT_DIR}/${TIMESTAMP}"*.json 2>/dev/null || echo "No result files found"
    echo ""
} > "${SUMMARY_FILE}"

echo -e "${GREEN}Summary saved to: ${SUMMARY_FILE}${NC}"

# 完成
echo ""
echo -e "${GREEN}=================================================================="
echo "                    All benchmarks completed!"
echo "=================================================================="
echo -e "${NC}"
echo "Results directory: ${OUTPUT_DIR}"
echo ""
echo "To analyze results, you can use:"
echo "  python ${SCRIPT_DIR}/analyze_numa_results.py ${OUTPUT_DIR}/${TIMESTAMP}_*.json"

