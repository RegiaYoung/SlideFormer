#!/bin/bash
#
# CPU Adam Benchmark Suite
# 
# Flows:
# 1. Detect System & Model Sizes
# 2. Benchmark Small Models (Fit in single NUMA)
#    - Phase 1a: Intra-NUMA Scaling (1..Half Threads) -> Baseline for NUMA comparison
#    - Phase 1b: Full-System Scaling (Full Threads)   -> Interleaved mode for throughput
# 3. Benchmark Large Models (Exceed single NUMA)
#    - Phase 1c: Full Scaling (1..Full Threads)       -> Interleaved mode
# 4. Cross-NUMA Analysis
#    - Phase 2: Inter-NUMA Scaling (Small Models, 1..Half Threads)
# 5. Reporting
#

set -e

# --- Configuration ---
RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Default input models
INPUT_MODELS="3B,8B,14B,32B,72B"
WARMUP=5
ITERS=20

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'
CYAN='\033[0;36m'

function print_header() { echo -e "${BLUE}=== $1 ===${NC}"; }

# 打印 banner
echo -e "${CYAN}"
cat << 'EOF'
  ____ ____  _   _      _       _                 ____             __ _ 
 / ___|  _ \| | | |    / \   __| | __ _ _ __ ___ |  _ \ _ __ ___  / _| |
| |   | |_) | | | |   / _ \ / _` |/ _` | '_ ` _ \|  __/| '__/ _ \| |_| |
| |___|  __/| |_| |  / ___ \ (_| | (_| | | | | | | |   | | | (_) |  _|_|
 \____|_|    \___/  /_/   \_\__,_|\__,_|_| |_| |_|_|   |_|  \___/|_| (_)

                Performance Benchmark Suite
EOF
echo -e "${NC}"

# --- 1. System Detection ---
print_header "System Detection"
if ! command -v numactl &> /dev/null; then echo "Error: numactl missing"; exit 1; fi

TOTAL_CPUS=$(nproc)
NUMA_NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
[ -z "$NUMA_NODES" ] && NUMA_NODES=1

# Get NUMA Node 0 Memory Size in GB
NUMA_MEM_KB=$(numactl --hardware | grep "node 0 size:" | awk '{print $4}')
# Handle unit suffixes if numactl output varies, assuming MB/GB or raw
# But typically numactl -H output is "128000 MB". Let's assume MB for safety parsing if needed, 
# but simplify to just running a python snippet for logic.

if [ "$NUMA_NODES" -gt 1 ]; then
    MAX_NUMA_THREADS=$((TOTAL_CPUS / NUMA_NODES))
else
    MAX_NUMA_THREADS=$TOTAL_CPUS
fi

echo "Total CPUs: $TOTAL_CPUS"
echo "NUMA Nodes: $NUMA_NODES"
echo "Max Threads per Node: $MAX_NUMA_THREADS"

# Thread Lists
# 1. Intra-NUMA list: 1, 2, ..., MAX_NUMA_THREADS
THREADS_HALF=""
t=1
while [ $t -le $MAX_NUMA_THREADS ]; do
    THREADS_HALF="${THREADS_HALF}${t},"
    t=$((t * 2))
done
if [[ "$THREADS_HALF" != *"${MAX_NUMA_THREADS},"* ]]; then THREADS_HALF="${THREADS_HALF}${MAX_NUMA_THREADS},"; fi
THREADS_HALF=${THREADS_HALF%,}

# 2. Full list: 1, 2, ..., TOTAL_CPUS
THREADS_FULL=""
t=1
while [ $t -le $TOTAL_CPUS ]; do
    THREADS_FULL="${THREADS_FULL}${t},"
    t=$((t * 2))
done
if [[ "$THREADS_FULL" != *"${TOTAL_CPUS},"* ]]; then THREADS_FULL="${THREADS_FULL}${TOTAL_CPUS},"; fi
THREADS_FULL=${THREADS_FULL%,}

# 3. Extension list: Just the high counts (for small models extension)
THREADS_EXT=""
IFS=',' read -ra ADDR <<< "$THREADS_FULL"
for t in "${ADDR[@]}"; do
    if [ "$t" -gt "$MAX_NUMA_THREADS" ]; then
        THREADS_EXT="${THREADS_EXT}${t},"
    fi
done
THREADS_EXT=${THREADS_EXT%,}

echo "Half Threads: $THREADS_HALF"
echo "Full Threads: $THREADS_FULL"
echo "Ext Threads:  $THREADS_EXT"
echo ""

# --- Split Models by Memory Requirements ---
print_header "Model Analysis"
SPLIT_SCRIPT="
import sys, os
sys.path.insert(0, '$(dirname "$0")')
try:
    from cpu_adam_profile.cpu_adam_benchmark import MODEL_CONFIGS, get_cpu_info 
except:
    # If running from benchmark dir directly
    sys.path.insert(0, '$(dirname "$0")/..')
    from cpu_adam_profile.cpu_adam_benchmark import MODEL_CONFIGS

models = '${INPUT_MODELS}'.split(',')
small = []
large = []

# Get simple node 0 mem capacity estimate (in GB).
# If cannot detect, assume 256GB as a safe fallback or parsed from args
import subprocess
try:
    res = subprocess.run(['numactl', '--hardware'], capture_output=True, text=True)
    # Parse 'node 0 size: X MB'
    import re
    m = re.search(r'node 0 size: (\d+) MB', res.stdout)
    if m:
        limit_gb = float(m.group(1)) / 1024 * 0.9 # 90% head room
    else:
        limit_gb = 512.0 # Fallback
except:
    limit_gb = 512.0

for m in models:
    if m not in MODEL_CONFIGS: continue
    req = MODEL_CONFIGS[m].estimate_optimizer_memory_gb()
    if req < limit_gb:
        small.append(m)
    else:
        large.append(m)

print(','.join(small) + '|' + ','.join(large))
"

OUT=$(python3 -c "$SPLIT_SCRIPT")
MODELS_SMALL=$(echo $OUT | cut -d'|' -f1)
MODELS_LARGE=$(echo $OUT | cut -d'|' -f2)

echo "Small Models (Fit Node 0): $MODELS_SMALL"
echo "Large Models (Need Interleave): $MODELS_LARGE"
echo ""

# --- Phase 1: Small Models Scaling (Baseline) ---
if [ -n "$MODELS_SMALL" ]; then
    print_header "Phase 1: Small Models Baseline (Node 0)"
    SAME_NUMA_FILE="${RESULTS_DIR}/${TIMESTAMP}_same_numa.json"
    
    # 1a. Run within single NUMA (Baseline for scaling & NUMA compare)
    numactl --cpunodebind=0 --membind=0 \
        python3 "$(dirname "$0")/cpu_adam_benchmark.py" \
        --models "${MODELS_SMALL}" \
        --threads "${THREADS_HALF}" \
        --warmup "${WARMUP}" \
        --iters "${ITERS}" \
        --output "${SAME_NUMA_FILE}"
        
    echo -e "${GREEN}✓ Baseline saved to ${SAME_NUMA_FILE}${NC}"

    # 1b. Extend to Full System (if we have more threads than one NUMA)
    if [ -n "$THREADS_EXT" ] && [ "$NUMA_NODES" -gt 1 ]; then
        echo "Extending scaling to full system (${THREADS_EXT} threads)..."
        EXT_FILE="${RESULTS_DIR}/${TIMESTAMP}_scaling_ext.json"
        
        numactl --interleave=all \
            python3 "$(dirname "$0")/cpu_adam_benchmark.py" \
            --models "${MODELS_SMALL}" \
            --threads "${THREADS_EXT}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}" \
            --output "${EXT_FILE}"
        echo -e "${GREEN}✓ Extension saved to ${EXT_FILE}${NC}"
    fi
fi

# --- Phase 1c: Large Models Scaling ---
if [ -n "$MODELS_LARGE" ]; then
    print_header "Phase 1c: Large Models Scaling (Interleave)"
    LARGE_FILE="${RESULTS_DIR}/${TIMESTAMP}_large.json"
    
    numactl --interleave=all \
        python3 "$(dirname "$0")/cpu_adam_benchmark.py" \
        --models "${MODELS_LARGE}" \
        --threads "${THREADS_FULL}" \
        --warmup "${WARMUP}" \
        --iters "${ITERS}" \
        --output "${LARGE_FILE}"
        
    echo -e "${GREEN}✓ Large models saved to ${LARGE_FILE}${NC}"
fi

# --- Phase 2: Cross NUMA (Small Models Only) ---
if [ -n "$MODELS_SMALL" ] && [ "$NUMA_NODES" -ge 2 ]; then
    print_header "Phase 2: Cross NUMA Comparison"
    CROSS_NUMA_FILE="${RESULTS_DIR}/${TIMESTAMP}_cross_numa_cpu0_mem1.json"
    
    numactl --cpunodebind=0 --membind=1 \
        python3 "$(dirname "$0")/cpu_adam_benchmark.py" \
        --models "${MODELS_SMALL}" \
        --threads "${THREADS_HALF}" \
        --warmup "${WARMUP}" \
        --iters "${ITERS}" \
        --output "${CROSS_NUMA_FILE}"
        
    echo -e "${GREEN}✓ Cross-NUMA saved to ${CROSS_NUMA_FILE}${NC}"
fi

# --- Phase 3: Reporting ---
print_header "Phase 3: Generating Reports"

FILES_TO_PLOT=""
[ -f "$SAME_NUMA_FILE" ] && FILES_TO_PLOT="$FILES_TO_PLOT $SAME_NUMA_FILE"
[ -f "$EXT_FILE" ] && FILES_TO_PLOT="$FILES_TO_PLOT $EXT_FILE"
[ -f "$LARGE_FILE" ] && FILES_TO_PLOT="$FILES_TO_PLOT $LARGE_FILE"

if [ -n "$FILES_TO_PLOT" ]; then
    echo "Generating Scaling Plot..."
    python3 "$(dirname "$0")/analyze_results.py" $FILES_TO_PLOT \
        --plot --output "${RESULTS_DIR}/${TIMESTAMP}_scaling.png"
fi

if [ -f "$SAME_NUMA_FILE" ] && [ -f "$CROSS_NUMA_FILE" ]; then
    echo "Generating NUMA Comparison..."
    python3 "$(dirname "$0")/analyze_results.py" \
        "${SAME_NUMA_FILE}" "${CROSS_NUMA_FILE}" \
        --compare-numa --plot \
        --output "${RESULTS_DIR}/${TIMESTAMP}_numa_comparison.png"
fi

echo -e "${GREEN}Benchmarks Completed!${NC}"
