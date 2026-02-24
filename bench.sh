#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 固定参数
SEQ_LEN=1024
WARM_STEPS=1
TEST_STEPS=3
USE_BF16=1  # 1表示使用bf16，0表示使用fp16
USE_LIGER=1  # 1表示使用Liger-kernel，0表示使用普通模型

# 新增的NVME卸载参数
AC_OFFLOAD_NVME=0  # 1表示激活值卸载到NVME，0表示不卸载
NVME_OFFLOAD_FRACTION=0.0  # NVME卸载比例，0.0、0.5、1.0
OFFLOAD_DIR="${OFFLOAD_DIR:-${SCRIPT_DIR}/offload_dir}"  # 卸载目录（可通过环境变量覆盖）

# 批次大小选项
BATCH_SIZES=(16 32 64) # 24 32 48 64 128
# BATCH_SIZES=(32)

# 模型路径选项
MODEL_PATHS=(
    "/home/scc/models/Llama-3.1-8B-Instruct/"
    # "/home/scc/models/Llama-3.3-70B-Instruct/"
    # "/home/scc/models/Qwen2.5-14B-Instruct/"
    # "/home/scc/models/Mistral-Small-24B-Instruct-2501/"
)

# 精度选项设置
PRECISION_FLAG=""
if [ $USE_BF16 -eq 1 ]; then
    PRECISION_FLAG="--use_bf16"
    PRECISION="bf16"
else
    PRECISION="fp16"
fi

LIGER_FLAG=""
if [ $USE_LIGER -eq 1 ]; then
    LIGER_FLAG="--use_liger"
    LIGER_STATUS="on"
else
    LIGER_STATUS="off"
fi

# NVME卸载选项设置
AC_OFFLOAD_FLAG=""
if [ $AC_OFFLOAD_NVME -eq 1 ]; then
    AC_OFFLOAD_FLAG="--ac_offload_nvme"
    AC_STATUS="ac-nvme"
else
    AC_STATUS="ac-cpu"
fi

if [ "$NVME_OFFLOAD_FRACTION" = "0.0" ]; then
    OS_STATUS="os-cpu"
elif [ "$NVME_OFFLOAD_FRACTION" = "0.5" ]; then
    OS_STATUS="os-nvme-50"
elif [ "$NVME_OFFLOAD_FRACTION" = "1.0" ]; then
    OS_STATUS="os-nvme-100"
else
    echo "Invalid NVME Offload Fraction: $NVME_OFFLOAD_FRACTION"
    exit 1
fi

# 设置输出CSV文件名，包含序列长度、精度、Liger状态和NVME状态
mkdir -p "${SCRIPT_DIR}/outputs"
RESULT_FILE="${SCRIPT_DIR}/outputs/bench_${SEQ_LEN}_${PRECISION}_${LIGER_STATUS}_${AC_STATUS}_${OS_STATUS}_$(date +%Y%m%d_%H%M%S).csv"

echo "Benchmark Begin..."
echo "Result save to: $RESULT_FILE"
echo "Precision: $PRECISION, Liger-kernel: $LIGER_STATUS"
echo "AC NVME Offload: $([ $AC_OFFLOAD_NVME -eq 1 ] && echo 'on' || echo 'off'), Fraction: $NVME_OFFLOAD_FRACTION"
echo "Offload Directory: $OFFLOAD_DIR"

# 遍历所有模型
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    echo "===== Benchmark Model: $MODEL_NAME ====="
    
    # 遍历所有批次大小
    for BS in "${BATCH_SIZES[@]}"; do
        echo "----- Batch Size: $BS -----"
        
        if [[ "${USE_NUMACTL:-1}" -eq 0 ]]; then
            python "${SCRIPT_DIR}/main_bench.py" \
                --model_path "${MODEL_PATH}" \
                --seq_len $SEQ_LEN \
                --batch_size $BS \
                --warm_step $WARM_STEPS \
                --test_step $TEST_STEPS \
                $PRECISION_FLAG \
                $LIGER_FLAG \
                $AC_OFFLOAD_FLAG \
                --nvme_offload_fraction $NVME_OFFLOAD_FRACTION \
                --offload_dir $OFFLOAD_DIR \
                --result_file $RESULT_FILE
        else
            numactl --cpunodebind=0 --membind=0 python "${SCRIPT_DIR}/main_bench.py" \
                --model_path "${MODEL_PATH}" \
                --seq_len $SEQ_LEN \
                --batch_size $BS \
                --warm_step $WARM_STEPS \
                --test_step $TEST_STEPS \
                $PRECISION_FLAG \
                $LIGER_FLAG \
                $AC_OFFLOAD_FLAG \
                --nvme_offload_fraction $NVME_OFFLOAD_FRACTION \
                --offload_dir $OFFLOAD_DIR \
                --result_file $RESULT_FILE
        fi

        echo "----------------------------------------"
    done
done

echo "Benchmark finished, result saved to $RESULT_FILE"