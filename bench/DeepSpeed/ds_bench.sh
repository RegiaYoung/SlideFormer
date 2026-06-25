#!/bin/bash

# 固定参数
SEQ_LEN=1024
WARM_STEPS=2
TEST_STEPS=3
USE_BF16=1  # 1表示使用bf16，0表示使用fp16
USE_LIGER=1  # 1表示使用Liger-kernel，0表示使用普通模型
LR=1e-5

# 批次大小选项 4 8 16 32 64 128
BATCH_SIZES=(4 8 16 32 64)

# 模型路径选项
MODEL_PATHS=(
    # "/home/scc/models/Llama-3.1-1B-Instruct/"
    # "/home/scc/models/Llama-3.1-3B-Instruct/"
    "/home/scc/models/Llama-3.1-8B-Instruct/"
    # "/home/scc/models/Llama-3.3-70B-Instruct/"
    # "/home/scc/models/Qwen2.5-3B-Instruct/"
    # "/home/scc/models/Qwen2.5-7B-Instruct/"
    # "/home/scc/models/Qwen2.5-14B-Instruct/"
    # "/home/scc/models/Qwen2.5-32B-Instruct/"
    # "/home/scc/models/Qwen2.5-72B-Instruct/"
    # "/home/scc/models/Mistral-Small-24B-Instruct-2501/"
)

# ZeRO阶段选项
ZERO_STAGES=(1 2 3)

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

# 设置输出CSV文件名，包含序列长度、精度和Liger状态
RESULT_FILE="ds_bench_${SEQ_LEN}_${PRECISION}_${LIGER_STATUS}_$(date +%Y%m%d_%H%M%S).csv"

echo "DeepSpeed Benchmark Begin..."
echo "Result save to: $RESULT_FILE"
echo "Precision: $PRECISION, Liger-kernel: $LIGER_STATUS"

rm -rf /RAID0/zero_stage_3

# 测试Vanilla DeepSpeed (不使用Offload)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     MODEL_NAME=$(basename $MODEL_PATH)
#     echo "===== Benchmark Model: $MODEL_NAME with ZeRO Only ====="
    
#     for ZERO in "${ZERO_STAGES[@]}"; do
#         echo "----- ZeRO Stage: $ZERO -----"
        
#         for BS in "${BATCH_SIZES[@]}"; do
#             echo "----- Batch Size: $BS -----"
            
#             python /home/scc/OOM/ds_bench.py \
#                           --model_path "$MODEL_PATH" \
#                           --seq_len $SEQ_LEN \
#                           --batch_size $BS \
#                           --warm_step $WARM_STEPS \
#                           --test_step $TEST_STEPS \
#                           $PRECISION_FLAG \
#                           $LIGER_FLAG \
#                           --lr $LR \
#                           --zero_stage $ZERO \
#                           --result_file $RESULT_FILE

#             echo "----------------------------------------"
#         done
#     done
# done

# 测试ZeRO-Offload (CPU Offload)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename $MODEL_PATH)
    echo "===== Benchmark Model: $MODEL_NAME with ZeRO-Offload ====="
    
    # ZeRO-Offload 需要至少ZeRO-2
    for ZERO in 3; do
        echo "----- ZeRO Stage: $ZERO with CPU Offload -----"
        
        for BS in "${BATCH_SIZES[@]}"; do
            echo "----- Batch Size: $BS -----"
            # Check for larger models (72B and 32B)
            if [[ "$MODEL_NAME" == "Qwen2.5-72B-Instruct" || "$MODEL_NAME" == "Qwen2.5-32B-Instruct" ]]; then
                python ./ds_bench.py \
                            --model_path "$MODEL_PATH" \
                            --seq_len $SEQ_LEN \
                            --batch_size $BS \
                            --warm_step $WARM_STEPS \
                            --test_step $TEST_STEPS \
                            $PRECISION_FLAG \
                            $LIGER_FLAG \
                            --lr $LR \
                            --zero_stage $ZERO \
                            --offload \
                            --result_file $RESULT_FILE
            else
                 numactl --cpunodebind=0 --membind=0 python ./ds_bench.py \
                          --model_path "$MODEL_PATH" \
                          --seq_len $SEQ_LEN \
                          --batch_size $BS \
                          --warm_step $WARM_STEPS \
                          --test_step $TEST_STEPS \
                          $PRECISION_FLAG \
                          $LIGER_FLAG \
                          --lr $LR \
                          --zero_stage $ZERO \
                          --offload \
                          --result_file $RESULT_FILE
            fi
            sleep 5s
            echo "----------------------------------------"
        done
    done
done

# # 测试ZeRO-Infinity (NVMe Offload)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     MODEL_NAME=$(basename $MODEL_PATH)
#     echo "===== Benchmark Model: $MODEL_NAME with ZeRO-Infinity ====="
    
#     # ZeRO-Infinity 需要ZeRO-3
#     ZERO=3
#     echo "----- ZeRO Stage: $ZERO with NVMe Offload -----"
    
#     for BS in "${BATCH_SIZES[@]}"; do
#         echo "----- Batch Size: $BS -----"
        
#         if [[ "$MODEL_PATH" == "Qwen2.5-72B-Instruct" || "$MODEL_PATH" == "Qwen2.5-32B-Instruct" ]]; then
#             python ./ds_bench.py \
#                         --model_path "$MODEL_PATH" \
#                         --seq_len $SEQ_LEN \
#                         --batch_size $BS \
#                         --warm_step $WARM_STEPS \
#                         --test_step $TEST_STEPS \
#                         $PRECISION_FLAG \
#                         $LIGER_FLAG \
#                         --lr $LR \
#                         --zero_stage $ZERO \
#                         --offload \
#                         --offload_nvme \
#                         --nvme_path "/RAID0" \
#                         --result_file $RESULT_FILE
#         else
#             numactl --cpunodebind=0 --membind=0 python ./ds_bench.py \
#                         --model_path "$MODEL_PATH" \
#                         --seq_len $SEQ_LEN \
#                         --batch_size $BS \
#                         --warm_step $WARM_STEPS \
#                         --test_step $TEST_STEPS \
#                         $PRECISION_FLAG \
#                         $LIGER_FLAG \
#                         --lr $LR \
#                         --zero_stage $ZERO \
#                         --offload \
#                         --offload_nvme \
#                         --nvme_path "/RAID0" \
#                         --result_file $RESULT_FILE
#         fi

#         rm -rf /RAID0/zero_stage_3
#         sleep 5s
#         echo "----------------------------------------"
#     done
# done