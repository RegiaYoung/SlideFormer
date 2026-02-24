#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=64

# 固定参数
SEQ_LEN=1024
WARM_STEPS=2
TEST_STEPS=3
USE_GEMINI=1  # 1表示使用Gemini插件，0表示使用普通FSDP
USE_XFORMERS=1  # 1表示使用flash attention，0表示不使用
USE_GRAD_CP=1  # 1表示使用梯度检查点，0表示不使用

# 批次大小选项 4 8 16 32 64 128
BATCH_SIZES=(8)

# 模型配置选项
MODEL_CONFIGS=(
    "Llama-3.1-8B-Instruct"
    # "Mistral-Small-24B-Instruct"
    # "Qwen2.5-3B-Instruct"
    # "Qwen2.5-7B-Instruct"
    # "Qwen2.5-14B-Instruct"
    # "Qwen2.5-32B-Instruct"
    # "Qwen2.5-72B-Instruct"
)

# 精度和Offload选项
OFFLOAD_OPTIM_FRAC=1.0
OFFLOAD_PARAM_FRAC=1.0

# 设置输出CSV文件名
RESULT_FILE="colossalai_bench_$(date +%Y%m%d_%H%M%S).csv"
echo "ColossalAI Benchmark Begin..."
echo "结果保存至: $RESULT_FILE"
echo "序列长度: $SEQ_LEN, 预热步数: $WARM_STEPS, 测试步数: $TEST_STEPS"
echo "Gemini插件: $([ $USE_GEMINI -eq 1 ] && echo '开启' || echo '关闭')"
echo "Flash Attention: $([ $USE_XFORMERS -eq 1 ] && echo '开启' || echo '关闭')"
echo "梯度检查点: $([ $USE_GRAD_CP -eq 1 ] && echo '开启' || echo '关闭')"
echo "优化器Offload比例: $OFFLOAD_OPTIM_FRAC, 参数Offload比例: $OFFLOAD_PARAM_FRAC"

# 遍历所有模型
for CONFIG in "${MODEL_CONFIGS[@]}"; do
    echo "===== 评测模型配置: $CONFIG ====="
    
    # 遍历所有批次大小
    for BS in "${BATCH_SIZES[@]}"; do
        echo "----- 批次大小: $BS -----"
        
        # 设置命令行参数
        GRAD_FLAG=""
        if [ $USE_GRAD_CP -eq 1 ]; then
            GRAD_FLAG="-g"
        fi
        
        XFORMERS_FLAG=""
        if [ $USE_XFORMERS -eq 1 ]; then
            XFORMERS_FLAG="-x"
        fi
        
        PLUGIN_TYPE=""
        if [ $USE_GEMINI -eq 1 ]; then
            PLUGIN_TYPE="gemini"
        else
            PLUGIN_TYPE="fsdp"
        fi
        # 运行benchmark
        echo "开始运行 ColossalAI benchmark: 模型=$CONFIG, 批次大小=$BS"

        if [[ $CONFIG =~ "Qwen2.5-72B-Instruct" || $CONFIG =~ "Qwen2.5-32B-Instruct" ]]; then
            colossalai run --nproc_per_node 1 benchmark.py \
                -p $PLUGIN_TYPE \
                -c $CONFIG \
                -b $BS \
                -l $SEQ_LEN \
                -s $(($WARM_STEPS + $TEST_STEPS)) \
                -i $WARM_STEPS \
                $GRAD_FLAG \
                $XFORMERS_FLAG \
                --offload_optim_frac $OFFLOAD_OPTIM_FRAC \
                --offload_param_frac $OFFLOAD_PARAM_FRAC \
                --result_file $RESULT_FILE
        else
            numactl --cpunodebind=0 --membind=0 colossalai run --nproc_per_node 1 benchmark.py \
                -p $PLUGIN_TYPE \
                -c $CONFIG \
                -b $BS \
                -l $SEQ_LEN \
                -s $(($WARM_STEPS + $TEST_STEPS)) \
                -i $WARM_STEPS \
                $GRAD_FLAG \
                $XFORMERS_FLAG \
                --offload_optim_frac $OFFLOAD_OPTIM_FRAC \
                --offload_param_frac $OFFLOAD_PARAM_FRAC \
                --result_file $RESULT_FILE
        fi

        echo "完成批次大小 $BS 的测试，结果已保存"
        echo "----------------------------------------"
        
    done
done

echo "ColossalAI Benchmark 完成，结果已保存至 $RESULT_FILE"
