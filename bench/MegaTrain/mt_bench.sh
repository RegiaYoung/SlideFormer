#!/bin/bash
# MegaTrain baseline benchmark runner — mirrors ds_bench.sh / ColossalAI gemini.sh.
#
# MegaTrain stores parameters in CPU RAM and streams layers to the GPU on demand
# (params + optimizer state on CPU, per-interval activation checkpointing), so it
# is compared against SlideFormer under the same single-GPU CPU-offload setting.
#
# Prerequisites (one-time): build MegaTrain's CUDA extensions into the active env.
#   See README.md. The block below builds them automatically if missing.

set -e
cd "$(dirname "$0")"

# ---- bench parameters (kept aligned with ds_bench.sh) ----------------------
SEQ_LEN=1024
WARM_STEPS=2
TEST_STEPS=3
LR=1e-5
WEIGHT_DECAY=0.1
USE_BF16=1          # 1 -> bf16 (matches SlideFormer), 0 -> fp16
USE_LIGER=1        # 1 -> Liger kernel on (matches ds_bench.sh), 0 -> off

BATCH_SIZES=(4 8 16 32 64)

MODEL_PATHS=(
    # "/home/scc/models/Qwen3-1.7B/"
    "/home/scc/models/Llama-3.1-8B-Instruct/"
    # "/home/scc/models/Qwen2.5-7B-Instruct/"
    # "/home/scc/models/Qwen2.5-14B-Instruct/"
    # "/home/scc/models/Qwen2.5-32B-Instruct/"
    # "/home/scc/models/Qwen2.5-72B-Instruct/"
)

# MegaTrain memory knobs (its activation-checkpoint equivalent)
CHECKPOINT_INTERVAL=4
NUM_GRAD_SLABS=12

PRECISION_FLAG=""
PRECISION="fp16"
if [ $USE_BF16 -eq 1 ]; then
    PRECISION_FLAG="--use_bf16"
    PRECISION="bf16"
fi

LIGER_FLAG=""
LIGER_STATUS="off"
if [ $USE_LIGER -eq 1 ]; then
    LIGER_FLAG="--use_liger"
    LIGER_STATUS="on"
else
    LIGER_FLAG="--no_liger"
fi

RESULT_FILE="mt_bench_${SEQ_LEN}_${PRECISION}_${LIGER_STATUS}_$(date +%Y%m%d_%H%M%S).csv"
echo "Result save to: $RESULT_FILE"
echo "Precision: $PRECISION | Liger-kernel: $LIGER_STATUS | seq_len: $SEQ_LEN"

# ---- ensure CUDA extensions are built --------------------------------------
PYBIN=$(which python)
if ! $PYBIN -c "import infinity_memory_ops" 2>/dev/null; then
    echo "Building MegaTrain CUDA extension infinity_memory_ops ..."
    ( cd csrc && $PYBIN -m pip install . --no-build-isolation )
fi
if ! $PYBIN -c "import cuda_pipeline" 2>/dev/null; then
    echo "Building MegaTrain CUDA extension cuda_pipeline ..."
    $PYBIN setup.py build_ext --inplace
fi

# ---- run -------------------------------------------------------------------
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "===== Benchmark Model: $MODEL_NAME with MegaTrain (CPU offload) ====="
    for BS in "${BATCH_SIZES[@]}"; do
        echo "----- Batch Size: $BS -----"
        numactl --cpunodebind=0 --membind=0 python ./mt_bench.py \
            --model_path "$MODEL_PATH" \
            --seq_len $SEQ_LEN \
            --batch_size $BS \
            --warm_step $WARM_STEPS \
            --test_step $TEST_STEPS \
            --lr $LR \
            --weight_decay $WEIGHT_DECAY \
            $PRECISION_FLAG \
            $LIGER_FLAG \
            --checkpoint_interval $CHECKPOINT_INTERVAL \
            --num_grad_slabs $NUM_GRAD_SLABS \
            --result_file "$RESULT_FILE"
        sleep 5s
    done
done

echo "Done. Results in $RESULT_FILE"
