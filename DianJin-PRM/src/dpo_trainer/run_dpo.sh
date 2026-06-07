#!/bin/bash
# DPO 训练启动脚本
# 参考: 实验方案.md 阶段三
# 用法: bash run_dpo.sh [experiment_tag] [max_train_samples]
set -x

# 激活环境
export PATH="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin:$PATH"
source /root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin/activate

# OMP 设置
export OMP_NUM_THREADS=1

# 项目路径
PROJECT_ROOT="/root/workspace/qwen-dianjin/DianJin-PRM/src/dpo_trainer"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="/root/autodl-tmp/dpo_checkpoint"

# 实验标签
EXPERIMENT_TAG=${1:-"dpo_"$(date +%Y%m%d_%H%M%S)}
MAX_TRAIN_SAMPLES=${2:-""}

# 检查数据是否存在
TRAIN_DATA="${DATA_DIR}/dpo_train.jsonl"
TEST_DATA="${DATA_DIR}/dpo_test.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Run prepare_dpo_data.py first."
    exit 1
fi

echo "=========================================="
echo "DPO Training - ${EXPERIMENT_TAG}"
echo "实验方案: 实验方案.md 阶段三"
echo "Time: $(date)"
echo "=========================================="

# 检查实验方案
echo ""
echo "[Pre-flight] 实验方案校验将在训练脚本中自动执行"
echo "[Pre-flight] DeepSpeed ZeRO-2 + CPU Offload"
echo "[Pre-flight] LoRA rank=16, alpha=32"
echo "[Pre-flight] lr=5e-6, beta=0.1, bf16"
echo "[Pre-flight] max_length=3072"
echo ""

# 构建命令
CMD="python ${PROJECT_ROOT}/dpo_train.py \
    --train_data ${TRAIN_DATA} \
    --eval_data ${TEST_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --max_length 3072 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_batch_size 1 \
    --grad_accum 16 \
    --num_epochs 1 \
    --warmup_steps 50 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --bf16 \
    --optim paged_adamw_8bit \
    --loss_type sigmoid \
    --deepspeed ${PROJECT_ROOT}/ds_config.json \
    --experiment_tag ${EXPERIMENT_TAG}"

if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    CMD="${CMD} --max_train_samples ${MAX_TRAIN_SAMPLES}"
fi

echo "Running: ${CMD}"
echo ""

# 记录开始时间
echo "Training started at: $(date)" > "${OUTPUT_DIR}/${EXPERIMENT_TAG}_start_time.txt"

# 执行训练
eval ${CMD}
EXIT_CODE=$?

echo "Training finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check logs at ${OUTPUT_DIR}/${EXPERIMENT_TAG}/"
fi

exit ${EXIT_CODE}
