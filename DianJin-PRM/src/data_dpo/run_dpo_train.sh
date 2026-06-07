#!/bin/bash
set -x

export PATH="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin:$PATH"
export OMP_NUM_THREADS=1
export HF_HOME="/root/autodl-tmp/huggingface"
export TRANSFORMERS_CACHE="/root/autodl-tmp/huggingface"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/huggingface"
export TRITON_CACHE_DIR="/root/autodl-tmp/triton"

PROJECT_ROOT="/root/workspace/qwen-dianjin/DianJin-PRM"
TRAINER_SCRIPT="${PROJECT_ROOT}/src/dpo_trainer/dpo_train.py"
DS_CONFIG="${PROJECT_ROOT}/src/dpo_trainer/ds_config.json"
DATA_DIR="${PROJECT_ROOT}/src/data_dpo"
LOG_DIR="${DATA_DIR}/logs"
OUTPUT_DIR="/root/autodl-tmp/dpo_checkpoint"

TRAIN_DATA="${DATA_DIR}/train/dpo_train.jsonl"
VAL_DATA="${DATA_DIR}/validate/dpo_val.jsonl"

EXPERIMENT_TAG=${1:-"dpo_$(date +%Y%m%d_%H%M%S)"}
MAX_TRAIN_SAMPLES=${2:-""}

mkdir -p "${LOG_DIR}"
mkdir -p "/root/autodl-tmp/triton"

LOG_FILE="${LOG_DIR}/${EXPERIMENT_TAG}_output.log"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "==========================================" | tee -a "${LOG_FILE}"
echo "DPO Training - ${EXPERIMENT_TAG}" | tee -a "${LOG_FILE}"
echo "Start: $(date)" | tee -a "${LOG_FILE}"
echo "Train data: ${TRAIN_DATA}" | tee -a "${LOG_FILE}"
echo "Val   data: ${VAL_DATA}" | tee -a "${LOG_FILE}"
echo "Log:        ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Output:     ${OUTPUT_DIR}/${EXPERIMENT_TAG}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

CMD="python ${TRAINER_SCRIPT} \
    --train_data ${TRAIN_DATA} \
    --eval_data ${VAL_DATA} \
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
    --experiment_tag ${EXPERIMENT_TAG}"

if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    CMD="${CMD} --max_train_samples ${MAX_TRAIN_SAMPLES}"
fi

echo "Running: ${CMD}" | tee -a "${LOG_FILE}"

eval ${CMD} 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
echo "Training finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
