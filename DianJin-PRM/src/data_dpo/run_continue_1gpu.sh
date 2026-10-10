#!/bin/bash
# DPO 重做计划单卡续跑脚本（剩余 epoch=2 续跑 + epoch=3）
# 用法: setsid bash run_continue_1gpu.sh
set -u

PROJECT_ROOT="/root/workspace/qwen-dianjin/DianJin-PRM"
DATA_DIR="$PROJECT_ROOT/src/data_dpo"
TRAINER="$PROJECT_ROOT/src/dpo_trainer/dpo_train.py"
PYTHON="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin/python"
OUTPUT_ROOT="/root/autodl-tmp/dpo_checkpoint"
RANKINGS="$DATA_DIR/rankings"
LOG_DIR="$DATA_DIR/logs/redo"
SUMMARY_FILE="$PROJECT_ROOT/docs/训练总结/DPO重做执行记录.md"
MASTER_LOG="$LOG_DIR/master.log"
GPU=0

export HF_HOME="/root/autodl-tmp/huggingface"
export TRANSFORMERS_CACHE="/root/autodl-tmp/huggingface"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/huggingface"
export OMP_NUM_THREADS=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

append_summary() {
  local desc=$1 tag=$2 status=$3 val_acc=$4 test_acc=$5 margin=$6 train_min=$7
  echo "| $(date '+%m-%d %H:%M') | $desc | $tag | $status | $val_acc | $test_acc | $margin | ${train_min}min |" >> "$SUMMARY_FILE"
}

run_job() {
  local beta=$1 epoch=$2 tag=$3 desc=$4 resume=$5
  local train_data="$RANKINGS/fusion_weighted/train/dpo_train.jsonl"
  local val_data="$RANKINGS/fusion_weighted/validate/dpo_val.jsonl"
  local test_data="$RANKINGS/fusion_weighted/test/dpo_test.jsonl"
  local job_log="$LOG_DIR/job_$tag.log"
  local eval_json="$LOG_DIR/eval_$tag.json"
  local lora_path="$OUTPUT_ROOT/$tag"

  log "[JOB] $desc ($tag) 开始: beta=$beta epoch=$epoch resume=$resume gpu=$GPU"
  local t0=$(date +%s)

  local cmd=( "$PYTHON" "$TRAINER" \
    --train_data "$train_data" --eval_data "$val_data" \
    --output_dir "$OUTPUT_ROOT" --max_length 3072 \
    --learning_rate 5e-6 --beta "$beta" --lora_r 16 --lora_alpha 32 \
    --per_device_batch_size 1 --grad_accum 16 --num_epochs "$epoch" \
    --warmup_steps 50 --logging_steps 10 --save_steps 200 --eval_steps 200 \
    --bf16 --optim paged_adamw_8bit --loss_type sigmoid \
    --experiment_tag "$tag" --skip_plan_check )
  if [ "$resume" = "1" ]; then
    cmd+=( --resume_from_checkpoint )
  fi

  CUDA_VISIBLE_DEVICES=$GPU "${cmd[@]}" >> "$job_log" 2>&1
  local train_rc=$?
  local train_min=$(( ($(date +%s) - t0) / 60 ))

  if [ $train_rc -ne 0 ]; then
    log "[JOB] $desc ($tag) 训练失败 rc=$train_rc"
    append_summary "$desc" "$tag" "训练失败(rc=$train_rc)" "-" "-" "-" "$train_min"
    return 1
  fi

  log "[JOB] $desc ($tag) 训练完成(${train_min}min)，开始评估"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$DATA_DIR/eval_dpo.py" \
    --lora "$lora_path" --beta "$beta" \
    --validate_path "$val_data" --test_path "$test_data" \
    --output "$eval_json" >> "$job_log" 2>&1
  local eval_rc=$?

  if [ $eval_rc -ne 0 ] || [ ! -f "$eval_json" ]; then
    log "[JOB] $desc ($tag) 评估失败 rc=$eval_rc"
    append_summary "$desc" "$tag" "训练OK评估失败" "-" "-" "-" "$train_min"
    return 0
  fi

  local val_acc=$(python3 -c "import json;d=json.load(open('$eval_json'));print(f\"{d['validate']['preference_accuracy']:.4f}\")" 2>/dev/null || echo "-")
  local test_acc=$(python3 -c "import json;d=json.load(open('$eval_json'));print(f\"{d['test']['preference_accuracy']:.4f}\")" 2>/dev/null || echo "-")
  local val_margin=$(python3 -c "import json;d=json.load(open('$eval_json'));print(f\"{d['validate']['mean_margin']:.2f}\")" 2>/dev/null || echo "-")

  log "[JOB] $desc ($tag) 完成: val_acc=$val_acc test_acc=$test_acc val_margin=$val_margin"
  append_summary "$desc" "$tag" "OK" "$val_acc" "$test_acc" "$val_margin" "$train_min"
  return 0
}

log "======================================================"
log "单卡续跑启动: GPU=$GPU，剩余 epoch=2(续跑) + epoch=3"
log "======================================================"

run_job 0.1 2 redo_fw_b0.1_e2 "epoch消融(epoch=2,续跑)" 1
run_job 0.1 3 redo_fw_b0.1_e3 "epoch消融(epoch=3)" 0

log "======================================================"
log "单卡续跑全部完成"
log "======================================================"
