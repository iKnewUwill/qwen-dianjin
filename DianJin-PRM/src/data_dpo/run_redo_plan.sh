#!/bin/bash
# DPO 重做计划编排脚本（2 卡调度 + 评估 + 总结 + GPU 监控）
# 用法: bash run_redo_plan.sh  （建议 nohup 后台运行）
set -u

# ===== 配置 =====
PROJECT_ROOT="/root/workspace/qwen-dianjin/DianJin-PRM"
DATA_DIR="$PROJECT_ROOT/src/data_dpo"
TRAINER="$PROJECT_ROOT/src/dpo_trainer/dpo_train.py"
PYTHON="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin/python"
OUTPUT_ROOT="/root/autodl-tmp/dpo_checkpoint"
CACHE="$DATA_DIR/pre/scored_cache.jsonl"
RANKINGS="$DATA_DIR/rankings"
LOG_DIR="$DATA_DIR/logs/redo"
SUMMARY_FILE="$PROJECT_ROOT/docs/训练总结/DPO重做执行记录.md"
MASTER_LOG="$LOG_DIR/master.log"

export HF_HOME="/root/autodl-tmp/huggingface"
export TRANSFORMERS_CACHE="/root/autodl-tmp/huggingface"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/huggingface"
export OMP_NUM_THREADS=1

mkdir -p "$LOG_DIR" "$RANKINGS"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

# ===== GPU 监控（后台每 5 分钟采样一次） =====
monitor_gpu() {
  local out="$LOG_DIR/gpu_monitor.log"
  echo "=== monitor start $(date '+%F %T') ===" >> "$out"
  while true; do
    {
      echo "=== $(date '+%F %T') ==="
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
    } >> "$out"
    sleep 300
  done
}

# ===== 总结追加 =====
init_summary() {
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  cat > "$SUMMARY_FILE" <<'EOF'
# DPO 重做计划执行记录

> 依据 `docs/重做计划.md`。排序口径：步骤平均 + 加权融合（步骤级 + 轨迹级）。2 卡并行。
> 9 个 DPO 训练：偏好构造方式消融 4 组 + β 消融 3 组 + epoch 消融 2 组。

## 执行状态

| 完成时间 | 任务 | tag | 状态 | val_acc | test_acc | val_margin | 训练耗时 |
|----------|------|-----|------|---------|----------|------------|----------|
EOF
}

append_summary() {
  local desc=$1 tag=$2 status=$3 val_acc=$4 test_acc=$5 margin=$6 train_min=$7
  echo "| $(date '+%m-%d %H:%M') | $desc | $tag | $status | $val_acc | $test_acc | $margin | ${train_min}min |" >> "$SUMMARY_FILE"
}

# ===== 单任务：训练 + 评估 + 总结 =====
run_job() {
  local mode=$1 beta=$2 epoch=$3 tag=$4 desc=$5 gpu=$6
  local train_data="$RANKINGS/$mode/train/dpo_train.jsonl"
  local val_data="$RANKINGS/$mode/validate/dpo_val.jsonl"
  local test_data="$RANKINGS/$mode/test/dpo_test.jsonl"
  local job_log="$LOG_DIR/job_$tag.log"
  local eval_json="$LOG_DIR/eval_$tag.json"
  local lora_path="$OUTPUT_ROOT/$tag"

  log "[JOB] $desc ($tag) 开始: mode=$mode beta=$beta epoch=$epoch gpu=$gpu"
  local t0=$(date +%s)

  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$TRAINER" \
    --train_data "$train_data" --eval_data "$val_data" \
    --output_dir "$OUTPUT_ROOT" --max_length 3072 \
    --learning_rate 5e-6 --beta "$beta" --lora_r 16 --lora_alpha 32 \
    --per_device_batch_size 1 --grad_accum 16 --num_epochs "$epoch" \
    --warmup_steps 50 --logging_steps 10 --save_steps 200 --eval_steps 200 \
    --bf16 --optim paged_adamw_8bit --loss_type sigmoid \
    --experiment_tag "$tag" --skip_plan_check \
    >> "$job_log" 2>&1
  local train_rc=$?
  local train_min=$(( ($(date +%s) - t0) / 60 ))

  if [ $train_rc -ne 0 ]; then
    log "[JOB] $desc ($tag) 训练失败 rc=$train_rc"
    append_summary "$desc" "$tag" "训练失败(rc=$train_rc)" "-" "-" "-" "${train_min}min"
    return 1
  fi

  log "[JOB] $desc ($tag) 训练完成(${train_min}min)，开始评估"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$DATA_DIR/eval_dpo.py" \
    --lora "$lora_path" --beta "$beta" \
    --validate_path "$val_data" --test_path "$test_data" \
    --output "$eval_json" >> "$job_log" 2>&1
  local eval_rc=$?

  if [ $eval_rc -ne 0 ] || [ ! -f "$eval_json" ]; then
    log "[JOB] $desc ($tag) 评估失败 rc=$eval_rc"
    append_summary "$desc" "$tag" "训练OK评估失败" "-" "-" "-" "${train_min}min"
    return 0
  fi

  local val_acc=$(python3 -c "import json;d=json.load(open('$eval_json'));print(f\"{d['validate']['preference_accuracy']:.4f}\")" 2>/dev/null || echo "-")
  local test_acc=$(python3 -c "import json;d=json.load(open('$eval_json'));print(f\"{d['test']['preference_accuracy']:.4f}\")" 2>/dev/null || echo "-")
  local val_margin=$(python3 -c "import json;d=json.load(open('$eval_json'));print(f\"{d['validate']['mean_margin']:.2f}\")" 2>/dev/null || echo "-")

  log "[JOB] $desc ($tag) 完成: val_acc=$val_acc test_acc=$test_acc val_margin=$val_margin"
  append_summary "$desc" "$tag" "OK" "$val_acc" "$test_acc" "$val_margin" "${train_min}min"
  return 0
}

# ===== 任务列表：mode|beta|epoch|tag|描述 =====
JOBS=(
  "fusion_weighted|0.1|1|redo_fw_b0.1_e1|完整方案(加权融合)"
  "trajectory|0.1|1|redo_traj_b0.1_e1|基线组(仅轨迹级)"
  "step_mean|0.1|1|redo_sm_b0.1_e1|实验组A(仅步骤平均)"
  "fusion_static|0.1|1|redo_fs_b0.1_e1|实验组B(静态α=0.5)"
  "fusion_weighted|0.01|1|redo_fw_b0.01_e1|β消融(β=0.01)"
  "fusion_weighted|0.05|1|redo_fw_b0.05_e1|β消融(β=0.05)"
  "fusion_weighted|0.50|1|redo_fw_b0.50_e1|β消融(β=0.50)"
  "fusion_weighted|0.1|2|redo_fw_b0.1_e2|epoch消融(epoch=2)"
  "fusion_weighted|0.1|3|redo_fw_b0.1_e3|epoch消融(epoch=3)"
)

# ===== 主流程 =====
log "======================================================"
log "DPO 重做计划启动: 2 卡并行, 9 个训练任务"
log "======================================================"

init_summary
monitor_gpu &
MONITOR_PID=$!

# ---- Phase 1: PRM 打分 + 缓存 ----
if [ -f "$CACHE" ]; then
  log "[Phase1] 缓存已存在，跳过打分: $CACHE"
else
  log "[Phase1] 开始 PRM 打分 + 缓存（GPU0, 约1h）..."
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$DATA_DIR/build_dpo_dataset.py" \
    --cache_path "$CACHE" >> "$LOG_DIR/score.log" 2>&1
  if [ $? -ne 0 ]; then
    log "[Phase1] 打分失败，终止"
    kill $MONITOR_PID 2>/dev/null
    exit 1
  fi
  log "[Phase1] 打分完成: $CACHE"
fi

# ---- Phase 2: 生成 4 种排序数据集 ----
log "[Phase2] 生成 4 种排序数据集..."
for mode in trajectory step_mean fusion_static fusion_weighted; do
  if [ -f "$RANKINGS/$mode/test/dpo_test.jsonl" ]; then
    log "[Phase2] $mode 已存在，跳过"
    continue
  fi
  "$PYTHON" "$DATA_DIR/build_dpo_dataset.py" \
    --from_cache --cache_path "$CACHE" --ranking_mode "$mode" \
    --output_dir "$RANKINGS/$mode" >> "$LOG_DIR/gen_$mode.log" 2>&1
  log "[Phase2] $mode 生成完成"
done

# ---- Phase 3: 9 个训练（2 卡并行） ----
log "[Phase3] 开始训练（2 卡并行，每轮 2 个任务）"
i=0
n=${#JOBS[@]}
while [ $i -lt $n ]; do
  IFS='|' read -r mode beta epoch tag desc <<< "${JOBS[$i]}"
  run_job "$mode" "$beta" "$epoch" "$tag" "$desc" 0 &
  pid0=$!

  if [ $((i + 1)) -lt $n ]; then
    IFS='|' read -r mode beta epoch tag desc <<< "${JOBS[$((i + 1))]}"
    run_job "$mode" "$beta" "$epoch" "$tag" "$desc" 1 &
    pid1=$!
    wait $pid0 $pid1
    i=$((i + 2))
  else
    wait $pid0
    i=$((i + 1))
  fi
done

log "[Phase3] 全部 9 个训练任务完成"

kill $MONITOR_PID 2>/dev/null
log "======================================================"
log "DPO 重做计划全部完成。总结见 $SUMMARY_FILE"
log "======================================================"
