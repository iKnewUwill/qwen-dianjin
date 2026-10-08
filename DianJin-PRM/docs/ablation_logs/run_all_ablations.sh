#!/bin/bash
# 消融重训驱动脚本：串行跑完所有需重训的实验（训练 + 评估）
# 断点续跑：每个实验完成后写 .DONE 标记，重跑本脚本自动跳过已完成实验。
set -u
ROOT=/root/workspace/qwen-dianjin/DianJin-PRM
LOGDIR=$ROOT/docs/ablation_logs
PRM_PY=/root/autodl-tmp/miniconda3/envs/dianjin-prm/bin/python
DPO_PY=/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin/python
TRAIN_DATA=$ROOT/src/data_dpo/train/dpo_train.jsonl
VAL_DATA=$ROOT/src/data_dpo/validate/dpo_val.jsonl
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

PROGRESS=$LOGDIR/progress.log
RESULTS=$LOGDIR/results_summary.md
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

run_step() {
  local name=$1 marker=$2; shift 2
  if [ -f "$marker" ]; then log "SKIP $name (已完成)"; return 0; fi
  log "START $name"
  "$@" >> "$LOGDIR/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$marker"; log "DONE $name"; else log "FAIL $name rc=$rc"; fi
  return $rc
}

prm_train_eval() {
  local name=$1 epochs=$2 samples=$3 ckpt=$4
  local extra=""
  [ -n "$samples" ] && extra="--max_train_samples $samples"
  run_step "$name" "$ckpt/.DONE" \
    "$PRM_PY" "$ROOT/src/prm_trainer/prm_train.py" --epochs "$epochs" $extra --output_path "$ckpt" || return 1
  local last
  last=$(ls -d "$ckpt"/checkpoint-* 2>/dev/null | sort -V | tail -1)
  if [ -z "$last" ]; then log "FAIL $name 无 checkpoint"; return 1; fi
  log "EVAL $name @ $last"
  "$PRM_PY" "$ROOT/src/prm_trainer/inference.py" --ckpt_path "$last" --temperature 1.0 >> "$LOGDIR/$name.log" 2>&1
  {
    echo "## $name (epochs=$epochs, samples=${samples:-全部})"
    grep -E "test results|步骤准确率|正确步骤召回率|错误步骤召回率|置信度差距" "$LOGDIR/$name.log" | tail -5
    echo ""
  } >> "$RESULTS"
}

dpo_train_eval() {
  local name=$1 beta=$2 epochs=$3 tag=$4
  local ckpt=/root/autodl-tmp/dpo_checkpoint/$tag
  run_step "$name" "$ckpt/.DONE" \
    "$DPO_PY" "$ROOT/src/dpo_trainer/dpo_train.py" \
      --train_data "$TRAIN_DATA" --eval_data "$VAL_DATA" \
      --output_dir /root/autodl-tmp/dpo_checkpoint \
      --max_length 3072 --learning_rate 5e-6 --beta "$beta" \
      --lora_r 16 --lora_alpha 32 --per_device_batch_size 1 --grad_accum 16 \
      --num_epochs "$epochs" --warmup_steps 50 --logging_steps 10 \
      --save_steps 200 --eval_steps 200 --bf16 \
      --optim paged_adamw_8bit --loss_type sigmoid \
      --experiment_tag "$tag" --skip_plan_check || return 1
  log "EVAL $name"
  "$DPO_PY" "$ROOT/src/data_dpo/eval_dpo.py" --lora "$ckpt" --beta "$beta" >> "$LOGDIR/$name.log" 2>&1
  {
    echo "## $name (beta=$beta, epochs=$epochs)"
    grep -E "validate:|test:" "$LOGDIR/$name.log" | tail -2
    echo ""
  } >> "$RESULTS"
}

# 结果汇总文件（首次启动时覆盖，后续追加）
if [ ! -f "$RESULTS" ]; then
  echo "# 消融重训结果汇总" > "$RESULTS"
  echo "首次启动: $(date)" >> "$RESULTS"
  echo "" >> "$RESULTS"
fi

log "=== 重训驱动启动 ==="

prm_train_eval ep3      3   ""  /root/autodl-tmp/checkpoint_prm_ab/ep3
dpo_train_eval beta_005 0.05 1  dpo_beta_005
dpo_train_eval beta_050 0.50 1  dpo_beta_050
dpo_train_eval beta_001 0.01 1  dpo_beta_001
prm_train_eval d500     5   500 /root/autodl-tmp/checkpoint_prm_ab/d500
prm_train_eval d1000    5   1000 /root/autodl-tmp/checkpoint_prm_ab/d1000
dpo_train_eval de2      0.1  2  dpo_epoch_2
dpo_train_eval de3      0.1  3  dpo_epoch_3

log "=== 全部完成 ==="
