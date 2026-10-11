#!/bin/bash
# 生成两个 DPO 模型的端到端评估输出（单卡串行：step_mean → fusion_weighted）
# 用法: setsid bash run_gen_eval_2models.sh
set -u

PYTHON="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin/python"
DATA_DIR="/root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo"
CKPT_ROOT="/root/autodl-tmp/dpo_checkpoint"
LOG_DIR="$DATA_DIR/logs/redeval"
mkdir -p "$LOG_DIR"

export HF_HOME="/root/autodl-tmp/huggingface"
export TRANSFORMERS_CACHE="/root/autodl-tmp/huggingface"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/huggingface"
export OMP_NUM_THREADS=1

log() { echo "[$(date '+%F %T')] $*"; }

log "=== 生成 step_mean 模型输出 ==="
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$DATA_DIR/generate_eval.py" \
  --model dpo --lora "$CKPT_ROOT/redo_sm_b0.1_e1" \
  --out_dir "$DATA_DIR/test/dpo_step_mean" --force \
  > "$LOG_DIR/gen_step_mean.log" 2>&1
log "step_mean 生成结束 rc=$?"

log "=== 生成 fusion_weighted 模型输出 ==="
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$DATA_DIR/generate_eval.py" \
  --model dpo --lora "$CKPT_ROOT/redo_fw_b0.1_e1" \
  --out_dir "$DATA_DIR/test/dpo_fusion_weighted" --force \
  > "$LOG_DIR/gen_fusion_weighted.log" 2>&1
log "fusion_weighted 生成结束 rc=$?"

log "=== 全部生成完成 ==="
