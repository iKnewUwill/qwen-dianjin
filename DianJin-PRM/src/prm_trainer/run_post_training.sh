#!/bin/bash
set -euo pipefail

# PRM训练完成后的一键执行脚本
# 1. 提取loss曲线
# 2. 构建DPO数据集（使用新版PRM + 温度参数）
# 3. 分析DPO数据margin分布

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/root/workspace/qwen-dianjin/DianJin-PRM/src"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/prm_trainer/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/post_training_${TIMESTAMP}.log"

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG_FILE}"
}

echo "========================================" | tee -a "${LOG_FILE}"
log "PRM后处理流程开始"
log "时间: $(date)"
echo "========================================" | tee -a "${LOG_FILE}"

# Step 1: Extract PRM loss
log ""
log "=" | tee -a "${LOG_FILE}"
log "Step 1/3: 提取PRM训练loss曲线"
log "=" | tee -a "${LOG_FILE}"

export PATH="/root/autodl-tmp/miniconda3/envs/dianjin-prm/bin:$PATH"
python3 "${SCRIPT_DIR}/extract_prm_loss.py" /root/autodl-tmp/checkpoint_prm_v2 2>&1 | tee -a "${LOG_FILE}"

# Step 2: Build DPO dataset
log ""
log "=" | tee -a "${LOG_FILE}"
log "Step 2/3: 构建DPO数据集（温度=2.0）"
log "=" | tee -a "${LOG_FILE}"

export PATH="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin:$PATH"
python3 "${PROJECT_ROOT}/data_dpo/build_dpo_dataset.py" 2>&1 | tee -a "${LOG_FILE}"

# Step 3: Analyze margins
log ""
log "=" | tee -a "${LOG_FILE}"
log "Step 3/3: 分析DPO数据margin分布"
log "=" | tee -a "${LOG_FILE}"

export PATH="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin:$PATH"
python3 "${PROJECT_ROOT}/data_dpo/analyze_dpo_margins.py" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
log "PRM后处理流程完成"
log "时间: $(date)"
log "日志: ${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
