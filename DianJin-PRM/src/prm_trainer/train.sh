#!/bin/bash
# PRM 训练启动脚本
# 用法: bash train.sh [额外参数...]
# 日志自动保存到 src/prm_trainer/logs/train_<timestamp>.log

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

echo "========================================" | tee -a "${LOG_FILE}"
echo "PRM Training Started" | tee -a "${LOG_FILE}"
echo "Time: $(date)" | tee -a "${LOG_FILE}"
echo "Command: python3 prm_train.py $*" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

python3 prm_train.py "$@" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "PRM Training Finished" | tee -a "${LOG_FILE}"
echo "Time: $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
