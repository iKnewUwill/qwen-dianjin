#!/bin/bash
# DPO 重做计划 30 分钟进度快照监控
# 每 30 分钟向 status_30min.log 追加一次快照（时间/GPU/运行任务/阶段/执行记录）
set -u
BASE="/root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo/logs/redo"
STATUS_LOG="$BASE/status_30min.log"
MASTER="$BASE/master.log"
RECORD="/root/workspace/qwen-dianjin/DianJin-PRM/docs/训练总结/DPO重做执行记录.md"

while true; do
  {
    echo "======================================================"
    echo "快照时间: $(date '+%F %T')"
    echo "--- GPU (index,util,mem_used) ---"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo "--- 运行中的训练/评估进程 ---"
    ps -eo etime,cmd | grep -E "dpo_train|eval_dpo" | grep -v grep | sed 's/^/  /'
    echo "--- master.log 末尾 3 行 ---"
    tail -3 "$MASTER" | sed 's/^/  /'
    echo "--- 执行记录末尾 3 行 ---"
    tail -3 "$RECORD" | sed 's/^/  /'
    echo ""
  } >> "$STATUS_LOG"
  sleep 1800
done
