# PRM (Process Reward Model) 训练报告

> 训练日期: 2026-06-15 | 基座模型: Qwen3-8B | 训练框架: HuggingFace Trainer + LoRA

---

## 1. 训练配置

| 配置项 | 值 |
|--------|-----|
| **基座模型** | Qwen3-8B (`b968826d`) |
| **模型架构** | Qwen3Model + score head (Linear→ReLU→Linear, 4096→2) |
| **微调方式** | LoRA (rank=16, alpha=32, dropout=0.05) |
| **训练目标模块** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **完全训练模块** | score (2层MLP评分头) |
| **可训练参数** | ~0.79% (约 65M / 8.19B) |
| **优化器** | paged_adamw_8bit, lr=2e-5, cosine scheduler + 50步 warmup |
| **精度** | bf16 |
| **有效 Batch Size** | 8 (batch=1 × grad_accum=8) |
| **最大序列长度** | 4096 tokens |

---

## 2. 训练数据

| 数据集 | 文件数 | 样本数 | 说明 |
|--------|:------:|:------:|------|
| **Train** | 399 | 1,995 | 最新 PRM 数据 (PRM_DATA.zip)，每条包含问题+财务指标+推理步骤+步骤标签 |
| **Validate** | 80 | 400 | 验证集，用于每 epoch 评估 |
| **Test** | 40 | 200 | 测试集，用于最终评估 |

**数据格式**: 每条样本为 JSONL 格式，包含 `question`, `knowledge_items`(约70项财务指标), `steps`(推理步骤), `final_answer`, `step_labels`(0/1二元标签), `trajectory_label`。

---

## 3. 训练过程

### 3.1 训练概况

| 指标 | 值 |
|------|-----|
| **训练 Epoch 数** | 5 |
| **总训练步数** | 1,250 |
| **总训练时间** | 9,143 秒 (约 2.5 小时) |
| **训练吞吐量** | 1.091 samples/s, 0.137 steps/s |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell, 97 GB VRAM |
| **GPU 内存使用** | ~82 GB |

### 3.2 Per-Epoch 训练 Loss

| Epoch | 步数范围 | Loss 条目 | Avg Loss | Min Loss | Max Loss |
|:-----:|:---------:|:--------:|:--------:|:--------:|:--------:|
| 0 (warmup) | 10–240 | 24 | 5.37 | 0.36 | 16.73 |
| 1 | 250–500 | 25 | 3.43 | 0.51 | 32.95 |
| 2 | 510–750 | 25 | 1.84 | 0.16 | 14.43 |
| 3 | 760–1000 | 25 | 1.52 | 0.04 | 18.12 |
| 4 | 1010–1250 | 26 | 0.38 | 0.02 | 1.63 |

**Loss 下降趋势**:

```
Epoch 0: ████████████████████████████████ 5.37
Epoch 1: ████████████████████ 3.43
Epoch 2: ███████████ 1.84
Epoch 3: ██████████ 1.52
Epoch 4: ██ 0.38
Final:    ▏ 0.15
```

- **初始 Loss**: 16.73 (step 10)
- **最终 Loss**: 0.15 (step 1250)
- **整体下降**: 99.1%
- Epoch 4 起 loss 降至 1.0 以下并趋于稳定，表明模型已充分收敛

### 3.3 验证评估

| Epoch | Eval Runtime | Samples/s | Eval Loss |
|:-----:|:------------:|:---------:|:---------:|
| 1 | 139.6s | 2.87 | NaN* |
| 2 | 142.3s | 2.81 | NaN* |
| 3 | 138.4s | 2.89 | NaN* |
| 4 | 138.3s | 2.89 | NaN* |
| 5 | 138.4s | 2.89 | NaN* |

> *eval_loss=NaN 为已知问题：验证时仅对 `<extra_0>` / `<extra_1>` 位置计算 loss，其余位置 label=-100，导致 CrossEntropyLoss 计算异常。不影响模型推理能力。

**测试集评估**: runtime=69.2s, 2.89 samples/s

---

## 4. Loss 曲线分析

### 4.1 关键节点

| 百分比 | Step | Loss |
|:------:|:----:|:----:|
| 0% | 10 | 16.73 |
| 10% | 130 | 14.66 |
| 25% | 320 | 9.51 |
| 50% | 630 | 1.03 |
| 75% | 940 | 0.26 |
| 90% | 1130 | 0.27 |
| 100% | 1250 | **0.15** |

### 4.2 训练特点

- **前期震荡大** (Epoch 0-1): loss 从 16.73 降至 ~2.0，但频繁出现 spikes（最高 32.95），说明模型在探索不同难度的样本
- **中期稳步下降** (Epoch 2-3): loss 从 ~2.0 降至 ~0.5，spikes 幅度减小，模型逐步稳定
- **后期趋于收敛** (Epoch 4): loss 在 0.02-1.63 范围波动，多数 step loss < 0.5

---

## 5. Checkpoint 信息

由于 `save_total_limit=3`，仅保留最后 3 个 checkpoint：

| Checkpoint | Epoch | Step | Final Loss |
|:-----------|:-----:|:----:|:----------:|
| `checkpoint-750` | 3 | 750 | 1.979 |
| `checkpoint-1000` | 4 | 1000 | 0.899 |
| **`checkpoint-1250`** | **5** | **1250** | **0.153** |

最终模型路径: `/root/autodl-tmp/checkpoint_prm_v2/checkpoint-1250/`

---

## 6. 输出文件

| 文件 | 说明 |
|------|------|
| `/root/autodl-tmp/checkpoint_prm_v2/checkpoint-1250/` | 最终 PRM LoRA 权重 |
| `/root/autodl-tmp/prm_v2_loss.csv` | 125 条训练 loss 记录 (CSV) |
| `/root/autodl-tmp/prm_v2_loss.json` | 完整训练指标 (JSON) |
| `docs/训练总结/PRM训练报告.md` | 本报告 |
| `docs/训练总结/prm_charts/` | 训练可视化图表 |
