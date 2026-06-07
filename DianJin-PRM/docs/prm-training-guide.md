# PRM 模型训练与评估操作手册

## 目录

1. [环境概览](#1-环境概览)
2. [模型训练操作方法](#2-模型训练操作方法)
3. [日志查看方法](#3-日志查看方法)
4. [Loss 下降过程数据存储位置](#4-loss-下降过程数据存储位置)
5. [模型评估代码执行方法](#5-模型评估代码执行方法)
6. [模型评估结果及保存位置](#6-模型评估结果及保存位置)

---

## 1. 环境概览

### 硬件

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 GB VRAM |
| CPU | Intel Xeon |
| 数据盘 | /root/autodl-tmp (150 GB) |

### 软件环境

| 项目 | 详情 |
|------|------|
| Conda 环境 | `/root/autodl-tmp/miniconda3/envs/dianjin-prm` |
| Python | 3.12.0 |
| PyTorch | 2.8.0+cu128 |
| transformers | 4.56.1 |
| peft | 0.17.1 |
| bitsandbytes | 0.49.2 |
| deepspeed | 0.17.6 |
| datasets | 4.0.0 |

### 目录结构

```
/root/workspace/qwen-dianjin/DianJin-PRM/
├── src/
│   ├── prm_trainer/          # PRM 训练脚本
│   │   ├── train.sh           # 训练启动脚本
│   │   ├── prm_train.py       # 训练主程序
│   │   ├── inference.py       # 推理/评估脚本
│   │   └── debug_loss.py      # 调试脚本
│   ├── model/
│   │   ├── fin_prm.py         # PRM 模型定义 (Qwen3ForProcessRewardModel)
│   │   ├── fin_config.py      # 模型配置 (Qwen3PRMConfig)
│   │   └── config.json        # 模型超参数配置
│   ├── data/
│   │   ├── train/             # 训练集 (310 个 JSONL 文件, 1330 条样本)
│   │   ├── validate/          # 验证集 (80 个 JSONL 文件, 240 条样本)
│   │   └── test/              # 测试集 (10 个 JSONL 文件, 30 条样本)
│   ├── dpo_trainer/           # DPO 训练 (阶段三, 独立环境)
│   └── verl/                  # VERL 强化学习框架 (备用)
└── docs/
    ├── prm-training-guide.md  # 本文档
    ├── run-report.md          # 早期运行报告
    └── experiment_record.md   # 实验记录
```

### 数据格式

每条样本为 JSON 对象，包含字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| question | string | 财务分析问题 |
| knowledge_items | dict | 财务指标键值对 (约70项) |
| steps | dict | 推理步骤，键为 "Step 1"/"Step 2"/"Step 3" |
| final_answer | string | 最终答案 |
| step_labels | list[int] | 步骤级标签，0=错误, 1=正确 |
| trajectory_label | int | 整体轨迹标签，0=错误, 1=正确 |

### 模型架构

```
Qwen3-8B backbone (Qwen3Model)
    ↓
hidden_states [batch, seq_len, 4096]
    ↓
score head: Linear(4096→4096) → ReLU → Linear(4096→2)
    ↓
logits [batch, seq_len, 2]  ← 在 <extra_0> / <extra_1> 位置做二分类
```

- **微调方式**: LoRA (rank=16, alpha=32), 仅训练 0.79% 参数
- **score head**: 全参数训练
- **特殊标记**: `<extra_0>` 分隔步骤, `<extra_1>` 标记末尾
- **损失函数**: CrossEntropyLoss，仅对特殊标记位置计算

---

## 2. 模型训练操作方法

### 2.1 启动训练

```bash
# 1. 激活环境
conda activate /root/autodl-tmp/miniconda3/envs/dianjin-prm

# 2. 设置线程数
export OMP_NUM_THREADS=1

# 3. 进入训练目录
cd /root/workspace/qwen-dianjin/DianJin-PRM/src/prm_trainer

# 4. 启动训练 (默认参数)
bash train.sh
```

### 2.2 自定义参数

`train.sh` 将所有参数透传给 `prm_train.py`，可通过命令行覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_data_dir` | `../data/train` | 训练集目录 |
| `--val_data_dir` | `../data/validate` | 验证集目录 |
| `--test_data_dir` | `../data/test` | 测试集目录 |
| `--output_path` | `/root/autodl-tmp/checkpoint` | 模型保存路径 |
| `--max_length` | **4096** | 最大序列长度 (训练数据最大 3216 tokens, 验证数据最大 ~4500) |
| `--epochs` | 30 | 训练轮数 |
| `--batch_size` | 1 | 每设备 batch size |
| `--grad_accum` | 8 | 梯度累积步数 (有效 batch = 8) |
| `--learning_rate` | 2e-5 | 学习率 |
| `--pretrained_model_path` | Qwen3-8B 路径 | 基座模型 |

### 2.3 训练命令示例

```bash
# 完整训练 (30 epochs)
bash train.sh --epochs 30 --max_length 4096

# 快速测试 (1 epoch, 小样本集)
bash train.sh --epochs 1 --max_length 4096 \
  --train_data_dir /root/workspace/qwen-dianjin/DianJin-PRM/src/data/test \
  --val_data_dir /root/workspace/qwen-dianjin/DianJin-PRM/src/data/test \
  --test_data_dir /root/workspace/qwen-dianjin/DianJin-PRM/src/data/test

# 保存到自定义路径
bash train.sh --output_path /root/autodl-tmp/checkpoint_myrun
```

### 2.4 训练配置细节

- **精度**: bf16
- **调度器**: cosine + 50 步 warmup
- **优化器**: paged_adamw_8bit (节省显存)
- **保存策略**: 每 epoch 保存一次, 保留最近 3 个 checkpoint
- **梯度检查点**: 关闭 (modules_to_save=["score"] 与 gradient_checkpointing 不兼容)
- **输出格式**: PEFT LoRA adapter (仅保存训练的参数，~231 MB)

### 2.5 训练后模型保存位置

```
/root/autodl-tmp/checkpoint/
├── checkpoint-835/    # epoch 5
├── checkpoint-1002/   # epoch 6
└── checkpoint-1169/   # epoch 7 (最新)
    ├── adapter_model.safetensors   # LoRA 权重 (231 MB)
    ├── adapter_config.json         # LoRA 配置
    ├── optimizer.pt                # 优化器状态 (可恢复训练)
    ├── scheduler.pt                # 调度器状态
    ├── trainer_state.json          # 训练状态 (loss, epoch 等)
    ├── training_args.bin           # 训练参数
    └── tokenizer 相关文件
```

---

## 3. 日志查看方法

### 3.1 实时训练日志

训练过程中标准输出会实时显示：
- 模型加载进度
- 每个 epoch 的训练进度条
- 验证评估结果

### 3.2 查看训练历史 (trainer_state.json)

每个 checkpoint 目录下的 `trainer_state.json` 包含完整的训练日志：

```bash
# 查看最新 checkpoint 的训练状态
python3 -c "
import json
with open('/root/autodl-tmp/checkpoint/checkpoint-1169/trainer_state.json') as f:
    s = json.load(f)
print(f'Total epochs: {s[\"epoch\"]}')
print(f'Total steps: {s[\"global_step\"]}')
print(f'Max planned steps: {s[\"max_steps\"]}')

# 查看所有 eval 记录
for entry in s['log_history']:
    if 'eval_loss' in entry:
        print(f'  Epoch {entry[\"epoch\"]:.1f}: eval_loss={entry[\"eval_loss\"]}, runtime={entry[\"eval_runtime\"]:.1f}s')

# 查看 loss 记录
losses = [(e['step'], e['loss']) for e in s['log_history'] if 'loss' in e]
print(f'Total loss entries: {len(losses)}')
print(f'First loss: step={losses[0][0]}, loss={losses[0][1]}')
print(f'Last loss: step={losses[-1][0]}, loss={losses[-1][1]}')
"
```

### 3.3 查看训练所用的硬件资源

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### 3.4 DPO 训练日志

DPO 训练日志位于 `DianJin-PRM/docs/grpo_run.log`（历史记录）。

---

## 4. Loss 下降过程数据存储位置

### 4.1 存储位置

训练 loss 数据存储在每个 checkpoint 的 `trainer_state.json` 中：

```
/root/autodl-tmp/checkpoint/
├── checkpoint-835/trainer_state.json    # epoch 1-5 的 loss
├── checkpoint-1002/trainer_state.json   # epoch 1-6 的 loss
└── checkpoint-1169/trainer_state.json   # epoch 1-7 的 loss (最完整)
```

### 4.2 各 checkpoint 的 loss 变化

| Checkpoint | Epochs | Steps | 初始 Loss | 最终 Loss |
|-----------|--------|-------|-----------|-----------|
| checkpoint-835 | 1-5 | 83 entries | 2736.26 (step 10) | 10.74 (step 830) |
| checkpoint-1002 | 1-6 | 100 entries | 2736.26 (step 10) | 0.13 (step 1000) |
| **checkpoint-1169** | 1-7 | 116 entries | 2736.26 (step 10) | **0.0001** (step 1160) |

### 4.3 提取 loss 数据为 CSV

```bash
python3 -c "
import json, csv
with open('/root/autodl-tmp/checkpoint/checkpoint-1169/trainer_state.json') as f:
    s = json.load(f)
losses = [(e['step'], e['loss']) for e in s['log_history'] if 'loss' in e and 'eval_loss' not in e]
with open('/root/autodl-tmp/loss_history.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['step', 'loss'])
    w.writerows(losses)
print(f'Exported {len(losses)} rows to /root/autodl-tmp/loss_history.csv')
"
```

### 4.4 Loss 曲线快速查看

```bash
python3 -c "
import json
with open('/root/autodl-tmp/checkpoint/checkpoint-1169/trainer_state.json') as f:
    s = json.load(f)
losses = [(e['step'], e['loss']) for e in s['log_history'] if 'loss' in e and 'eval_loss' not in e]
print('Step\tLoss')
for step, loss in losses:
    print(f'{step}\t{loss:.4f}')
"
```

---

## 5. 模型评估代码执行方法

### 5.1 推理脚本位置

```
/root/workspace/qwen-dianjin/DianJin-PRM/src/prm_trainer/inference.py
```

### 5.2 执行评估

```bash
# 1. 激活环境
conda activate /root/autodl-tmp/miniconda3/envs/dianjin-prm

# 2. 设置线程数
export OMP_NUM_THREADS=1

# 3. 运行推理
python3 /root/workspace/qwen-dianjin/DianJin-PRM/src/prm_trainer/inference.py
```

### 5.3 推理逻辑说明

1. **加载模型**: 加载 Qwen3-8B 基座 + LoRA adapter (checkpoint-1169)
2. **智能截断**: 自动裁剪 knowledge_items 部分，保证 `<extra_0>` 和 `<extra_1>` 不被截断
3. **逐样本推理**: 每个样本独立 tokenize → forward → 提取步骤级概率
4. **输出格式**: JSON，包含每个步骤的 prob_0、prob_1、预测标签、真实标签

### 5.4 单独评估一个样本

```python
import os, torch, sys
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'
sys.path.insert(0, '/root/workspace/qwen-dianjin/DianJin-PRM/src')
from transformers import AutoTokenizer, AutoModel
from model.fin_prm import Qwen3ForProcessRewardModel
from model.fin_config import Qwen3PRMConfig
from peft import PeftModel

ckpt_path = '/root/autodl-tmp/checkpoint/checkpoint-1169'
base_path = '/root/autodl-tmp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'

config = Qwen3PRMConfig.from_pretrained('/root/workspace/qwen-dianjin/DianJin-PRM/src/model/config.json')
model = Qwen3ForProcessRewardModel(config=config)
pretrained = AutoModel.from_pretrained(base_path)
model.model.load_state_dict(pretrained.state_dict(), strict=True)
model = PeftModel.from_pretrained(model, ckpt_path)
model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_0>', '<extra_1>']})

# 构造输入文本
text = "##Question\n问题文本\n\n##Knowledge\n键:值\n\n##Thinking Trajectory\n步骤1<extra_0>步骤2<extra_0>步骤3<extra_0>\n\n##Final Answer\n答案<extra_1>"
enc = tokenizer(text, truncation=True, max_length=4096, return_tensors='pt').to('cuda')

with torch.no_grad():
    logits = model(**enc).logits.float()

sep0 = tokenizer.encode('<extra_0>', add_special_tokens=False)[0]
sep1 = tokenizer.encode('<extra_1>', add_special_tokens=False)[0]
positions = (enc.input_ids[0] == sep0) | (enc.input_ids[0] == sep1)

for i, pos in enumerate(positions.nonzero(as_tuple=True)[0]):
    prob = torch.nn.functional.softmax(logits[0, pos], dim=-1)
    print(f'Step {i+1}: P(correct)={prob[1].item():.4f}, P(wrong)={prob[0].item():.4f}')
```

---

## 6. 模型评估结果及保存位置

### 6.1 结果文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `/root/autodl-tmp/inference_validate.json` | 228 KB | 验证集评估结果 (240 条) |
| `/root/autodl-tmp/inference_test.json` | 29 KB | 测试集评估结果 (30 条) |

### 6.2 结果格式

每条评估结果包含：

```json
{
  "index": 0,
  "question": "结合应收账款与存货周转效率...",
  "step_labels": [1, 0, 0],
  "trajectory_label": 1,
  "step_scores": [
    {
      "step_key": "Step 1",
      "true_label": 1,
      "pred_label": 1,
      "prob_1": 0.998,
      "result": "correct"
    },
    {
      "step_key": "Step 2",
      "true_label": 0,
      "pred_label": 0,
      "prob_1": 0.034,
      "result": "correct"
    }
  ],
  "final_score": {
    "prob_1": 1.0,
    "pred": 1,
    "true": 1,
    "result": "correct"
  }
}
```

### 6.3 评估指标

#### 验证集 (Validate, 240 条)

| 指标 | 数值 |
|------|------|
| 步骤级准确率 | **86.11%** (620/720) |
| 正确步骤平均置信度 | 0.9806 |
| 错误步骤平均置信度 | 0.9013 |
| 轨迹级准确率 | 143/240 |

#### 测试集 (Test, 30 条)

| 指标 | 数值 |
|------|------|
| 步骤级准确率 | **85.56%** (77/90) |
| 正确步骤平均置信度 | 0.9599 |
| 错误步骤平均置信度 | 0.8779 |
| 轨迹级准确率 | 27/30 |

### 6.4 快速查看评估摘要

```bash
python3 -c "
import json
for name in ['validate', 'test']:
    with open(f'/root/autodl-tmp/inference_{name}.json') as f:
        data = json.load(f)
    correct = sum(1 for r in data for s in r['step_scores'] if s['true_label'] is not None and s['pred_label'] == s['true_label'])
    total = sum(1 for r in data for s in r['step_scores'] if s['true_label'] is not None)
    print(f'{name}: {correct}/{total} = {correct/total:.4f}')
"
```

### 6.5 已知问题和注意事项

| 问题 | 说明 | 影响 |
|------|------|------|
| eval_loss NaN | Trainer 评估时 loss 为 NaN | 不影响推理，仅影响训练过程中的验证指标显示 |
| 仅训练 7/30 epoch | 训练被中断，未完成全部 epoch | 继续训练可能进一步提高准确率 |
| 模型过拟合倾向 | 错误步骤置信度也高达 0.88-0.90 | 建议增加训练数据或使用更强的正则化 |
| token 截断问题 | 验证集 token 数 4273-4526，需智能截断 kownledge_items | 已通过 build_text 中的二分搜索自动处理 |

---

## 附录：常见操作命令

```bash
# 查看 GPU 状态
nvidia-smi

# 查看已训练模型
ls -lh /root/autodl-tmp/checkpoint/

# 查看训练日志
python3 -c "import json; s=json.load(open('/root/autodl-tmp/checkpoint/checkpoint-1169/trainer_state.json')); print(f'epoch={s[\"epoch\"]}, steps={s[\"global_step\"]}')"

# 查看评估结果摘要
python3 -c "
import json
for n in ['validate','test']:
    d=json.load(open(f'/root/autodl-tmp/inference_{n}.json'))
    c=sum(1 for r in d for s in r['step_scores'] if s['true_label'] is not None and s['pred_label']==s['true_label'])
    t=sum(1 for r in d for s in r['step_scores'] if s['true_label'] is not None)
    print(f'{n}: {c}/{t}={c/t:.4f}')
"

# 统计数据量
echo "Train: $(ls /root/workspace/qwen-dianjin/DianJin-PRM/src/data/train/*.jsonl | wc -l) files"
echo "Validate: $(ls /root/workspace/qwen-dianjin/DianJin-PRM/src/data/validate/*.jsonl | wc -l) files"
echo "Test: $(ls /root/workspace/qwen-dianjin/DianJin-PRM/src/data/test/*.jsonl | wc -l) files"

# 导出 loss 为 CSV
python3 -c "import json,csv; s=json.load(open('/root/autodl-tmp/checkpoint/checkpoint-1169/trainer_state.json')); l=[(e['step'],e['loss']) for e in s['log_history'] if 'loss' in e]; csv.writer(open('/root/autodl-tmp/loss.csv','w')).writerows([('step','loss')]+l); print(f'{len(l)} rows')"
```
