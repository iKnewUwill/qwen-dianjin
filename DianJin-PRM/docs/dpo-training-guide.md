# DPO 策略模型训练与操作手册

## 目录

1. [环境概览](#1-环境概览)
2. [模型架构](#2-模型架构)
3. [模型存放位置](#3-模型存放位置)
4. [日志存放位置](#4-日志存放位置)
5. [模型输出结果保存位置](#5-模型输出结果保存位置)
6. [模型训练方法](#6-模型训练方法)
7. [模型操作方法](#7-模型操作方法)

---

## 1. 环境概览

### 硬件

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 GB VRAM |
| CPU | Intel Xeon |
| 数据盘 | /root/autodl-tmp |

### 软件环境 (DPO 专用)

| 项目 | 详情 |
|------|------|
| Conda 环境 | `/root/autodl-tmp/miniconda3/envs/dianjin-dpo` |
| Python | 3.12.0 |
| PyTorch | 2.8.0+cu128 |
| transformers | 4.56.1 |
| trl | 0.27.0 |
| peft | 0.17.1 |
| bitsandbytes | 0.49.2 |
| deepspeed | 0.17.6 |
| datasets | 4.0.0 |

### 目录结构

```
/root/workspace/qwen-dianjin/DianJin-PRM/
├── src/
│   ├── dpo_trainer/              # DPO 训练核心代码
│   │   ├── dpo_train.py          # 训练主程序（TRL DPOTrainer）
│   │   ├── prepare_dpo_data.py   # 原始 DPO 数据准备（基于 CFLUE）
│   │   ├── ds_config.json        # DeepSpeed ZeRO-2 配置
│   │   ├── run_dpo.sh            # 原始训练启动脚本
│   │   └── data/                 # 原始 DPO 数据输出目录
│   │
│   ├── data_dpo/                 # DPO 数据仓库（当前任务）
│   │   ├── pre/                  # 原始样本（400 个 JSONL 文件）
│   │   ├── train/                # 训练集（dpo_train.jsonl）
│   │   ├── validate/             # 验证集（dpo_val.jsonl）
│   │   ├── test/                 # 测试集（dpo_test.jsonl）
│   │   ├── logs/                 # 训练日志输出
│   │   ├── build_dpo_dataset.py  # PRM 打分 → DPO 数据构建脚本
│   │   └── run_dpo_train.sh      # DPO 训练启动脚本
│   │
│   ├── prm_trainer/              # PRM 过程奖励模型（上游模型）
│   │   ├── prm_train.py          # PRM 训练主程序
│   │   └── inference.py          # PRM 推理/打分脚本
│   │
│   ├── model/
│   │   ├── fin_prm.py            # PRM 模型定义
│   │   ├── fin_config.py         # PRM 模型配置
│   │   └── config.json           # 模型超参数配置
│   │
│   └── templates/
│       └── rollout_prompt.txt    # 滚动采样 prompt 模板
│
└── docs/
    ├── prm-training-guide.md     # PRM 训练操作手册
    └── dpo-training-guide.md     # 本文档
```

### 数据格式

DPO 偏好数据为 JSONL 格式，每行一条样本：

| 字段 | 类型 | 说明 |
|------|------|------|
| prompt | string | 问题 + 金融知识 + 思考指令（rollout_prompt 模板格式化） |
| chosen | string | PRM 最高分回答，含 `<\|begin_of_thought\|>` / `<\|end_of_solution\|>` 标签 |
| rejected | string | PRM 最低分回答，格式同上 |
| metadata | object | 打分元信息（chosen_score, rejected_score, all_scores） |

---

## 2. 模型架构

```
┌─────────────────────────────────────────────────────────┐
│                  DPO 训练流程                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: PRM 打分                                       │
│  ┌──────────────┐     ┌────────────────┐                │
│  │ pre/ 原始样本  │ ──▶ │ PRM 过程奖励模型 │                │
│  │ (3候选人/问题) │     │ (Qwen3-8B+LoRA) │                │
│  └──────────────┘     └───────┬────────┘                │
│                               │ 打分                    │
│                               ▼                         │
│                  ┌────────────────────┐                  │
│                  │ 选出 chosen/rejected│                  │
│                  └────────┬───────────┘                  │
│                           │                             │
│  Step 2: DPO 训练         ▼                             │
│  ┌──────────────┐     ┌────────────────┐                │
│  │ DPO JSONL     │ ──▶ │ TRL DPOTrainer │                │
│  │ (7:2:1 拆分)  │     │                │                │
│  └──────────────┘     │ Policy: Qwen3-8B│               │
│                       │ + LoRA          │               │
│                       └───────┬────────┘                │
│                               │                         │
│                               ▼                         │
│                  ┌────────────────────┐                  │
│                  │ DPO LoRA 适配器权重 │                  │
│                  └────────────────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

Policy Model (策略模型):
  Qwen3-8B backbone
    ↓ + LoRA (rank=16, alpha=32)
  8.19B 参数量，仅训练 0.53% (43.6M)
  target_modules: q_proj, k_proj, v_proj, o_proj,
                  gate_proj, up_proj, down_proj

Reference Model (参考模型):
  冻结的初始 Qwen3-8B 权重（训练中不更新，由 TRL 内部处理）

DPO Loss:
  sigmoid loss, beta=0.1
  optim: 8-bit paged AdamW, lr=5e-6, cosine scheduler
  seq_length: 3072, effective batch: 16
```

---

## 3. 模型存放位置

### 基座模型

| 模型 | 路径 | 说明 |
|------|------|------|
| Qwen3-8B 基座 | `/root/autodl-tmp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218` | DPO Policy/Reference 模型共享同一基座 |
| HuggingFace 缓存 | `/root/autodl-tmp/huggingface/` | HuggingFace 通用缓存目录 |

### PRM 模型（上游打分用）

| 项目 | 路径 |
|------|------|
| PRM 配置 | `/root/workspace/qwen-dianjin/DianJin-PRM/src/model/config.json` |
| PRM LoRA 权重 | `/root/autodl-tmp/checkpoint/checkpoint-1169/` |
| PRM 模型定义 | `src/model/fin_prm.py` → `Qwen3ForProcessRewardModel` |
| PRM 模型配置 | `src/model/fin_config.py` → `Qwen3PRMConfig` |

### DPO 训练输出（checkpoint）

| 项目 | 路径 |
|------|------|
| 默认输出根目录 | `/root/autodl-tmp/dpo_checkpoint/` |
| 单次实验目录 | `/root/autodl-tmp/dpo_checkpoint/{experiment_tag}/` |

单次实验目录内容：

```
dpo_checkpoint/dpo_20260607_205216/
├── adapter_config.json        # LoRA 适配器配置
├── adapter_model.safetensors  # LoRA 权重文件（最终模型）
├── tokenizer.json             # Tokenizer 文件
├── tokenizer_config.json      # Tokenizer 配置
├── special_tokens_map.json    # 特殊 token 映射
├── added_tokens.json          # 新增 token 列表
├── chat_template.jinja        # 对话模板
├── training_config.json       # 训练超参数快照
├── training_args.bin          # TrainingArguments 序列化
├── train_metrics.json         # 训练指标汇总
├── training.log              # 训练日志（Python logging）
├── runs/                      # TensorBoard 事件文件
├── checkpoint-3/              # 中间 checkpoint
├── README.md                  # PEFT 自动生成
└── merges.txt                 # BPE 合并规则
```

---

## 4. 日志存放位置

### 训练日志

| 日志类型 | 路径 | 说明 |
|----------|------|------|
| Shell 输出日志 | `src/data_dpo/logs/{tag}_output.log` | bash 脚本全部输出（含进度条、metrics） |
| Python 内部日志 | `dpo_checkpoint/{tag}/training.log` | Python logging 输出（INFO 级别） |
| TensorBoard 事件 | `dpo_checkpoint/{tag}/runs/` | 可视化训练曲线 |

### 日志内容示例

Shell 输出日志 (`{tag}_output.log`) 包含：
- 模型加载进度
- 实验方案校验结果
- 训练步骤级 metrics（loss, rewards, gradients）
- 训练完成后的汇总指标

Python 内部日志 (`training.log`) 包含：
- 时间戳格式的 INFO/WARNING 级别日志
- 参数校验详情
- 训练异常信息

### 查看日志

```bash
# 查看最新训练完整日志
cat src/data_dpo/logs/dpo_$(date +%Y%m%d)*_output.log

# 实时监控训练
tail -f src/data_dpo/logs/dpo_*_output.log

# 查看 Python 内部日志
cat /root/autodl-tmp/dpo_checkpoint/{experiment_tag}/training.log

# 启动 TensorBoard
tensorboard --logdir /root/autodl-tmp/dpo_checkpoint/{experiment_tag}/runs/
```

---

## 5. 模型输出结果保存位置

### 输出结果汇总

| 输出类型 | 路径 | 格式 |
|----------|------|------|
| DPO 训练数据 | `src/data_dpo/{train,validate,test}/dpo_*.jsonl` | JSONL |
| LoRA 适配器权重 | `dpo_checkpoint/{tag}/adapter_model.safetensors` | safetensors |
| 训练超参数 | `dpo_checkpoint/{tag}/training_config.json` | JSON |
| 训练指标 | `dpo_checkpoint/{tag}/train_metrics.json` | JSON |
| 中间 checkpoint | `dpo_checkpoint/{tag}/checkpoint-N/` | 每 save_steps 步保存 |
| PRM 打分结果 | 嵌入在 DPO 数据文件的 `metadata` 字段中 | JSON |

### 训练指标说明 (`train_metrics.json`)

```json
{
  "train_runtime": 75.02,          // 总训练时间（秒）
  "train_samples_per_second": 0.56, // 每秒处理样本数
  "train_steps_per_second": 0.04,   // 每秒训练步数
  "train_loss": 0.6825,            // 最终训练 loss
  "epoch": 1.0                      // 完成的 epoch 数
}
```

### 步骤级指标（训练过程中输出）

| 指标 | 说明 |
|------|------|
| loss | DPO loss 值（越低越好） |
| rewards/chosen | chosen 回答的奖励均值 |
| rewards/rejected | rejected 回答的奖励均值 |
| rewards/margins | chosen - rejected 奖励差（应为正） |
| rewards/accuracies | 奖励准确率（chosen > rejected 的比例） |
| logps/chosen | chosen 回答的对数概率 |
| logps/rejected | rejected 回答的对数概率 |

---

## 6. 模型训练方法

### 6.1 完整训练流程

```
原始样本 (pre/) → PRM 打分 → 构建 DPO 数据 → DPO 训练 → LoRA 模型
```

### 6.2 Step 1: 构建 DPO 数据集

使用 PRM 模型对 `pre/` 目录下的样本打分，选出偏好对：

```bash
# 激活 DPO 环境
export PATH="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin:$PATH"

# 运行数据构建脚本
cd /root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo
python build_dpo_dataset.py
```

**脚本参数**（在脚本内修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| NUM_FILES | 20 | 从 pre/ 读取的文件数 |
| SEED | 42 | 随机种子（保证可复现） |
| MAX_SEQ_LENGTH | 4096 | PRM 输入最大长度 |
| train:val:test | 7:2:1 | 数据拆分比例 |

**输出**:

- `train/dpo_train.jsonl` — 训练集
- `validate/dpo_val.jsonl` — 验证集
- `test/dpo_test.jsonl` — 测试集

### 6.3 Step 2: 启动 DPO 训练

```bash
# 方式一：使用封装脚本（推荐）
cd /root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo
bash run_dpo_train.sh [experiment_tag] [max_train_samples]

# 示例
bash run_dpo_train.sh dpo_exp_001          # 默认标签
bash run_dpo_train.sh dpo_exp_002 32       # 限制训练样本数

# 方式二：直接调用 Python
python /root/workspace/qwen-dianjin/DianJin-PRM/src/dpo_trainer/dpo_train.py \
    --train_data src/data_dpo/train/dpo_train.jsonl \
    --eval_data src/data_dpo/validate/dpo_val.jsonl \
    --output_dir /root/autodl-tmp/dpo_checkpoint \
    --max_length 3072 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_batch_size 1 \
    --grad_accum 16 \
    --num_epochs 1 \
    --warmup_steps 50 \
    --bf16 \
    --optim paged_adamw_8bit \
    --loss_type sigmoid \
    --experiment_tag dpo_test
```

### 6.4 训练参数完整列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --train_data | (必填) | DPO 训练数据 JSONL 路径 |
| --eval_data | None | DPO 验证数据 JSONL 路径 |
| --output_dir | /root/autodl-tmp/dpo_checkpoint | 模型输出根目录 |
| --max_length | 3072 | 最大序列长度 |
| --learning_rate | 5e-6 | 学习率 |
| --beta | 0.1 | DPO beta 参数（控制偏离参考模型的程度） |
| --lora_r | 16 | LoRA rank |
| --lora_alpha | 32 | LoRA alpha（缩放因子） |
| --per_device_batch_size | 1 | 每 GPU 批次大小 |
| --grad_accum | 16 | 梯度累积步数（effective batch = 1×16 = 16） |
| --num_epochs | 1 | 训练轮数 |
| --max_steps | -1 | 最大训练步数（覆盖 epoch） |
| --warmup_steps | 50 | 学习率预热步数 |
| --loss_type | sigmoid | DPO loss 类型：sigmoid / hinge / ipo / kto_pair |
| --bf16 | True | 使用 bf16 混合精度 |
| --optim | paged_adamw_8bit | 优化器类型 |
| --deepspeed | None | DeepSpeed 配置文件路径（需要 MPI 环境） |
| --experiment_tag | auto | 实验标签（用于输出目录命名） |
| --skip_plan_check | False | 跳过实验方案校验 |

### 6.5 实验方案校验

训练启动时自动校验参数是否符合 `实验方案.md` 阶段三的设计：

```
实验方案校验:
  [✓] 使用 DPO (非 GRPO)
  [✓] 学习率 5e-06 == 5e-6
  [✓] beta 0.1 == 0.1
  [✓] LoRA rank 16 == 16
  [✓] LoRA alpha 32 == 32
  [✓] bf16 = True
  [✓] max_seq_length 3072 == 3072
  [✓] optimizer paged_adamw_8bit == paged_adamw_8bit
```

---

## 7. 模型操作方法

### 7.1 加载训练好的 DPO 模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 基座模型路径
base_path = "/root/autodl-tmp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
# DPO LoRA 权重路径
lora_path = "/root/autodl-tmp/dpo_checkpoint/dpo_20260607_205216"

tokenizer = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(model, lora_path)

# 推理
inputs = tokenizer("你的问题", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 7.2 合并 LoRA 权重到基座模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_path = "..."
lora_path = "/root/autodl-tmp/dpo_checkpoint/dpo_20260607_205216"
save_path = "/root/autodl-tmp/dpo_merged"

model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto")
model = PeftModel.from_pretrained(model, lora_path)
merged = model.merge_and_unload()
merged.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(save_path)
```

### 7.3 使用 DPO 模型进行推理

DPO 模型在训练过程中使用了 rollout_prompt 模板，推理时需要保持一致：

```python
# 使用 rollout_prompt.txt 模板
with open("/root/workspace/qwen-dianjin/DianJin-PRM/src/templates/rollout_prompt.txt") as f:
    template = f.read()

prompt = template.format(question="你的财务分析问题")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)
```

### 7.4 重新构建 DPO 数据集

如果 pre/ 目录下的文件有更新，或想使用不同数量的文件：

```bash
# 1. 修改 build_dpo_dataset.py 中的 NUM_FILES 参数
#    或在命令行修改：
sed -i 's/NUM_FILES = 20/NUM_FILES = 50/' src/data_dpo/build_dpo_dataset.py

# 2. 清空旧数据
rm -f src/data_dpo/train/*.jsonl
rm -f src/data_dpo/validate/*.jsonl
rm -f src/data_dpo/test/*.jsonl

# 3. 重新构建
cd /root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo
python build_dpo_dataset.py

# 4. 检查结果
wc -l train/*.jsonl validate/*.jsonl test/*.jsonl
```

### 7.5 环境切换

```bash
# DPO 训练环境
export PATH="/root/autodl-tmp/miniconda3/envs/dianjin-dpo/bin:$PATH"

# PRM 训练环境
export PATH="/root/autodl-tmp/miniconda3/envs/dianjin-prm/bin:$PATH"
```
