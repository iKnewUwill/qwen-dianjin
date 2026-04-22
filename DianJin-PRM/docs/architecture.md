# DianJin-PRM 架构设计文档

## 项目概述

DianJin-PRM (DianJin-Fin-PRM) 是一个面向金融领域的 Process Reward Model (PRM)，用于评估金融推理任务中中间推理步骤的质量。该模型在 Qwen2.5-7B-Instruct 基础上进行 Token 分类改造，通过 LoRA 微调方式在金融推理数据上训练，支持步骤级和轨迹级的奖励评估。

## 目录结构

```
DianJin-PRM/
├── docs/                              # 项目文档
│   ├── run-report.md                  # 运行报告
│   └── architecture.md                # 架构设计文档
├── src/
│   ├── data/                          # 数据目录
│   │   ├── train/                     # 训练集 (7 JSONL files, 35 samples)
│   │   ├── validate/                  # 验证集 (2 JSONL files, 10 samples)
│   │   └── test/                      # 测试集 (1 JSONL file, 5 samples)
│   ├── model/                         # 模型定义
│   │   ├── config.json                # PRM 模型配置 (基于 Qwen2.5-7B + num_labels=2)
│   │   ├── fin_config.py              # Qwen2PRMConfig (继承 Qwen2Config)
│   │   └── fin_prm.py                 # Qwen2ForProcessRewardModel (Token分类架构)
│   ├── prm_trainer/                   # PRM 训练器
│   │   ├── prm_train.py               # 主训练脚本 (LoRA + SFTTrainer)
│   │   ├── train.sh                   # 训练启动脚本
│   │   ├── deepspeed_config.json      # DeepSpeed ZeRO-2 配置
│   │   └── requirements.txt           # Python 依赖
│   ├── grpo_trainer/                  # GRPO 下游应用训练器
│   │   ├── data_preprocess.py         # GRPO 数据预处理
│   │   └── fin_prm_grpo.sh            # GRPO 训练启动脚本
│   ├── verl/                          # VERL 框架 (RL训练基础设施)
│   │   ├── models/                    # 模型实现 (Qwen2, LLaMA, Megatron等)
│   │   ├── trainer/                   # 训练器 (PPO, SFT, Eval)
│   │   ├── workers/                   # 分布式工作器 (Actor, Critic, Reward)
│   │   └── utils/                     # 工具集 (数据集, 检查点, 指标等)
│   └── scripts/                       # 安装脚本
├── images/                            # 文档图片
├── README.md                          # 英文文档
└── README_zh.md                       # 中文文档
```

## 核心模块说明

### 1. 模型层 (src/model/)

**Qwen2PRMConfig** (fin_config.py): 继承 `transformers.Qwen2Config`，新增 `alpha` 和 `num_labels` 两个自定义参数。`alpha` 用于奖励缩放系数，`num_labels=2` 表示二分类（正确/错误步骤）。通过继承 `Qwen2Config`，自动获得 `pad_token_id`、`layer_types` 等 Qwen2 模型所需的全部配置项。

**Qwen2ForProcessRewardModel** (fin_prm.py): 基于 `Qwen2PreTrainedModel` 的 Token 分类模型。核心结构为：

- `self.model`: Qwen2Model 基座，输出 hidden_states
- `self.score`: 两层全连接头 (Linear(hidden_size, hidden_size) → ReLU → Linear(hidden_size, 2))，对每个 token 位置输出 2 类 logits
- `forward()`: 将 Qwen2Model 输出的 hidden_states 通过 score head 得到 logits，使用 `CrossEntropyLoss` 计算 loss
- `make_step_rewards()`: 推理时从 logits 提取各步骤的正类概率作为奖励分数

标签设计：`-100` 表示忽略位置，`<extra_0>` token 位置设置为 `step_labels`，`<extra_1>` token 位置设置为 `trajectory_label`。

### 2. 训练层 (src/prm_trainer/)

**DataCollatorForProcessReward**: 自定义数据整理器，将 JSONL 中的字段拼接为模型输入：

```
##Question\n{question}\n\n##Knowledge\n{knowledge_text}\n\n##Thinking Trajectory\n{step1}<extra_0>{step2}<extra_0>{step3}<extra_0>\n\n##Final Answer\n{final_answer}<extra_1>
```

`<extra_0>` 作为步骤分隔符，`<extra_1>` 作为序列结束标记。标签张量仅在分隔符 token 位置设置 step_labels，最后一个位置设置 trajectory_label，其余位置为 -100。

**load_jsonl_dataset**: 从目录自动扫描所有 `.jsonl` 文件，逐个加载后通过 `concatenate_datasets` 合并为统一数据集。

**训练流程**:

1. 加载 Qwen2PRMConfig 和创建 Qwen2ForProcessRewardModel
2. 从 Qwen2.5-7B-Instruct 加载预训练权重到 `model.model`
3. 添加 `<extra_0>`、`<extra_1>` 特殊 token，调整 embedding 大小
4. 应用 LoRA 包装（7 个线性层 + score head 全参数训练）
5. SFTTrainer 训练，每 epoch 进行验证集评估
6. 训练结束后在测试集上评估

### 3. 数据格式

每条 JSONL 样本包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| question | string | 金融领域问题文本 |
| knowledge_items | dict | 结构化财务指标字典，键为指标名称，值为数值 |
| steps | dict | 推理步骤 {"Step 1": "...", "Step 2": "...", "Step 3": "..."} |
| step_labels | list[int] | 步骤标签 [1, 1, 1]，1=正确，0=错误 |
| final_answer | string | 最终答案文本 |
| trajectory_label | int | 轨迹标签，1=正确 |

### 4. LoRA 配置

| 参数 | 值 |
|------|-----|
| rank | 16 |
| alpha | 32 |
| dropout | 0.05 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| bias | none |
| task_type | TOKEN_CLS |
| modules_to_save | ["score"] (全参数训练) |
| 可训练参数占比 | 0.75% (53M / 7.1B) |

### 5. 下游应用 (src/grpo_trainer/)

GRPO (Group Relative Policy Optimization) 使用 Fin-PRM 作为奖励模型进行强化学习训练。`data_preprocess.py` 负责将原始数据转换为 GRPO 训练所需的 parquet 格式，`fin_prm_grpo.sh` 启动基于 Qwen2.5-7B-Instruct 策略模型的 GRPO 训练。

### 6. VERL 框架 (src/verl/)

基于 VERL 的分布式 RL 训练框架，提供以下能力：

- 多种模型并行策略: FSDP, Megatron, vLLM, SGLang
- PPO 训练流水线: Actor-Critic 架构的完整实现
- 分布式工作器: Actor (策略模型), Critic (价值模型), Reward Model (奖励模型)
- 工具集: 数据集处理、检查点管理、训练指标收集、内存优化等
