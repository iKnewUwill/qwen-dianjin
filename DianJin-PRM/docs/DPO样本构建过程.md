# DPO 样本构建过程说明

> 本文档梳理 DPO（Direct Preference Optimization）偏好样本的完整构建流程，重点说明 **步骤级评分** 与 **轨迹级评分** 两种评分机制在 PRM 模型与 DPO 数据构建中的实际使用情况。当前排序口径已更新为「步骤平均 + 加权融合」（结合步骤级与轨迹级评分，摒弃 235B 双重验证），详见 §7。

---

## 1. 总体流程

```
pre/ 原始样本
   │  (4000 个问题，每个问题 4 条候选回答)
   ▼
加载 PRM 过程奖励模型 (Qwen3-8B + LoRA)
   │
   ▼
PRM 对每条候选回答打分
   │
   ▼
同问题内按分数排序 → 最高分 = chosen，最低分 = rejected
   │
   ▼
按 7:2:1 拆分 → train / validate / test (JSONL)
   │
   ▼
TRL DPOTrainer 训练策略模型
```

核心构建脚本：`src/data_dpo/build_dpo_dataset.py`

---

## 2. 输入数据格式（pre/）

`src/data_dpo/pre/dpo_pre.jsonl`，每行一条样本，结构如下：

```json
{
  "question": "结合盾安环境2024年末应收账款周转天数...的结构性风险。",
  "knowledge_items": {
    "应收账款周转天数_当前值": 123.4,
    "存货周转天数_较上期变动率": 0.12,
    ...
  },
  "answer": [
    {
      "steps": {
        "步骤1": "第一步推理分析...",
        "步骤2": "第二步推理分析...",
        "步骤3": "第三步推理分析..."
      },
      "final_answer": "盾安环境2024年末应收账款周转天数增加..."
    },
    ...  // 共 4 条候选回答
  ]
}
```

- 每个问题样本携带 `4` 条候选回答（`answer` 数组）。
- 每条候选回答由 `steps`（步骤字典，键为「步骤1/步骤2/步骤3」）和 `final_answer`（最终结论）组成。
- `knowledge_items` 中的值为 `None` 的项会在构建时被过滤（避免污染 PRM 输入）。

---

## 3. PRM 模型与两种评分机制

### 3.1 模型架构

PRM（Process Reward Model，过程奖励模型）定义于 `src/model/fin_prm.py`（`Qwen3ForProcessRewardModel`）：

```
Qwen3-8B backbone (36 层)
        │
        ▼
评分头 (score head)：Linear(hidden) → ReLU → Linear(2)
        │
        ▼
2 分类 logits（logit_0 = 错误概率，logit_1 = 正确概率）
```

- 这是 **Token 分类** 式架构（非生成式），在 Qwen3 backbone 之上加 2 层 MLP，输出 0/1 二分类 logits。
- `num_labels = 2`，`alpha = 1.0`。
- 微调方式为 LoRA（rank=16, alpha=32），评分头（`score`）为全量训练（`modules_to_save=["score"]`）。

### 3.2 两种评分标记（⭐关键）

模型使用两个特殊 token 标记不同的评分位置：

| 特殊 token | 作用 | 对应评分类型 |
|:---:|------|------|
| `<extra_0>` | **步骤分隔符**，位于每个推理步骤之后 | **步骤级评分** (step-level) |
| `<extra_1>` | **结束标记**，位于最终结论之后 | **轨迹级评分** (trajectory-level) |

输入文本格式（由 `build_prm_text` / PRM 训练的 collator 共同约定）：

```
##Question
{question}

##Knowledge
{k1: v1}
{k2: v2}

##Thinking Trajectory
{步骤1}<extra_0>{步骤2}<extra_0>{步骤3}<extra_0>

##Final Answer
{final_answer}<extra_1>
```

### 3.3 两种评分在 PRM 训练中的使用

`src/prm_trainer/prm_train.py` 的 `DataCollatorForProcessReward` 构造训练标签：

- 在 **每个 `<extra_0>` 位置** 放置 `step_labels[i]`（该步骤正确性 0/1 标签）→ **步骤级监督**。
- 在 **末尾 `<extra_1>` 位置** 放置 `trajectory_label`（整条轨迹正确性 0/1 标签）→ **轨迹级监督**。
- 损失函数仅在 `<extra_0>` 与 `<extra_1>` 位置计算 CrossEntropyLoss，其余位置 `-100` 忽略。

因此 **PRM 模型同时具备步骤级与轨迹级两种评分能力**：

| 评分类型 | 打分位置 | 含义 |
|------|------|------|
| 步骤级评分 | 每个 `<extra_0>` 位置 | 判断「该推理步骤」是否正确 |
| 轨迹级评分 | 末尾 `<extra_1>` 位置 | 判断「整条推理 + 最终结论」是否正确 |

### 3.4 两种评分在 PRM 推理中的使用

`src/prm_trainer/inference.py` 的 `score_sample` 对 `<extra_0>` 与 `<extra_1>` **所有位置** 打分：

- `scores[:-1]` → 各步骤的分数，用于计算 **步骤级准确率**（与 `step_labels` 比对）。
- `scores[-1]` → 末尾 `<extra_1>` 的分数，用于与 `trajectory_label` 比对（轨迹级准确率）。

---

## 4. DPO 样本构建中的评分使用（⭐重点）

### 4.1 核心打分函数

`src/data_dpo/build_dpo_dataset.py` 中的 `score_answer`：

```python
def score_answer(model, tokenizer, text, sep0_id, sep1_id, temperature=TEMPERATURE, device='cuda'):
    enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    ...
    logits = outputs.logits.float()

    # 步骤级：每个 <extra_0> 位置
    step_positions = (input_ids[0] == sep0_id).nonzero(as_tuple=True)[0]
    step_probs = [
        torch.nn.functional.softmax(logits[0, p] / temperature, dim=-1)[1].item()
        for p in step_positions
    ]

    # 轨迹级：末尾 <extra_1> 位置
    final_positions = (input_ids[0] == sep1_id).nonzero(as_tuple=True)[0]
    pos = final_positions[-1].item()
    prob = torch.nn.functional.softmax(logits[0, pos] / temperature, dim=-1)

    return {
        'step_probs': step_probs,      # 步骤级正确概率（逐步）
        'traj_prob': prob[1].item(),   # 轨迹级正确概率
    }
```

### 4.2 结论：DPO 构建采用**步骤平均 + 加权融合**（步骤级 + 轨迹级）

| 评分类型 | 是否在 DPO 构建中使用 | 说明 |
|:---:|:---:|------|
| **步骤级评分**（`<extra_0>` 位置） | ✅ **使用** | 对每个 `<extra_0>` 位置做 softmax，取各步骤 `prob_1` 后平均 |
| **轨迹级评分**（`<extra_1>` 位置） | ✅ **使用** | 取末尾 `<extra_1>` 位置的 `prob_1` 作为整条候选的轨迹级分数 |

**关键要点：**

1. `score_answer` 同时读取 `<extra_0>`（`sep0_id`）与 `<extra_1>`（`sep1_id`）两类位置的分数。
2. 综合得分 = \( \alpha \times \text{mean}(step\_probs) + \beta \times traj\_prob \)（步骤平均 + 加权融合）。
3. 摒弃 235B 双重验证，排序信号完全由 PRM 步骤级 + 轨迹级双层评分提供，不依赖额外结果验证模型。

### 4.3 打分温度

```python
TEMPERATURE = 2.0
```

- softmax 前对 logits 除以温度 2.0，使概率分布更平滑。
- 目的：缓解旧版 PRM 的 logits 极端化（`prob_1` 恒为 1.0）的过拟合问题，使 chosen/rejected 之间的 margin 更合理。

---

## 5. chosen / rejected 选择逻辑

对每个问题的 4 条候选回答分别打分后：

```python
scored.sort(key=lambda x: x['combined'], reverse=True)

if len(scored) >= 2:
    margin = scored[0]['combined'] - scored[-1]['combined']
    dpo_pairs.append({
        'prompt':  dpo_prompt,
        'chosen':  scored[0]['dpo_text'],   # 综合得分最高者
        'rejected': scored[-1]['dpo_text'],  # 综合得分最低者
        'metadata': {
            'chosen_score':  scored[0]['combined'],
            'rejected_score': scored[-1]['combined'],
            'margin': round(margin, 6),
            'num_candidates': len(scored),
            'all_scores': [a['combined'] for a in scored],
            'chosen_step_probs': scored[0]['step_probs'],
            'rejected_step_probs': scored[-1]['step_probs'],
        }
    })
```

- **chosen** = 综合得分（步骤平均 + 加权融合）最高的候选回答。
- **rejected** = 综合得分最低的候选回答。
- `combined = α × mean(step_probs) + β × traj_prob`，α/β 为可调融合权重。
- `margin = chosen_score - rejected_score`，作为偏好信号强弱的度量（越大越可靠）。

---

## 6. 输出数据格式（TRL DPO 标准格式）

输出为 JSONL，每行一个偏好对，字段与 `dpo-training-guide.md` 一致：

| 字段 | 类型 | 说明 |
|------|------|------|
| prompt | string | 问题 + 知识 + 思考指令（rollout_prompt 模板格式化） |
| chosen | string | 综合得分（步骤平均 + 加权融合）最高回答 |
| rejected | string | 综合得分最低回答 |
| metadata | object | `chosen_score` / `rejected_score` / `margin` / `num_candidates` / `all_scores` / 步骤级分数 |

### 6.1 文本组装格式

- **prompt**（`build_dpo_prompt`）：`rollout_prompt.txt` 模板 + 问题 + `参考知识：` 知识项列表。
- **chosen / rejected**（`build_dpo_answer`）：

```
<|begin_of_thought|>
{步骤1}
{步骤2}
{步骤3}
<|end_of_thought|>
<|begin_of_solution|>
{final_answer}
<|end_of_solution|>
```

### 6.2 数据拆分

按 `random.seed(42)` 打乱后按 **7:2:1** 拆分：

| 拆分 | 目录 | 文件 |
|:---:|------|------|
| train | `src/data_dpo/train/` | `dpo_train.jsonl` |
| validate | `src/data_dpo/validate/` | `dpo_val.jsonl` |
| test | `src/data_dpo/test/` | `dpo_test.jsonl` |

---

## 7. 更新说明（排序口径变更）

依据更新后的 `docs/实验方案.md` 阶段二，DPO 偏好排序口径调整为 **「步骤平均 + 加权融合」**，结合 PRM 步骤级评分与轨迹级评分，摒弃 235B 双重验证：

| 维度 | 旧口径（纯轨迹级） | 新口径（步骤平均 + 加权融合） |
|------|------|------|
| 步骤级评分 | ❌ 未使用 | ✅ 逐 `<extra_0>` 位置打分，取平均 |
| 轨迹级评分 | ✅ 直接用 `<extra_1>` 位置 `prob_1` | ✅ 作为加权融合的一项 |
| 235B 双重验证 | ❌ 未启用 | ❌ 摒弃 |
| 综合评分 | 单点轨迹级 `prob_1` | `α × mean(step_probs) + β × traj_prob` |

> 当前 `build_dpo_dataset.py` 仍为「纯轨迹级」实现，需按 §4 所示改造 `score_answer`（同时读取 `<extra_0>` 与 `<extra_1>` 分数）并调整排序逻辑后，重新构建 DPO 数据集。

---

## 8. 相关脚本清单

| 脚本 | 作用 | 与评分的关联 |
|------|------|------|
| `src/data_dpo/build_dpo_dataset.py` | **DPO 样本构建主脚本** | 步骤平均 + 加权融合（`<extra_0>` + `<extra_1>`） |
| `src/model/fin_prm.py` | PRM 模型定义 | 定义评分头，`make_step_rewards` 可提取逐 token 分数 |
| `src/prm_trainer/prm_train.py` | PRM 训练 | 同时训练步骤级（`<extra_0>`）与轨迹级（`<extra_1>`）标签 |
| `src/prm_trainer/inference.py` | PRM 推理/评估 | 提取所有 `<extra_0>` + `<extra_1>` 位置的分数 |
| `src/data_dpo/analyze_dpo_margins.py` | margin 分布分析 | 分析构建后数据的 margin/阈值 |
| `src/data_dpo/eval_dpo.py` | DPO 训练后偏好评估 | 基于 DPO 隐式奖励计算偏好准确率 |
| `src/data_dpo/generate_eval.py` | DPO vs Base 评估生成 | 用基座/DPO 模型生成「分析过程+结论」 |
| `src/data_dpo/verify_generation.py` | 校验评估生成完整性 | — |
| `src/dpo_trainer/prepare_dpo_data.py` | 旧版 DPO 数据准备（基于 CFLUE + ground truth） | 规则构造正确/错误推理，非 PRM 打分 |
| `src/dpo_trainer/dpo_train.py` | DPO 训练主程序（TRL DPOTrainer） | 消费构建出的偏好对 |

---

## 9. 一句话总结

> DPO 样本构建使用 **PRM 模型的步骤级评分（`<extra_0>`）与轨迹级评分（`<extra_1>`）**，通过「步骤平均 + 加权融合」（`α × mean(step_probs) + β × traj_prob`）对同一问题下的多条候选回答排序，最高分作 chosen、最低分作 rejected；摒弃 235B 双重验证，排序信号完全由 PRM 步骤级 + 轨迹级双层评分提供。
