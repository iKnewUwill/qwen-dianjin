# 实验记录

## 实验概览
| 项目 | 内容 |
|------|------|
| 目标 | 基于LLM微调的银行金融财务推理优化 |
| 框架 | PRM (Process Reward Model) + GRPO (Group Relative Policy Optimization) |
| 基座模型 | Qwen3-8B |
| 实验日期 | 2026-05-31 ~ 2026-06-01 |

---

## 阶段一：PRM 训练

### 代码变更

#### 1. Qwen2.5-7B → Qwen3-8B 迁移
| 文件 | 变更内容 |
|------|----------|
| `src/model/fin_config.py` | `Qwen2Config` → `Qwen3Config`, `Qwen2PRMConfig` → `Qwen3PRMConfig` |
| `src/model/fin_prm.py` | `Qwen2Model` → `Qwen3Model`, `Qwen2PreTrainedModel` → `Qwen3PreTrainedModel` |
| `src/model/config.json` | 更新为 Qwen3-8B 配置: hidden_size=4096, 36 layers, 32 heads, head_dim=128, vocab_size=151936 |
| `src/prm_trainer/prm_train.py` | 更新模型路径为 Qwen3-8B |

#### 2. SFTTrainer → Trainer 迁移
- **问题**: `trl.SFTTrainer` 专为因果语言模型设计，不兼容 Token 分类任务的 `TokenClassifierOutput`，导致 `loss=0`, `grad_norm=0`。
- **解决**: 改用 `transformers.Trainer`，它直接使用模型的 `forward()` 返回的 loss。
- **变更文件**: `src/prm_trainer/prm_train.py`

#### 3. gradient_checkpointing 禁用
- **问题**: `gradient_checkpointing=True` 与 PEFT 的 `modules_to_save=["score"]` 有兼容性问题，导致梯度为 0。
- **解决**: 关闭 gradient_checkpointing。98GB VRAM 足够支持 8B 模型 + LoRA 全参数训练。
- **变更文件**: `src/prm_trainer/prm_train.py`

#### 4. deepspeed launcher 弃用
- **原因**: 单卡训练不需要 deepspeed 分布式启动器，直接使用 `python3`。
- **变更文件**: `src/prm_trainer/train.sh`

### 训练结果
| 参数 | 值 |
|------|------|
| 训练样本 | 35 条 |
| 验证样本 | 10 条 |
| 测试样本 | 5 条 |
| 训练轮数 | 30 epochs |
| 可训练参数 | 60,436,482 / 7,644,546,052 (0.79%) |
| 训练总耗时 | 741 秒 (12分21秒) |
| 训练集最终 loss | ≈0 (过拟合) |
| 验证集 loss | 1.083 (从 0.648 上升，过拟合) |
| 测试集 loss | 0.682 |

### 问题与解决

1. **SFTTrainer 不兼容 Token 分类**
   - 症状: 训练时 loss=0, grad_norm=0
   - 根因: SFTTrainer 专为因果LM设计，不支持 TokenClassifierOutput
   - 解决: 改用 transformers.Trainer

2. **gradient_checkpointing 与 PEFT modules_to_save 不兼容**
   - 症状: 训练时 loss=0, grad_norm=0
   - 根因: 未知，可能是 gradient_checkpointing 与自定义 module 的交互问题
   - 解决: 禁用 gradient_checkpointing

3. **数据量不足导致过拟合**
   - 症状: 训练 loss 降为 0，验证 loss 上升
   - 分析: 35 条样本对 8B 模型来说太少，即使 LoRA 也会快速过拟合
   - 建议: 增加训练数据量至 1000+ 条

---

## 阶段二：GRPO 环境准备

### 环境信息
| 参数 | 值 |
|------|------|
| 环境名 | dianjin-grpo |
| 位置 | `/root/autodl-tmp/miniconda3/envs/dianjin-grpo` |
| Python | 3.12.0 |
| PyTorch | 2.8.0+cu128 |
| vllm | 0.8.5 |
| ray | 2.47.1 |
| verl | 0.4.0.dev0 (本地) |
| disk 占用 | ~3.5GB (在数据盘) |

### 数据预处理
- CFLUE 金融问答数据集: 30,907 条
- 分割: 29,907 训练 + 1,000 验证
- 格式: parquet (verl 标准格式)
- 输出: `src/output/train.parquet`, `src/output/test.parquet`

### 已知问题

1. **CUDA 驱动库损坏**
   - 现象: `libcuda.so.595.58.03` 为 0 字节空文件
   - 根因: pip install vllm 过程中覆盖了 nvidia-container-runtime 挂载的驱动库
   - 恢复方法: 重启容器 或 重新安装 NVIDIA 驱动
   - 临时方案: `export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH`（但 compat 库版本 570 与驱动 595 不完全兼容）

2. **torch 2.8.0+cu128 依赖冲突**
   - 问题: torch 需要 nvidia 包版本与 pip 预装的冲突
   - 解决: `pip install --no-deps` 安装 torch，然后手动修复依赖版本

### GRPO 启动命令
```bash
# 激活环境
conda activate /root/autodl-tmp/miniconda3/envs/dianjin-grpo

# 启动 GRPO 训练（单卡版本）
cd /root/workspace/qwen-dianjin/DianJin-PRM/src/grpo_trainer
bash fin_prm_grpo_single_gpu.sh
```

---

## 待办事项（需连接 GPU 后执行）

- [ ] 修复 CUDA 驱动库（重启容器）
- [ ] 验证 GRPO 环境 CUDA 可用性
- [ ] 运行 GRPO 训练
- [ ] 测试 PRM 模型作为奖励模型集成到 GRPO
- [ ] 消融实验（基线组、实验组 A/B/C、完整方案组）
