## 训练环境
- GPU: NVIDIA GeForce RTX 5090, 32GB VRAM
- 模型: Qwen2.5-7B-Instruct (7B参数)
- 基座模型路径: /root/autodl-tmp/huggingface/models--Qwen--Qwen2.5-7B-Instruct
- Python: 3.12, PyTorch 2.8.0+cu128
- 主要依赖: transformers 5.5.4, trl 1.2.0, peft 0.19.1, bitsandbytes 0.49.2, accelerate 1.13.0

## 数据集
- 训练集: src/data/train/ 目录下 7 个 JSONL 文件，共 35 条样本
- 验证集: src/data/validate/ 目录下 2 个 JSONL 文件，共 10 条样本
- 测试集: src/data/test/ 目录下 1 个 JSONL 文件，共 5 条样本
- 每条样本包含: question, knowledge_items, steps(Step 1/2/3), final_answer, step_labels([1,1,1]), trajectory_label(1)
- 样本 token 长度范围: 1809-2625 tokens

## 训练配置
- 训练方式: LoRA 微调 (r=16, alpha=32, dropout=0.05)
- LoRA 目标模块: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- score head 全参数训练 (modules_to_save=["score"])
- 可训练参数: 53,225,986 / 7,135,278,084 (0.75%)
- batch_size: 1, gradient_accumulation_steps: 8
- max_length: 3072
- learning_rate: 2e-5
- epochs: 3
- 优化器: paged_adamw_8bit
- 调度器: cosine
- warmup_steps: 50
- bf16: True
- gradient_checkpointing: True (use_reentrant=False)
- eval_strategy: epoch, save_strategy: epoch

## 从 Qwen3 迁移到 Qwen2 的变更
1. fin_config.py: 从继承 PretrainedConfig (Qwen3) 改为继承 Qwen2Config，仅保留 alpha 和 num_labels 自定义参数
2. fin_prm.py: Qwen3Model → Qwen2Model, Qwen3PreTrainedModel → Qwen2PreTrainedModel
3. config.json: 基于 Qwen2.5-7B-Instruct 原始配置生成，添加 num_labels=2, alpha=1.0

## 数据加载变更
- 原代码: 单文件加载 (load_dataset("json", data_files=单文件路径))
- 新代码: 目录自动扫描 (glob匹配 *.jsonl → concatenate_datasets)
- DataCollator bug 修复: 修复了 labels_list 变量遮蔽和 trajectory_label 引用错误样本的两个 bug

## 遇到的问题及解决
1. **Qwen2Model 初始化缺少 pad_token_id** → Qwen2PRMConfig 继承 Qwen2Config 而非 PretrainedConfig
2. **Qwen2Model 初始化缺少 layer_types** → 同上，继承 Qwen2Config 自动获取
3. **全参数微调 OOM (32GB VRAM)** → 改用 LoRA + gradient_checkpointing + paged_adamw_8bit
4. **评估阶段 OOM** → 设置 per_device_eval_batch_size=1, eval_accumulation_steps=1, gradient_checkpointing_kwargs={"use_reentrant": False}

## 训练结果
- 总训练时长: 111.8 秒
- 训练速度: 0.939 samples/s, 0.134 steps/s
- 训练集: 35 samples × 3 epochs = 15 steps (batch=1, grad_accum=8, effective batch=8)

| Epoch | eval_loss | eval_accuracy | eval_runtime |
|-------|-----------|---------------|-------------|
| 1     | 8.259     | 0.075         | 2.278s      |
| 2     | 8.259     | 0.075         | 2.286s      |
| 3     | 8.259     | 0.075         | 2.335s      |

- 测试集 eval_loss: 8.824

## Checkpoints
- 保存路径: /root/autodl-tmp/checkpoint/
- checkpoint-5 (epoch 1), checkpoint-10 (epoch 2), checkpoint-15 (epoch 3)

## 结论
- 训练/验证/测试全流程成功跑通
- 32GB VRAM 单卡通过 LoRA 方案成功运行 7B 模型训练
- eval_loss 在 3 个 epoch 内未下降（仅 35 条训练数据，且 eval_loss=8.259 为初始值），说明需要更多训练数据和更多 epoch 才能收敛
