"""
DPO 策略模型训练脚本
====================
根据 实验方案.md 阶段三设计：
  - 使用 TRL DPOTrainer
  - Policy model: Qwen3-8B (base)
  - Reference model: frozen initial weights
  - LoRA (rank=16, alpha=32)
  - Learning rate: 5e-6, beta: 0.1
  - bf16, DeepSpeed ZeRO-2 + CPU Offload
  - Max seq length: 3072, effective batch size: 16
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/huggingface"
os.environ["TRITON_CACHE_DIR"] = "/root/autodl-tmp/triton"
os.makedirs("/root/autodl-tmp/triton", exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ============ 实验方案校验 ============

EXPERIMENTAL_PLAN_CHECKS = {
    "algorithm": "DPO (Direct Preference Optimization)",
    "training_framework": "TRL (Transformer Reinforcement Learning)",
    "policy_model": "Qwen3-8B",
    "reference_model": "frozen Qwen3-8B initial weights",
    "loss_function": "DPO standard loss",
    "beta": 0.1,
    "finetune_method": "LoRA (rank=16, alpha=32)",
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "optimizer": "8-bit AdamW (paged)",
    "learning_rate": 5e-6,
    "lr_scheduler": "cosine",
    "warmup_steps": 50,
    "precision": "bf16",
    "distributed_strategy": "DeepSpeed ZeRO-2 + CPU Offload",
    "effective_batch_size": 16,
    "max_seq_length": 3072,
    "num_epochs": "1-3",
}


def validate_plan(args) -> bool:
    """每次训练前校验参数是否与实验方案一致（requirement #4）"""
    checks = []

    # 算法校验
    checks.append(("使用 DPO (非 GRPO)", True))  # 本脚本即 DPO

    # 学习率
    lr_ok = abs(args.learning_rate - 5e-6) < 1e-7
    checks.append((f"学习率 {args.learning_rate} == 5e-6", lr_ok))

    # beta
    beta_ok = abs(args.beta - 0.1) < 1e-7
    checks.append((f"beta {args.beta} == 0.1", beta_ok))

    # LoRA rank
    rank_ok = args.lora_r == 16
    checks.append((f"LoRA rank {args.lora_r} == 16", rank_ok))

    # LoRA alpha
    alpha_ok = args.lora_alpha == 32
    checks.append((f"LoRA alpha {args.lora_alpha} == 32", alpha_ok))

    # bf16
    bf16_ok = args.bf16
    checks.append((f"bf16 = {args.bf16}", bf16_ok))

    # max_seq_length
    seq_ok = args.max_length == 3072
    checks.append((f"max_seq_length {args.max_length} == 3072", seq_ok))

    # optimizer
    optim_ok = args.optim == "paged_adamw_8bit"
    checks.append((f"optimizer {args.optim} == paged_adamw_8bit", optim_ok))

    logger.info("=" * 60)
    logger.info("实验方案校验 (实验方案.md 阶段三)")
    logger.info("=" * 60)
    all_pass = True
    for desc, result in checks:
        status = "✓" if result else "✗"
        logger.info(f"  [{status}] {desc}")
        if not result:
            all_pass = False

    if not all_pass:
        logger.warning("! 部分参数与实验方案不一致，请检查")
        logger.warning("! 确认继续 (y/N)? 如需终止请按 Ctrl+C")
        # 不阻塞，仅仅警告
    else:
        logger.info("  所有参数校验通过 ✓")

    logger.info("=" * 60)
    return all_pass


# ============ 模型加载 ============


def load_model_and_tokenizer(model_path: str, use_flash_attn: bool = False):
    """加载 Qwen3-8B 基座模型和 tokenizer"""
    logger.info(f"加载基座模型: {model_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",  # DPO 需要 left padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_implementation = "flash_attention_2" if use_flash_attn else "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        device_map=None,
    )

    model.config.use_cache = False  # 训练时禁用 KV cache

    logger.info(f"模型参数量: {model.num_parameters() / 1e9:.2f}B")
    return model, tokenizer


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """应用 LoRA"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============ 数据加载 ============


def load_dpo_dataset(data_path: str, max_samples: Optional[int] = None):
    """加载 DPO 格式数据集"""
    logger.info(f"加载 DPO 数据: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    logger.info(f"数据集大小: {len(dataset)} 条")
    return dataset


# ============ 主训练流程 ============


def main():
    parser = argparse.ArgumentParser(description="DPO 策略模型训练")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径，默认使用 Qwen3-8B 基座")
    parser.add_argument("--train_data", type=str, required=True,
                        help="DPO 训练数据路径 (JSONL)")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="DPO 验证数据路径 (JSONL)")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/dpo_checkpoint",
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=3072,
                        help="最大序列长度")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="学习率")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta 参数")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--per_device_batch_size", type=int, default=1,
                        help="每设备批次大小")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="梯度累积步数 (1*16=16 eff. batch)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="最大训练步数（覆盖 epoch）")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="预热步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志间隔")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="保存间隔")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="评估间隔")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="使用 bf16")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit",
                        help="优化器")
    parser.add_argument("--use_flash_attn", action="store_true", default=False,
                        help="使用 Flash Attention 2")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="最大训练样本数（用于快速测试）")
    parser.add_argument("--max_eval_samples", type=int, default=100,
                        help="最大评估样本数")
    parser.add_argument("--skip_plan_check", action="store_true",
                        help="跳过实验方案校验")
    parser.add_argument("--experiment_tag", type=str, default=None,
                        help="实验标签，用于日志命名")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed 配置文件路径")
    # DPO 特有的
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "kto_pair"],
                        help="DPO loss 类型")

    args = parser.parse_args()

    # 设置模型默认路径
    if args.model_path is None:
        args.model_path = (
            "/root/autodl-tmp/huggingface/"
            "models--Qwen--Qwen3-8B/snapshots/"
            "b968826d9c46dd6066d109eabc6255188de91218"
        )

    # 实验标签
    experiment_tag = args.experiment_tag or datetime.now().strftime(
        "dpo_%Y%m%d_%H%M%S"
    )

    # 输出目录
    output_dir = os.path.join(args.output_dir, experiment_tag)
    os.makedirs(output_dir, exist_ok=True)

    # 日志文件
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(file_handler)

    # ---- 实验方案校验 ----
    logger.info(f"实验: {experiment_tag}")
    logger.info(f"实验方案参考: 实验方案.md 阶段三")
    logger.info(f"开始时间: {datetime.now().isoformat()}")

    if not args.skip_plan_check:
        validate_plan(args)

    # ---- 保存训练配置 ----
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    logger.info(f"训练配置已保存: {config_path}")

    # ---- 加载模型 ----
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.use_flash_attn
    )

    # ---- 应用 LoRA ----
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    # ---- 加载数据 ----
    train_dataset = load_dpo_dataset(
        args.train_data, args.max_train_samples
    )
    eval_dataset = None
    if args.eval_data:
        eval_dataset = load_dpo_dataset(
            args.eval_data, args.max_eval_samples
        )

    # ---- 配置 DPO 训练参数 ----
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        beta=args.beta,
        loss_type=args.loss_type,
        bf16=args.bf16,
        fp16=False,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        report_to="tensorboard",
        max_length=args.max_length,
        max_prompt_length=int(args.max_length * 0.6),
        disable_tqdm=False,
        ddp_find_unused_parameters=False,
    )

    # ---- 初始化 DPOTrainer ----
    logger.info("初始化 DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ---- 开始训练 ----
    logger.info("=" * 60)
    logger.info(f"开始 DPO 训练: {experiment_tag}")
    logger.info(f"  模型: {args.model_path}")
    logger.info(f"  训练数据: {args.train_data} ({len(train_dataset)} 条)")
    logger.info(f"  评估数据: {args.eval_data or '无'}")
    logger.info(f"  批次: {args.per_device_batch_size} × "
                f"{args.grad_accum} = "
                f"{args.per_device_batch_size * args.grad_accum}")
    logger.info(f"  max_length: {args.max_length}")
    logger.info(f"  beta: {args.beta}")
    logger.info(f"  loss_type: {args.loss_type}")
    logger.info(f"  有效 batch size: "
                f"{args.per_device_batch_size * args.grad_accum}")
    logger.info("=" * 60)

    try:
        train_result = dpo_trainer.train()
        logger.info("训练完成!")

        # 保存最终模型
        dpo_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存: {output_dir}")

        # 保存训练指标
        metrics = train_result.metrics
        metrics_path = os.path.join(output_dir, "train_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"结束时间: {datetime.now().isoformat()}")
    logger.info(f"实验日志: {log_file}")
    logger.info(f"模型输出: {output_dir}")


if __name__ == "__main__":
    main()
