import os
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/autodl-tmp/huggingface'
os.environ['TRITON_CACHE_DIR'] = '/root/autodl-tmp/triton'
os.makedirs('/root/autodl-tmp/triton', exist_ok=True)

import argparse
import glob
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
import torch
from model.fin_prm import Qwen2ForProcessRewardModel
from model.fin_config import Qwen2PRMConfig


class DataCollatorForProcessReward:
    def __init__(self, tokenizer, max_length=3072):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        texts, all_labels = [], []
        for f in features:
            steps = f["steps"]
            step_contents = list(steps.values())
            trace = '<extra_0>'.join(step_contents) + '<extra_0>'

            knowledge_items = f.get("knowledge_items", {})
            knowledge_text = "\n".join([f"{k}: {v}" for k, v in knowledge_items.items()])

            step_labels = f["step_labels"]
            all_labels.append((step_labels, f["trajectory_label"]))
            text = f"##Question\n{f['question']}\n\n##Knowledge\n{knowledge_text}\n\n##Thinking Trajectory\n{trace}\n\n##Final Answer\n{f['final_answer']}<extra_1>"
            texts.append(text)

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        step_sep_id_0 = self.tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
        step_sep_id_1 = self.tokenizer.encode("<extra_1>", add_special_tokens=False)[0]
        input_ids = batch["input_ids"]

        label_tensors = []
        for (step_labels, trajectory_label), input_id in zip(all_labels, input_ids):
            label_tensor = torch.full_like(input_id, -100)
            mask = (input_id == step_sep_id_0) | (input_id == step_sep_id_1)
            label_positions = mask.nonzero(as_tuple=True)[0]
            for pos, lab in zip(label_positions, step_labels):
                label_tensor[pos] = lab
            if input_id[-1] == step_sep_id_1:
                label_tensor[-1] = trajectory_label
            label_tensors.append(label_tensor)
        batch["labels"] = torch.stack(label_tensors)
        return batch


def load_jsonl_dataset(data_dir):
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {data_dir}")
    datasets = []
    for f in jsonl_files:
        ds = load_dataset("json", data_files=f, split="train")
        datasets.append(ds)
    return concatenate_datasets(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/root/workspace/qwen-dianjin/DianJin-PRM/src/model/config.json")
    parser.add_argument('--pretrained_model_path', type=str, default="/root/autodl-tmp/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument('--train_data_dir', type=str, default='/root/workspace/qwen-dianjin/DianJin-PRM/src/data/train')
    parser.add_argument('--val_data_dir', type=str, default='/root/workspace/qwen-dianjin/DianJin-PRM/src/data/validate')
    parser.add_argument('--test_data_dir', type=str, default='/root/workspace/qwen-dianjin/DianJin-PRM/src/data/test')
    parser.add_argument('--output_path', type=str, default="/root/autodl-tmp/checkpoint")
    parser.add_argument('--max_length', type=int, default=3072)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    args = parser.parse_args()

    config = Qwen2PRMConfig.from_pretrained(args.config_path)
    model = Qwen2ForProcessRewardModel(config=config)
    pretrained_model = AutoModel.from_pretrained(args.pretrained_model_path)
    model.model.load_state_dict(pretrained_model.state_dict(), strict=True)
    del pretrained_model
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    print("Model and tokenizer loaded successfully.")

    special_tokens_dict = {'additional_special_tokens': ['<extra_0>', '<extra_1>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="TOKEN_CLS",
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_jsonl_dataset(args.train_data_dir)
    print(f"Train dataset: {len(train_dataset)} samples")
    val_dataset = load_jsonl_dataset(args.val_data_dir)
    print(f"Validation dataset: {len(val_dataset)} samples")
    test_dataset = load_jsonl_dataset(args.test_data_dir)
    print(f"Test dataset: {len(test_dataset)} samples")

    data_collator = DataCollatorForProcessReward(tokenizer, max_length=args.max_length)

    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        eval_accumulation_steps=1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
        output_dir=args.output_path,
        do_train=True,
        do_eval=True,
        max_length=args.max_length,
        dataset_text_field='text',
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        disable_tqdm=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    trainer.train()

    print("Training complete. Running test evaluation...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
