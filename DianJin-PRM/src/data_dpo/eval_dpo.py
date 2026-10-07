"""DPO 训练后验证集/测试集偏好评估。"""
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/huggingface"

BASE = "/root/autodl-tmp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
LORA = "/root/autodl-tmp/dpo_checkpoint/dpo_v3_4000_20261001_203618"
DATA_DIR = "/root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo"
BETA = 0.1
MAX_LENGTH = 3072
SPLITS = {
    "validate": os.path.join(DATA_DIR, "validate", "dpo_val.jsonl"),
    "test": os.path.join(DATA_DIR, "test", "dpo_test.jsonl"),
}


def load_model_with_peft(peft_path=None):
    m = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, device_map="cuda",
    )
    if peft_path:
        m = PeftModel.from_pretrained(m, peft_path)
    m.eval()
    return m


def logprob_of_response(model, tokenizer, prompt, response):
    enc = tokenizer(prompt + response, return_tensors="pt", truncation=True,
                    max_length=MAX_LENGTH).to("cuda")
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH)
    prompt_len = prompt_enc["input_ids"].shape[1]
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    if prompt_len >= seq_len:
        return 0.0
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=enc["attention_mask"]).logits
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, prompt_len:]
    gathered = torch.gather(log_probs[:, prompt_len - 1:, :], 2,
                            targets.unsqueeze(-1)).squeeze(-1)
    return gathered.sum().item()


def evaluate(model, ref_model, tokenizer, data_path):
    pairs = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    correct = 0
    margins = []
    chosen_rewards = []
    rejected_rewards = []
    for i, p in enumerate(pairs):
        prompt = p["prompt"]
        chosen = p["chosen"]
        rejected = p["rejected"]

        lp_c_policy = logprob_of_response(model, tokenizer, prompt, chosen)
        lp_r_policy = logprob_of_response(model, tokenizer, prompt, rejected)
        lp_c_ref = logprob_of_response(ref_model, tokenizer, prompt, chosen)
        lp_r_ref = logprob_of_response(ref_model, tokenizer, prompt, rejected)

        reward_c = BETA * (lp_c_policy - lp_c_ref)
        reward_r = BETA * (lp_r_policy - lp_r_ref)
        margin = reward_c - reward_r
        if margin > 0:
            correct += 1
        margins.append(margin)
        chosen_rewards.append(reward_c)
        rejected_rewards.append(reward_r)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(pairs)}] acc_sofar={correct / (i + 1):.4f}", flush=True)

    n = len(pairs)
    return {
        "count": n,
        "preference_accuracy": correct / n if n else 0.0,
        "mean_reward_chosen": sum(chosen_rewards) / n if n else 0.0,
        "mean_reward_rejected": sum(rejected_rewards) / n if n else 0.0,
        "mean_margin": sum(margins) / n if n else 0.0,
        "correct": correct,
    }


def main():
    print("加载 tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载 reference 模型 (base) ...", flush=True)
    ref_model = load_model_with_peft(None)

    print("加载 policy 模型 (base + LoRA) ...", flush=True)
    model = load_model_with_peft(LORA)

    results = {}
    for name, path in SPLITS.items():
        print(f"\n=== 评估 {name} ===", flush=True)
        results[name] = evaluate(model, ref_model, tokenizer, path)
        r = results[name]
        print(f"  {name}: accuracy={r['preference_accuracy']:.4f} "
              f"({r['correct']}/{r['count']}), mean_margin={r['mean_margin']:.4f}, "
              f"chosen_reward={r['mean_reward_chosen']:.4f}, "
              f"rejected_reward={r['mean_reward_rejected']:.4f}", flush=True)

    out = "/root/autodl-tmp/dpo_v3_eval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}", flush=True)


if __name__ == "__main__":
    main()
