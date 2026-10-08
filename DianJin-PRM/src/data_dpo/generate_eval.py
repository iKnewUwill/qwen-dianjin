"""
DPO vs Base 评估生成脚本
========================
依据 docs/235B评估方案.md：
- 对 dpo_test.jsonl 的 400 个样本，用基座 Qwen3-8B 与 DPO 模型分别生成「分析过程 + 分析结论」
- 提示词使用 data_dpo/prompt.py 中的 solver_system_prompt（要求 JSON 输出）
- 输出 JSON（分析过程 + 分析结论），提取失败最多重试 3 次
- 结果保存到 data_dpo/test/{base,dpo}/，编号一一对应（0000.json ~ 0399.json）

用法:
    python generate_eval.py --model base|dpo [--start_index 0] [--max_samples 10] [--force]
"""

import os
import sys
import json
import re
import time
import argparse
from datetime import datetime

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/huggingface"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from prompt import PromptsDPOConfig

BASE_MODEL = "/root/autodl-tmp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
LORA_PATH = "/root/autodl-tmp/dpo_checkpoint/dpo_v3_4000_20261001_203618"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(DATA_DIR, "test", "dpo_test.jsonl")

# prompt 字段中的反解标记（与 build_dpo_dataset.py 的 build_dpo_prompt 保持一致）
GUIDELINES_MARKER = "through the above guidelines:"
KNOWLEDGE_MARKER = "\n\n参考知识：\n"

# 生成参数：基座与 DPO 完全一致（235B评估方案 §3）
GEN_KWARGS = dict(
    max_new_tokens=2048,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.05,
)
MAX_RETRIES = 3  # 提取失败后最多重复生成的次数；总生成次数 = 1 + MAX_RETRIES


def extract_question_knowledge(prompt):
    """从 dpo_test.jsonl 的 prompt 字段反解 question 与 knowledge_items。"""
    q_idx = prompt.rfind(GUIDELINES_MARKER)
    if q_idx == -1:
        raise ValueError(f"未找到问题标记: {GUIDELINES_MARKER!r}")

    q_start = q_idx + len(GUIDELINES_MARKER)
    k_idx = prompt.find(KNOWLEDGE_MARKER, q_start)
    if k_idx == -1:
        question = prompt[q_start:].strip()
        knowledge_text = ""
    else:
        question = prompt[q_start:k_idx].strip()
        knowledge_text = prompt[k_idx + len(KNOWLEDGE_MARKER):].strip()

    knowledge_items = {}
    for line in knowledge_text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v in ("None", "null", ""):
            continue
        try:
            v = float(v)
            if v.is_integer():
                v = int(v)
        except ValueError:
            pass
        knowledge_items[k] = v

    return question, knowledge_items


def _strip_fences(text):
    t = text.strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    return t


def _strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_json(text):
    """从模型原始输出中尽力提取 JSON 对象；失败返回 None。"""
    if not text:
        return None

    candidates = [
        text,
        _strip_fences(text),
        _strip_think_tags(text),
        _strip_fences(_strip_think_tags(text)),
    ]
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def load_model_and_tokenizer(model_type):
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="cuda",
    )
    if model_type == "dpo":
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer


def build_messages(system_prompt, question, knowledge_items):
    ki_json = json.dumps(knowledge_items, ensure_ascii=False, indent=2)
    user = f"问题：{question}\n\n已知财务数据（JSON 格式）：\n{ki_json}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def generate_batch(model, tokenizer, messages_list):
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, padding_side="left"
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            **GEN_KWARGS,
        )

    generated = outputs[:, prompt_len:]
    raws = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [r.strip() for r in raws]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="DPO/Base 评估生成")
    parser.add_argument("--model", required=True, choices=["base", "dpo"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="覆盖生成时的 max_new_tokens（默认 2048）")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens is not None:
        GEN_KWARGS["max_new_tokens"] = args.max_new_tokens

    system_prompt = PromptsDPOConfig().solver_system_prompt
    out_dir = os.path.join(DATA_DIR, "test", args.model)
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    total = len(samples)
    end = total if args.max_samples is None else min(total, args.start_index + args.max_samples)
    indices = list(range(args.start_index, end))

    log(f"模型类型: {args.model}")
    log(f"测试集样本总数: {total}, 本次处理索引 [{args.start_index}, {end}) 共 {len(indices)} 条")
    log(f"生成参数: {GEN_KWARGS}, batch_size={args.batch_size}")

    log("加载模型与 tokenizer ...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    gpu_free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    log(f"模型加载完成, GPU 剩余显存: {gpu_free:.1f} GB")

    items = []
    skipped = 0
    for idx in indices:
        out_path = os.path.join(out_dir, f"{idx:04d}.json")
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue
        question, knowledge_items = extract_question_knowledge(samples[idx].get("prompt", ""))
        items.append({
            "idx": idx,
            "question": question,
            "knowledge_items": knowledge_items,
            "messages": build_messages(system_prompt, question, knowledge_items),
            "raw": "",
            "parsed": None,
            "attempts": 0,
            "gen_time_sec": 0.0,
        })

    if skipped:
        log(f"跳过已存在输出: {skipped} 条")
    log(f"待生成: {len(items)} 条")

    def write_result(it):
        out_path = os.path.join(out_dir, f"{it['idx']:04d}.json")
        result = {
            "id": it["idx"],
            "model": args.model,
            "question": it["question"],
            "knowledge_items": it["knowledge_items"],
            "raw_output": it["raw"],
            "generation": it["parsed"],
            "num_retries": it["attempts"] - 1,
            "gen_time_sec": round(it["gen_time_sec"], 3),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    t_start = time.time()
    for attempt in range(1, MAX_RETRIES + 2):
        pending = [it for it in items if it["parsed"] is None and it["attempts"] < attempt]
        if not pending:
            break
        log(f"--- 第 {attempt} 轮: {len(pending)} 条 ---")
        for b in range(0, len(pending), args.batch_size):
            chunk = pending[b:b + args.batch_size]
            t0 = time.time()
            raws = generate_batch(model, tokenizer, [it["messages"] for it in chunk])
            dt = time.time() - t0
            per = dt / len(chunk)
            for it, raw in zip(chunk, raws):
                it["raw"] = raw
                it["attempts"] = attempt
                it["gen_time_sec"] += per
                it["parsed"] = extract_json(raw)
                if it["parsed"] is not None:
                    write_result(it)
            done_ok = sum(1 for it in chunk if it["parsed"] is not None)
            done_total = sum(1 for it in items if it["parsed"] is not None)
            elapsed = time.time() - t_start
            rate = elapsed / done_total if done_total else 0
            eta_min = (rate * (len(items) - done_total)) / 60 if done_total else 0
            log(f"  批次 {b // args.batch_size + 1}: {len(chunk)} 条 {dt:.1f}s "
                f"({per:.1f}s/条) 成功 {done_ok}/{len(chunk)} "
                f"| 累计 {done_total}/{len(items)} ({done_total / len(items) * 100:.0f}%) "
                f"| ETA {eta_min:.0f}min")

    for it in items:
        if it["parsed"] is None:
            write_result(it)

    t_total = time.time() - t_start
    n = len(items)
    ok_count = sum(1 for it in items if it["parsed"] is not None)
    fail_count = n - ok_count
    total_attempts = sum(it["attempts"] for it in items)

    log("=" * 60)
    log(f"完成: 生成 {n} 条 (成功解析 {ok_count}, 失败 {fail_count}, 总尝试 {total_attempts} 次)")
    if n:
        log(f"速度统计: 总耗时 {t_total:.1f}s, 平均 {t_total / n:.2f}s/样本 (batch={args.batch_size})")
    full_eta = (t_total / n) * total / 60 if n else 0
    log(f"按当前速度全量 {total} 条预计耗时: {full_eta:.0f} 分钟 ({full_eta / 60:.1f} 小时)")
    log(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
