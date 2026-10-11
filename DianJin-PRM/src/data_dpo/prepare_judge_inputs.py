#!/usr/bin/env python3
"""准备三维度评判输入：合并 base / dpo_step_mean / dpo_fusion_weighted 输出，
打乱顺序（盲评），格式化为 235B 评判可直接使用的 JSONL。

用法:
    python prepare_judge_inputs.py [--out judge_inputs.jsonl]

输出每行:
    {"sample_id": "dpo_step_mean_0000", "model": "dpo_step_mean",
     "question": "...", "knowledge_items": {...},
     "steps_text": "步骤1: ...\n步骤2: ...", "final_answer": "..."}
其中 model 字段用于事后反盲评（评判时不要把它喂给 235B）。
"""
import os
import sys
import json
import argparse
import random

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(DATA_DIR, "test")

DIRS = {
    "base": os.path.join(TEST_DIR, "base"),
    "dpo_step_mean": os.path.join(TEST_DIR, "dpo_step_mean"),
    "dpo_fusion_weighted": os.path.join(TEST_DIR, "dpo_fusion_weighted"),
}

SEED = 42


def steps_to_text(generation):
    proc = generation.get("分析过程", {})
    if isinstance(proc, dict):
        return "\n".join(f"{k}: {v}" for k, v in proc.items())
    return str(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "judge_inputs.jsonl"))
    args = parser.parse_args()

    entries = []
    for model_tag, d in DIRS.items():
        if not os.path.isdir(d):
            print(f"[warn] 目录不存在，跳过: {d}")
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
        for f in files:
            obj = json.load(open(os.path.join(d, f), encoding="utf-8"))
            gen = obj.get("generation")
            if not isinstance(gen, dict) or "分析过程" not in gen or "分析结论" not in gen:
                continue
            entries.append({
                "sample_id": f"{model_tag}_{f[:-5]}",
                "model": model_tag,
                "question": obj.get("question", ""),
                "knowledge_items": obj.get("knowledge_items", {}),
                "steps_text": steps_to_text(gen),
                "final_answer": gen.get("分析结论", ""),
            })

    random.seed(SEED)
    random.shuffle(entries)

    with open(args.out, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"评判输入已生成: {args.out} (共 {len(entries)} 条, 已打乱盲评)")
    from collections import Counter
    print("分布:", dict(Counter(e["model"] for e in entries)))


if __name__ == "__main__":
    main()
