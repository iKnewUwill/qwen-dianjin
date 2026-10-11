#!/usr/bin/env python3
"""汇总三维度评判结果，输出 DPO vs BASE 对比表。

用法:
    python aggregate_judge.py --judge_inputs judge_inputs.jsonl --judge_results judge_results.jsonl

judge_results.jsonl 每行（由你调 235B 后自行生成）:
    {"sample_id": "base_0000", "过程": 1, "结论": 1, "数据引用": 1}
评分口径（与 235B评估方案.md §5 一致）:
    过程/数据引用: 0 或 1；结论: 0 / 0.5 / 1
"""
import json
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_inputs", required=True)
    parser.add_argument("--judge_results", required=True)
    args = parser.parse_args()

    id2model = {}
    with open(args.judge_inputs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            id2model[e["sample_id"]] = e["model"]

    scores = defaultdict(list)
    with open(args.judge_results, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r["sample_id"]
            if sid not in id2model:
                continue
            model = id2model[sid]
            scores[model].append(r)

    models = ["base", "dpo_step_mean", "dpo_fusion_weighted"]
    print(f"{'模型':<20} {'过程正确率':>10} {'结论正确率':>10} {'数据引用准确率':>12} {'样本数':>6}")
    print("-" * 62)
    for m in models:
        rows = scores.get(m, [])
        if not rows:
            print(f"{m:<20} {'-':>10} {'-':>10} {'-':>12} {0:>6}")
            continue
        n = len(rows)
        proc = sum(r.get("过程", 0) for r in rows) / n
        concl = sum(1 for r in rows if r.get("结论", 0) == 1) / n
        cite = sum(r.get("数据引用", 0) for r in rows) / n
        print(f"{m:<20} {proc:>10.2%} {concl:>10.2%} {cite:>12.2%} {n:>6}")

    print("\n提升幅度（相对 base）:")
    base_rows = scores.get("base", [])
    if base_rows:
        bn = len(base_rows)
        b = {
            "过程": sum(r.get("过程", 0) for r in base_rows) / bn,
            "结论": sum(1 for r in base_rows if r.get("结论", 0) == 1) / bn,
            "数据引用": sum(r.get("数据引用", 0) for r in base_rows) / bn,
        }
        for m in ["dpo_step_mean", "dpo_fusion_weighted"]:
            rows = scores.get(m, [])
            if not rows:
                continue
            n = len(rows)
            d = {
                "过程": sum(r.get("过程", 0) for r in rows) / n,
                "结论": sum(1 for r in rows if r.get("结论", 0) == 1) / n,
                "数据引用": sum(r.get("数据引用", 0) for r in rows) / n,
            }
            lift = {k: d[k] - b[k] for k in d}
            print(f"  {m}: 过程 {lift['过程']:+.2%}, 结论 {lift['结论']:+.2%}, 数据引用 {lift['数据引用']:+.2%}")


if __name__ == "__main__":
    main()
