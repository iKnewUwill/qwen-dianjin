"""校验 DPO vs Base 评估生成结果的完整性。

检查项：
- base/ 与 dpo/ 目录文件数（应为 400）
- 每个文件是否为有效 JSON 且含「分析过程 + 分析结论」
- 编号 0-399 一一对应
- 生成失败 / 重试统计

用法:
    python verify_generation.py [--data_dir src/data_dpo/test]
"""

import os
import sys
import json
import argparse
from collections import Counter


def verify_one_dir(d):
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    ids = set()
    parsed_ok = 0
    fail_parse = []
    missing_structure = []
    retry_counter = Counter()

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            fail_parse.append(fp)
            continue

        idx = obj.get("id")
        ids.add(idx)
        retry_counter[obj.get("num_retries")] += 1

        gen = obj.get("generation")
        if isinstance(gen, dict) and "分析过程" in gen and "分析结论" in gen:
            parsed_ok += 1
        else:
            missing_structure.append((idx, fp))

    return {
        "dir": d,
        "total_files": len(files),
        "parsed_ok": parsed_ok,
        "fail_parse": fail_parse,
        "missing_structure": missing_structure,
        "ids": ids,
        "retry_counter": retry_counter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test"))
    args = parser.parse_args()

    base_dir = os.path.join(args.data_dir, "base")
    dpo_dir = os.path.join(args.data_dir, "dpo")

    report = {}
    for label, d in [("base", base_dir), ("dpo", dpo_dir)]:
        if not os.path.isdir(d):
            print(f"[{label}] 目录不存在: {d}")
            continue
        r = verify_one_dir(d)
        report[label] = r
        print("=" * 60)
        print(f"[{label}] {d}")
        print(f"  文件总数: {r['total_files']}")
        print(f"  有效解析(含分析过程+分析结论): {r['parsed_ok']}")
        print(f"  JSON 解析失败: {len(r['fail_parse'])}")
        for fp in r["fail_parse"][:10]:
            print(f"    - {os.path.basename(fp)}")
        print(f"  缺少结构(分析过程/分析结论): {len(r['missing_structure'])}")
        for idx, fp in r["missing_structure"][:10]:
            print(f"    - id={idx} {os.path.basename(fp)}")
        print(f"  重试次数分布: {dict(sorted(r['retry_counter'].items()))}")

    if "base" in report and "dpo" in report:
        base_ids = report["base"]["ids"]
        dpo_ids = report["dpo"]["ids"]
        expected = set(range(400))
        print("=" * 60)
        print("[编号对应]")
        print(f"  base 覆盖 {len(base_ids)} 个 id, dpo 覆盖 {len(dpo_ids)} 个 id")
        print(f"  base 缺失: {sorted(expected - base_ids)}")
        print(f"  dpo 缺失: {sorted(expected - dpo_ids)}")
        print(f"  base/dpo id 集合一致: {base_ids == dpo_ids}")

        full_ok = (
            len(base_ids) == 400
            and len(dpo_ids) == 400
            and base_ids == dpo_ids
            and report["base"]["parsed_ok"] == 400
            and report["dpo"]["parsed_ok"] == 400
        )
        print("=" * 60)
        print(f"最终结论: {'✅ 全部通过' if full_ok else '❌ 存在缺口，见上方明细'}")


if __name__ == "__main__":
    main()
