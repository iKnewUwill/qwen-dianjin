"""
DPO 数据集 margin 分析脚本
分析 build_dpo_dataset.py 构建的 DPO 数据集中的 margin 分布。
"""
import os
import sys
import json
import glob
import numpy as np

DATA_DIR = '/root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo'
OUTPUT_PATH = '/root/autodl-tmp/dpo_margin_analysis.json'


def load_jsonl(filepath):
    samples = []
    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


def analyze_split(name, filepath):
    samples = load_jsonl(filepath)
    if not samples:
        print(f"  {name}: 无数据")
        return None, None

    margins = [s['metadata']['margin'] for s in samples]
    chosen_scores = [s['metadata']['chosen_score'] for s in samples]
    rejected_scores = [s['metadata']['rejected_score'] for s in samples]
    num_candidates = [s['metadata']['num_candidates'] for s in samples]

    margins = np.array(margins)
    chosen_scores = np.array(chosen_scores)
    rejected_scores = np.array(rejected_scores)

    stats = {
        'count': len(samples),
        'margin': {
            'mean': float(np.mean(margins)),
            'median': float(np.median(margins)),
            'std': float(np.std(margins)),
            'min': float(np.min(margins)),
            'max': float(np.max(margins)),
            'p25': float(np.percentile(margins, 25)),
            'p75': float(np.percentile(margins, 75)),
            'p90': float(np.percentile(margins, 90)),
            'p95': float(np.percentile(margins, 95)),
        },
        'chosen_score': {
            'mean': float(np.mean(chosen_scores)),
            'median': float(np.median(chosen_scores)),
            'std': float(np.std(chosen_scores)),
        },
        'rejected_score': {
            'mean': float(np.mean(rejected_scores)),
            'median': float(np.median(rejected_scores)),
            'std': float(np.std(rejected_scores)),
        },
        'candidates': {
            'mean': float(np.mean(num_candidates)),
            'min': int(np.min(num_candidates)),
            'max': int(np.max(num_candidates)),
        },
    }

    buckets = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bucket_counts = {}
    for i in range(len(buckets) - 1):
        low, high = buckets[i], buckets[i + 1]
        count = int(np.sum((margins >= low) & (margins < high)))
        bucket_counts[f'[{low:.2f}, {high:.2f})'] = count
    count_ge_1 = int(np.sum(margins >= 1.0))
    if count_ge_1 > 0:
        bucket_counts['[1.00, ...]'] = count_ge_1

    stats['margin_distribution'] = bucket_counts

    # Threshold analysis (from 实验方案.md: margin > 0.2 is the quality threshold)
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    threshold_stats = {}
    for t in thresholds:
        count_above = int(np.sum(margins >= t))
        threshold_stats[f'margin_ge_{t}'] = {
            'count': count_above,
            'ratio': round(count_above / len(samples), 4),
        }
    stats['threshold_analysis'] = threshold_stats

    print(f"  {name}: {len(samples)} 条偏好对")
    print(f"    Margin: mean={stats['margin']['mean']:.4f}, median={stats['margin']['median']:.4f}, "
          f"std={stats['margin']['std']:.4f}")
    print(f"    Margin range: [{stats['margin']['min']:.4f}, {stats['margin']['max']:.4f}]")
    print(f"    Margin P25={stats['margin']['p25']:.4f}, P75={stats['margin']['p75']:.4f}, "
          f"P90={stats['margin']['p90']:.4f}")
    print(f"    Chosen score: mean={stats['chosen_score']['mean']:.4f}, "
          f"Rejected score: mean={stats['rejected_score']['mean']:.4f}")
    print(f"    Margin distribution:")
    for bucket, count in bucket_counts.items():
        bar = '█' * int(count / max(1, len(samples)) * 50)
        print(f"      {bucket:>15}: {count:>5} ({count/len(samples)*100:5.1f}%) {bar}")
    print(f"    Threshold analysis (margin >= t):")
    for t_str, tstat in threshold_stats.items():
        t_val = float(t_str.replace('margin_ge_', ''))
        print(f"      >= {t_val:.2f}: {tstat['count']:>5} ({tstat['ratio']*100:5.1f}%)")

    return stats, margins


def main():
    print("=" * 60)
    print("DPO 数据集 Margin 分析")
    print("=" * 60)

    all_stats = {}
    all_margins = []

    for name, filename in [
        ('train', 'train/dpo_train.jsonl'),
        ('validate', 'validate/dpo_val.jsonl'),
        ('test', 'test/dpo_test.jsonl'),
    ]:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"\n--- {name} ---")
        stats, margins = analyze_split(name, filepath)
        if stats:
            all_stats[name] = stats
            all_margins.extend(margins.tolist())

    if all_margins:
        all_margins = np.array(all_margins)
        print(f"\n{'=' * 60}")
        print(f"整体统计 (all splits)")
        print(f"{'=' * 60}")
        print(f"  总偏好对: {len(all_margins)}")
        print(f"  Margin mean={np.mean(all_margins):.4f}, median={np.median(all_margins):.4f}, "
              f"std={np.std(all_margins):.4f}")
        print(f"  Margin >= 0.2: {int(np.sum(all_margins >= 0.2))} "
              f"({np.sum(all_margins >= 0.2)/len(all_margins)*100:.1f}%)")
        print(f"  Margin >= 0.1: {int(np.sum(all_margins >= 0.1))} "
              f"({np.sum(all_margins >= 0.1)/len(all_margins)*100:.1f}%)")
        print(f"  Margin >= 0.05: {int(np.sum(all_margins >= 0.05))} "
              f"({np.sum(all_margins >= 0.05)/len(all_margins)*100:.1f}%)")
        print(f"  Margin >= 0.5: {int(np.sum(all_margins >= 0.5))} "
              f"({np.sum(all_margins >= 0.5)/len(all_margins)*100:.1f}%)")

        all_stats['overall'] = {
            'count': int(len(all_margins)),
            'margin_mean': float(np.mean(all_margins)),
            'margin_median': float(np.median(all_margins)),
            'margin_std': float(np.std(all_margins)),
            'margin_ge_0_05': float(np.sum(all_margins >= 0.05) / len(all_margins)),
            'margin_ge_0_1': float(np.sum(all_margins >= 0.1) / len(all_margins)),
            'margin_ge_0_2': float(np.sum(all_margins >= 0.2) / len(all_margins)),
            'margin_ge_0_5': float(np.sum(all_margins >= 0.5) / len(all_margins)),
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存到: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
