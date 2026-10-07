"""
DPO 数据构建（4000 样本）margin 分布可视化脚本。

用法: python dpo_margin_visualization.py
输出: docs/训练总结/dpo_charts/ 下 18~22 号图表
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = '/root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo'
SPLITS = {
    'train': os.path.join(BASE, 'train', 'dpo_train.jsonl'),
    'validate': os.path.join(BASE, 'validate', 'dpo_val.jsonl'),
    'test': os.path.join(BASE, 'test', 'dpo_test.jsonl'),
}
CHART_DIR = '/root/workspace/qwen-dianjin/DianJin-PRM/docs/训练总结/dpo_charts'
os.makedirs(CHART_DIR, exist_ok=True)

STYLE = {'dpi': 150, 'bbox_inches': 'tight', 'pad_inches': 0.2}
COLORS = {'train': '#2196F3', 'validate': '#4CAF50', 'test': '#FF9800'}


def load_split(name):
    margins, chosen, rejected = [], [], []
    with open(SPLITS[name], encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            m = d['metadata']
            margins.append(m['margin'])
            chosen.append(m['chosen_score'])
            rejected.append(m['rejected_score'])
    return np.array(margins), np.array(chosen), np.array(rejected)


def save_fig(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, **STYLE)
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_margin_distribution(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    buckets = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [f'{buckets[i]:.2f}-\n{buckets[i+1]:.2f}' for i in range(len(buckets) - 1)]

    for i, name in enumerate(['train', 'validate', 'test']):
        ax = axes[i]
        margins = data[name]['margins']
        counts, _ = np.histogram(margins, bins=buckets)
        x = np.arange(len(labels))
        bars = ax.bar(x, counts, color=COLORS[name], alpha=0.7, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel('Margin Range')
        ax.set_ylabel('Count')
        ax.set_title(f'{name.capitalize()} (n={len(margins)})', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                        str(int(c)), ha='center', fontsize=7)

    fig.suptitle('DPO Margin Distribution by Split (4000 samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '18_margin_distribution_4000.png')


def plot_margin_stats(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    names = ['train', 'validate', 'test']
    margin_data = [data[n]['margins'] for n in names]
    bp = ax.boxplot(margin_data, tick_labels=['Train', 'Validate', 'Test'], patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    for patch, name in zip(bp['boxes'], names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.6)
    for i, name in enumerate(names):
        m = data[name]['margins']
        ax.annotate(f'mean={m.mean():.3f}', xy=(i + 1, m.mean()),
                    xytext=(i + 1.25, m.mean() + 0.06),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
    ax.set_ylabel('Margin')
    ax.set_title('Margin Distribution (Box Plot)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    x = np.arange(len(thresholds))
    width = 0.25
    for i, name in enumerate(names):
        margins = data[name]['margins']
        ratios = [100 * np.mean(margins >= t) for t in thresholds]
        ax.bar(x + i * width, ratios, width, label=name.capitalize(), color=COLORS[name],
               alpha=0.7, edgecolor='white')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'>= {t}' for t in thresholds])
    ax.set_ylabel('% of Pairs')
    ax.set_title('Threshold Attainment Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    fig.suptitle('DPO Margin Statistics (4000 samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '19_margin_stats_4000.png')


def plot_score_analysis(data):
    all_margins = np.concatenate([data[n]['margins'] for n in ['train', 'validate', 'test']])
    all_chosen = np.concatenate([data[n]['chosen'] for n in ['train', 'validate', 'test']])
    all_rejected = np.concatenate([data[n]['rejected'] for n in ['train', 'validate', 'test']])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    scatter = ax.scatter(all_rejected, all_chosen, c=all_margins, cmap='RdYlGn', alpha=0.5,
                         s=20, edgecolors='gray', linewidth=0.2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='chosen = rejected')
    ax.set_xlabel('Rejected Score')
    ax.set_ylabel('Chosen Score')
    ax.set_title('Chosen vs Rejected Score (all splits)', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Margin', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(all_margins, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(all_margins.mean(), color='red', linewidth=1.5, label=f'mean={all_margins.mean():.3f}')
    ax.axvline(0.2, color='orange', linestyle='--', linewidth=1.5, label='threshold=0.2')
    ax.set_xlabel('Margin')
    ax.set_ylabel('Count')
    ax.set_title('Overall Margin Histogram', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    sorted_m = np.sort(all_margins)
    ax.plot(np.arange(len(sorted_m)), sorted_m, color='#4CAF50', linewidth=1.2)
    ax.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='threshold=0.2')
    ax.fill_between(np.arange(len(sorted_m)), sorted_m, alpha=0.15, color='#4CAF50')
    ax.set_xlabel('Pair Index (sorted by margin)')
    ax.set_ylabel('Margin')
    ax.set_title('Sorted Margin Curve', fontweight='bold')
    above = int(np.sum(all_margins >= 0.2))
    ax.annotate(f'{above}/{len(all_margins)} above 0.2\n({above / len(all_margins) * 100:.1f}%)',
                xy=(len(all_margins) - above, 0.2),
                xytext=(len(all_margins) * 0.25, 0.55),
                arrowprops=dict(arrowstyle='->', color='orange'), fontsize=9, color='orange')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('DPO Data Quality Analysis (4000 samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '20_score_analysis_4000.png')


def plot_baseline_comparison():
    categories = ['Mean\nMargin', 'Median\nMargin', '% Margin\n>= 0.2', '% Margin\n>= 0.5',
                  'Chosen\nAvg', 'Rejected\nAvg']
    old_values = [0.348, 0.311, 64.5, 23.7, 0.833, 0.500]
    new_values = [0.686, 0.762, 95.4, 77.0, 0.858, 0.171]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, [v * 100 if v <= 1 else v for v in old_values], width,
                   label='Old build (6/15, 399 pairs)', color='#F44336', alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width / 2, [v * 100 if v <= 1 else v for v in new_values], width,
                   label='New build (10/01, 4000 pairs)', color='#4CAF50', alpha=0.7, edgecolor='white')

    for bars, vals in [(bars1, old_values), (bars2, new_values)]:
        for bar, val in zip(bars, vals):
            label = f'{val * 100:.1f}%' if val <= 1 else f'{val:.1f}%'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Value (%)')
    ax.set_title('DPO Data Quality: Old vs New Build', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 125)
    save_fig(fig, '21_baseline_comparison_4000.png')


def plot_overall_summary(data):
    all_margins = np.concatenate([data[n]['margins'] for n in ['train', 'validate', 'test']])

    fig, ax = plt.subplots(figsize=(8, 8))
    ge_05 = 100 * np.mean(all_margins >= 0.5)
    sizes = [ge_05, 100 - ge_05]
    labels = [f'Margin >= 0.5\n({ge_05:.1f}%)', f'Margin < 0.5\n({100 - ge_05:.1f}%)']
    ax.pie(sizes, explode=(0.05, 0), labels=labels, colors=['#4CAF50', '#FFC107'],
           startangle=90, textprops={'fontsize': 11},
           autopct='', wedgeprops=dict(edgecolor='white'))

    stats_text = (
        f'Mean Margin: {all_margins.mean():.3f}\n'
        f'Median Margin: {np.median(all_margins):.3f}\n'
        f'Std Dev: {all_margins.std():.3f}\n'
        f'Min / Max: {all_margins.min():.3f} / {all_margins.max():.3f}\n'
        f'\n'
        f'>= 0.05: {100 * np.mean(all_margins >= 0.05):.1f}%\n'
        f'>= 0.10: {100 * np.mean(all_margins >= 0.10):.1f}%\n'
        f'>= 0.20: {100 * np.mean(all_margins >= 0.20):.1f}%\n'
        f'>= 0.50: {100 * np.mean(all_margins >= 0.50):.1f}%'
    )
    ax.text(-1.5, -1.2, stats_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
    ax.set_title('DPO Overall Quality\n(4000 pairs, 3 splits combined)', fontsize=14, fontweight='bold')
    save_fig(fig, '22_overall_summary_4000.png')


def main():
    print('Loading DPO split data...')
    data = {}
    for name in ['train', 'validate', 'test']:
        margins, chosen, rejected = load_split(name)
        data[name] = {'margins': margins, 'chosen': chosen, 'rejected': rejected}
        print(f'  {name}: {len(margins)} pairs, mean margin={margins.mean():.4f}')

    print('\n[DPO Charts - 4000 samples]')
    plot_margin_distribution(data)
    plot_margin_stats(data)
    plot_score_analysis(data)
    plot_baseline_comparison()
    plot_overall_summary(data)
    print(f'\nDone! 5 charts saved to {CHART_DIR}')


if __name__ == '__main__':
    main()
