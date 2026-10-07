"""
训练报告配套可视化脚本
生成 PRM 训练和 DPO 数据构建的关键图表。

用法: python training_visualization.py
图表输出: docs/训练总结/prm_charts/ 和 docs/训练总结/dpo_charts/
"""
import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
LOSS_JSON = '/root/autodl-tmp/prm_v2_loss.json'
LOSS_CSV = '/root/autodl-tmp/prm_v2_loss.csv'
MARGIN_JSON = '/root/autodl-tmp/dpo_margin_analysis.json'
DPO_TRAIN_JSONL = '/root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo/train/dpo_train.jsonl'

OUTPUT_DIR = '/root/workspace/qwen-dianjin/DianJin-PRM/docs/训练总结'
PRM_CHART_DIR = os.path.join(OUTPUT_DIR, 'prm_charts')
DPO_CHART_DIR = os.path.join(OUTPUT_DIR, 'dpo_charts')

os.makedirs(PRM_CHART_DIR, exist_ok=True)
os.makedirs(DPO_CHART_DIR, exist_ok=True)

STYLE = {
    'dpi': 150,
    'bbox_inches': 'tight',
    'pad_inches': 0.2,
}


def load_prm_data():
    with open(LOSS_JSON) as f:
        data = json.load(f)
    losses = data['train_losses']
    evals = data['eval_results']
    steps = [e['step'] for e in losses]
    loss_vals = [e['loss'] for e in losses]
    epochs = [e['step'] / 250 for e in losses]
    return steps, loss_vals, epochs, evals, data


def load_dpo_data():
    with open(MARGIN_JSON) as f:
        data = json.load(f)
    return data


def load_dpo_raw_margins():
    margins = []
    scores_chosen = []
    scores_rejected = []
    with open(DPO_TRAIN_JSONL) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                margins.append(d['metadata']['margin'])
                scores_chosen.append(d['metadata']['chosen_score'])
                scores_rejected.append(d['metadata']['rejected_score'])
    return np.array(margins), np.array(scores_chosen), np.array(scores_rejected)


def save_fig(fig, name, chart_dir):
    path = os.path.join(chart_dir, name)
    fig.savefig(path, **STYLE)
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ============================================================
# PRM Charts
# ============================================================
def plot_prm_loss_curve():
    steps, loss_vals, epochs, evals, data = load_prm_data()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw loss
    ax.plot(steps, loss_vals, alpha=0.3, color='steelblue', linewidth=1, label='Raw Loss (per 10 steps)')

    # Smoothed (moving average, window=5)
    if len(loss_vals) >= 5:
        smoothed = np.convolve(loss_vals, np.ones(5)/5, mode='valid')
        smooth_steps = steps[2:-2]
        ax.plot(smooth_steps, smoothed, color='darkblue', linewidth=2, label='Smoothed (window=5)')

    # Epoch boundary lines
    for ep in range(1, 6):
        ax.axvline(x=ep * 250, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.text(ep * 250, ax.get_ylim()[1] * 0.92, f'Epoch\n{ep}', ha='center', fontsize=9,
                color='gray', alpha=0.7)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('PRM Training Loss Curve (5 Epochs, 1250 Steps)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1250)

    save_fig(fig, '01_loss_curve.png', PRM_CHART_DIR)


def plot_prm_per_epoch_loss():
    steps, loss_vals, epochs, evals, data = load_prm_data()
    epoch_boundaries = [0, 250, 500, 750, 1000, 1250]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for i, ax in enumerate(axes):
        start, end = epoch_boundaries[i], epoch_boundaries[i+1]
        epoch_losses = [l for s, l in zip(steps, loss_vals) if start < s <= end]
        epoch_steps = [s for s in steps if start < s <= end]

        ax.plot(epoch_steps, epoch_losses, 'o-', color=colors[i], markersize=4, linewidth=1.2)
        avg = np.mean(epoch_losses)
        ax.axhline(y=avg, color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.5, 0.92, f'Epoch {i+1}\navg={avg:.2f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Step', fontsize=9)

    axes[0].set_ylabel('Loss', fontsize=11)
    fig.suptitle('PRM Loss per Epoch', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_fig(fig, '02_loss_per_epoch.png', PRM_CHART_DIR)


def plot_prm_loss_distribution():
    steps, loss_vals, epochs, evals, data = load_prm_data()
    epoch_boundaries = [0, 250, 500, 750, 1000, 1250]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for i in range(5):
        ax = axes[i]
        start, end = epoch_boundaries[i], epoch_boundaries[i+1]
        epoch_losses = [l for s, l in zip(steps, loss_vals) if start < s <= end]

        ax.hist(epoch_losses, bins=15, color=colors[i], alpha=0.7, edgecolor='white')
        ax.axvline(np.mean(epoch_losses), color='darkred', linestyle='--', linewidth=1.5,
                   label=f'mean={np.mean(epoch_losses):.2f}')
        ax.set_title(f'Epoch {i+1} (n={len(epoch_losses)})', fontweight='bold')
        ax.set_xlabel('Loss')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Overall histogram
    ax = axes[5]
    ax.hist(loss_vals, bins=30, color='#607D8B', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(loss_vals), color='darkred', linestyle='--', linewidth=1.5,
               label=f'overall mean={np.mean(loss_vals):.2f}')
    ax.set_title('All Epochs Combined', fontweight='bold')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('PRM Loss Distribution by Epoch', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_fig(fig, '03_loss_distribution.png', PRM_CHART_DIR)


def plot_prm_eval_runtime():
    steps, loss_vals, epochs, evals, data = load_prm_data()

    fig, ax = plt.subplots()
    ep_nums = [e['epoch'] for e in evals]
    runtimes = [e['eval_runtime'] for e in evals]
    sps = [e['eval_samples_per_second'] for e in evals]

    ax.bar(ep_nums, runtimes, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Eval Runtime (s)', fontsize=12, color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax.twinx()
    ax2.plot(ep_nums, sps, 'o-', color='darkorange', linewidth=2, markersize=8)
    ax2.set_ylabel('Samples / Second', fontsize=12, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    ax.set_title('PRM Evaluation Performance per Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(ep_nums)
    ax.set_xticklabels([int(e) for e in ep_nums])
    ax.grid(True, alpha=0.3, axis='y')

    save_fig(fig, '04_eval_performance.png', PRM_CHART_DIR)


def plot_prm_convergence_metrics():
    steps, loss_vals, epochs, evals, data = load_prm_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Initial vs Final loss
    ax = axes[0]
    categories = ['Initial\n(Step 10)', 'Epoch 1 End', 'Epoch 2 End', 'Epoch 3 End', 'Epoch 4 End', 'Final\n(Step 1250)']
    values = [loss_vals[0],
              loss_vals[24],
              loss_vals[49],
              loss_vals[74],
              loss_vals[99],
              loss_vals[-1]]
    colors_bar = ['#F44336', '#FF9800', '#FFEB3B', '#8BC34A', '#4CAF50', '#00695C']
    bars = ax.bar(categories, values, color=colors_bar, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss at Epoch Boundaries', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Loss reduction rate
    ax = axes[1]
    epoch_avgs = []
    boundaries = [(0,250), (250,500), (500,750), (750,1000), (1000,1250)]
    for start, end in boundaries:
        epoch_losses = [l for s, l in zip(steps, loss_vals) if start < s <= end]
        epoch_avgs.append(np.mean(epoch_losses))
    reduction_pct = [100 * (epoch_avgs[0] - v) / epoch_avgs[0] for v in epoch_avgs]
    ax.plot(range(1, 6), reduction_pct, 'o-', color='#E91E63', linewidth=2, markersize=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss Reduction (%)', fontsize=11)
    ax.set_title('Cumulative Loss Reduction', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # 3. Loss stability (std per epoch)
    ax = axes[2]
    epoch_stds = []
    for start, end in boundaries:
        epoch_losses = [l for s, l in zip(steps, loss_vals) if start < s <= end]
        epoch_stds.append(np.std(epoch_losses))
    ax.bar(range(1, 6), epoch_stds, color=['#F44336', '#FF9800', '#8BC34A', '#4CAF50', '#00695C'], edgecolor='white')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss Std Dev', fontsize=11)
    ax.set_title('Loss Stability (lower = more stable)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('PRM Training Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_fig(fig, '05_convergence_analysis.png', PRM_CHART_DIR)


# ============================================================
# DPO Charts
# ============================================================
def plot_dpo_margin_distribution():
    data = load_dpo_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    split_names = ['train', 'validate', 'test']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, (name, color) in enumerate(zip(split_names, colors)):
        ax = axes[i]
        split = data[name]
        dist = split['margin_distribution']
        labels = list(dist.keys())
        counts = list(dist.values())

        bars = ax.bar(range(len(labels)), counts, color=color, alpha=0.7, edgecolor='white')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'{name.capitalize()} (n={split["count"]})', fontweight='bold')
        ax.set_xlabel('Margin Range')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', fontsize=7)

    fig.suptitle('DPO Margin Distribution by Dataset Split', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_fig(fig, '11_margin_distribution.png', DPO_CHART_DIR)


def plot_dpo_margin_stats():
    data = load_dpo_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Box plot of margins
    ax = axes[0]
    splits = ['train', 'validate', 'test']
    margin_data = []
    for name in splits:
        s = data[name]['margin']
        margin_data.append([s['min'], s['p25'], s['median'], s['p75'], s['max']])

    bp = ax.boxplot(margin_data, tick_labels=['Train', 'Validate', 'Test'], patch_artist=True,
                     showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    for patch, color in zip(bp['boxes'], ['#2196F3', '#4CAF50', '#FF9800']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add mean annotation
    for i, name in enumerate(splits):
        mean_val = data[name]['margin']['mean']
        ax.annotate(f'mean={mean_val:.3f}', xy=(i+1, mean_val),
                    xytext=(i+1.3, mean_val + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red')

    ax.set_ylabel('Margin', fontsize=12)
    ax.set_title('Margin Distribution (Box Plot)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Threshold attainment
    ax = axes[1]
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    x = np.arange(len(thresholds))
    width = 0.25

    for i, (name, color) in enumerate(zip(splits, ['#2196F3', '#4CAF50', '#FF9800'])):
        ratios = [data[name]['threshold_analysis'][f'margin_ge_{t}']['ratio'] * 100 for t in thresholds]
        ax.bar(x + i * width, ratios, width, label=name, color=color, alpha=0.7, edgecolor='white')

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'>= {t}' for t in thresholds])
    ax.set_ylabel('% of Pairs', fontsize=12)
    ax.set_title('Threshold Attainment Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    fig.suptitle('DPO Margin Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_fig(fig, '12_margin_stats.png', DPO_CHART_DIR)


def plot_dpo_score_scatter():
    margins, chosen, rejected = load_dpo_raw_margins()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Chosen vs Rejected scatter
    ax = axes[0]
    scatter = ax.scatter(rejected, chosen, c=margins, cmap='RdYlGn', alpha=0.6, s=30, edgecolors='gray', linewidth=0.3)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='chosen = rejected')
    ax.set_xlabel('Rejected Score', fontsize=11)
    ax.set_ylabel('Chosen Score', fontsize=11)
    ax.set_title('Chosen vs Rejected Score', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Margin', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Margin histogram
    ax = axes[1]
    ax.hist(margins, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(margins), color='red', linestyle='-', linewidth=1.5,
               label=f'mean={np.mean(margins):.3f}')
    ax.axvline(0.2, color='orange', linestyle='--', linewidth=1.5, label='threshold=0.2')
    ax.set_xlabel('Margin', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Margin Distribution (Train)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Sorted margins (quality curve)
    ax = axes[2]
    sorted_margins = np.sort(margins)
    ax.plot(np.arange(len(sorted_margins)), sorted_margins, color='#4CAF50', linewidth=1.5)
    ax.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='threshold=0.2')
    ax.fill_between(np.arange(len(sorted_margins)), sorted_margins, alpha=0.2, color='#4CAF50')
    ax.set_xlabel('Pair Index (sorted by margin)', fontsize=11)
    ax.set_ylabel('Margin', fontsize=11)
    ax.set_title('Sorted Margin Curve (Train)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add annotation for threshold crossing
    above_thresh = np.sum(margins >= 0.2)
    ax.annotate(f'{above_thresh}/{len(margins)} above 0.2\n({above_thresh/len(margins)*100:.1f}%)',
                xy=(len(margins) - above_thresh, 0.2),
                xytext=(len(margins)*0.3, 0.6),
                arrowprops=dict(arrowstyle='->', color='orange'),
                fontsize=9, color='orange')

    fig.suptitle('DPO Data Quality Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_fig(fig, '13_score_analysis.png', DPO_CHART_DIR)


def plot_dpo_baseline_comparison():
    categories = ['Mean Margin', 'Median Margin', '% Margin >= 0.1', '% Margin >= 0.2', '% Margin >= 0.5',
                  'Chosen Avg', 'Rejected Avg']

    old_values = [0.004, 0.000, 0.8, 0.5, 0.0, 1.000, 0.996]
    new_values = [0.348, 0.311, 82.0, 66.2, 26.3, 0.833, 0.500]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, [v*100 if v <= 1 else v for v in old_values], width,
                   label='Old PRM (ckpt-1169, T=1.0, 7ep)', color='#F44336', alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width/2, [v*100 if v <= 1 else v for v in new_values], width,
                   label='New PRM (ckpt-1250, T=2.0, 5ep)', color='#4CAF50', alpha=0.7, edgecolor='white')

    for bars, vals, color in [(bars1, old_values, '#F44336'), (bars2, new_values, '#4CAF50')]:
        for bar, val in zip(bars, vals):
            display_val = f'{val*100:.0f}%' if val <= 1 else f'{val:.1f}'
            if val < 0.01:
                display_val = f'{val:.1%}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    display_val, ha='center', va='bottom', fontsize=7, fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Value (%)', fontsize=12)
    ax.set_title('DPO Data Quality: Old vs New PRM', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 120)

    # Add improvement annotations
    improvements = ['87x', '--', '100x', '129x', '--', 'Better', 'Better']
    for i, (x_pos, imp) in enumerate(zip(x, improvements)):
        if imp not in ['--', 'Better']:
            ax.annotate(imp, xy=(x_pos, 105), ha='center', fontsize=9, fontweight='bold',
                        color='#00695C',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.8))

    save_fig(fig, '14_baseline_comparison.png', DPO_CHART_DIR)


def plot_dpo_overall_summary():
    data = load_dpo_data()
    overall = data['overall']

    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [overall['margin_ge_0_5'] * 100,
             100 - overall['margin_ge_0_5'] * 100]
    labels = [f'Margin >= 0.5\n({overall["margin_ge_0_5"]*100:.1f}%)',
              f'Margin < 0.5\n({(1-overall["margin_ge_0_5"])*100:.1f}%)']
    colors_pie = ['#4CAF50', '#FFC107']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                       autopct='', startangle=90, textprops={'fontsize': 11})

    ax.set_title('DPO Overall Quality\n(399 pairs, 3 splits combined)', fontsize=14, fontweight='bold')

    # Add key stats as text
    stats_text = (
        f'Mean Margin: {overall["margin_mean"]:.3f}\n'
        f'Median Margin: {overall["margin_median"]:.3f}\n'
        f'Std Dev: {overall["margin_std"]:.3f}\n'
        f'\n'
        f'>= 0.05: {overall["margin_ge_0_05"]*100:.1f}%\n'
        f'>= 0.1 : {overall["margin_ge_0_1"]*100:.1f}%\n'
        f'>= 0.2 : {overall["margin_ge_0_2"]*100:.1f}%\n'
        f'>= 0.5 : {overall["margin_ge_0_5"]*100:.1f}%'
    )
    ax.text(-1.5, -1.2, stats_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    save_fig(fig, '15_overall_summary.png', DPO_CHART_DIR)


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 60)
    print('Training Visualization Generator')
    print('=' * 60)

    print('\n[PRM Charts]')
    plot_prm_loss_curve()
    plot_prm_per_epoch_loss()
    plot_prm_loss_distribution()
    plot_prm_eval_runtime()
    plot_prm_convergence_metrics()
    print(f'  PRM charts saved to {PRM_CHART_DIR}')

    print('\n[DPO Charts]')
    plot_dpo_margin_distribution()
    plot_dpo_margin_stats()
    plot_dpo_score_scatter()
    plot_dpo_baseline_comparison()
    plot_dpo_overall_summary()
    print(f'  DPO charts saved to {DPO_CHART_DIR}')

    print(f'\nDone! {5+5} charts generated.')


if __name__ == '__main__':
    main()
