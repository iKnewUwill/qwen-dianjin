"""
提取 PRM 训练 loss 曲线数据，保存为 CSV 和 JSON。
用法: python extract_prm_loss.py [checkpoint_dir]
"""
import json
import csv
import os
import sys
import glob

DEFAULT_CKPT_DIR = '/root/autodl-tmp/checkpoint_prm_v2'
OUTPUT_CSV = '/root/autodl-tmp/prm_v2_loss.csv'
OUTPUT_JSON = '/root/autodl-tmp/prm_v2_loss.json'


def find_latest_checkpoint(ckpt_dir):
    import re
    ckpts = glob.glob(os.path.join(ckpt_dir, 'checkpoint-*'))
    if not ckpts:
        return None
    def ckpt_num(p):
        m = re.search(r'checkpoint-(\d+)', p)
        return int(m.group(1)) if m else 0
    ckpts.sort(key=ckpt_num)
    return ckpts[-1]


def extract_loss(ckpt_dir):
    ckpt = find_latest_checkpoint(ckpt_dir)
    if not ckpt:
        print(f"未找到 checkpoint in {ckpt_dir}")
        return None

    state_file = os.path.join(ckpt, 'trainer_state.json')
    if not os.path.isfile(state_file):
        print(f"未找到 trainer_state.json in {ckpt}")
        return None

    with open(state_file, 'r') as f:
        state = json.load(f)

    log_history = state.get('log_history', [])
    epoch_info = {
        'epoch': state.get('epoch', 0),
        'global_step': state.get('global_step', 0),
        'best_metric': state.get('best_metric'),
        'best_model_checkpoint': state.get('best_model_checkpoint'),
    }

    train_losses = [(e['step'], e['loss']) for e in log_history if 'loss' in e and 'eval_loss' not in e]
    eval_entries = [e for e in log_history if 'eval_loss' in e]

    result = {
        'checkpoint': ckpt,
        'epoch': epoch_info['epoch'],
        'global_step': epoch_info['global_step'],
        'best_metric': epoch_info['best_metric'],
        'num_train_loss_entries': len(train_losses),
        'num_eval_entries': len(eval_entries),
        'train_losses': [{'step': s, 'loss': l} for s, l in train_losses],
        'eval_results': eval_entries,
    }

    if train_losses:
        result['initial_loss'] = train_losses[0][1]
        result['final_loss'] = train_losses[-1][1]

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss'])
        for step, loss in train_losses:
            writer.writerow([step, loss])
    print(f"Loss CSV saved to {OUTPUT_CSV} ({len(train_losses)} rows)")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Loss JSON saved to {OUTPUT_JSON}")

    print(f"\n{'=' * 50}")
    print(f"PRM Training Summary")
    print(f"{'=' * 50}")
    print(f"  Checkpoint: {os.path.basename(ckpt)}")
    print(f"  Epoch: {epoch_info['epoch']:.2f}")
    print(f"  Global Step: {epoch_info['global_step']}")
    print(f"  Training loss entries: {len(train_losses)}")
    print(f"  Eval entries: {len(eval_entries)}")
    if train_losses:
        print(f"  Initial loss: {train_losses[0][1]:.4f} (step {train_losses[0][0]})")
        print(f"  Final loss:   {train_losses[-1][1]:.6f} (step {train_losses[-1][0]})")
    if eval_entries:
        for eval_entry in eval_entries:
            print(f"  Eval @ epoch {eval_entry.get('epoch', '?'):.1f}: "
                  f"loss={eval_entry.get('eval_loss', 'N/A')}, "
                  f"runtime={eval_entry.get('eval_runtime', 0):.1f}s")

    if len(train_losses) > 10:
        print(f"\n  Loss curve (sampled):")
        for pct in [0, 10, 25, 50, 75, 90, 100]:
            idx = min(int(len(train_losses) * pct / 100), len(train_losses) - 1)
            step, loss = train_losses[idx]
            print(f"    {pct:>3}% | step {step:>5}: loss={loss:.6f}")

    return result


if __name__ == '__main__':
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT_DIR
    extract_loss(ckpt_dir)
