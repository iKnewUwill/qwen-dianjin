"""
使用 PRM 模型对 pre/ 样本打分，构建 DPO 数据集（步骤级 + 轨迹级评分，4 种排序模式）。

两阶段设计（避免重复 PRM 打分）：
  阶段1（打分 + 缓存）:
      python build_dpo_dataset.py --cache_path pre/scored_cache.jsonl
      对每个问题的候选回答用 PRM 打分，一次输出步骤级(<extra_0>)与轨迹级(<extra_1>)分数，写入缓存。
  阶段2（从缓存生成，可多次、快速）:
      python build_dpo_dataset.py --from_cache pre/scored_cache.jsonl \
          --ranking_mode <mode> --output_dir rankings/<mode>

排序模式 ranking_mode：
  - trajectory      : 仅轨迹级，combined = traj_prob（旧方案，基线组）
  - step_mean       : 仅步骤平均，combined = mean(step_probs)（实验组A）
  - fusion_static   : 静态权重 α=0.5，combined = 0.5*mean(step_probs) + 0.5*traj_prob（实验组B）
  - fusion_weighted : 加权融合，combined = alpha*mean(step_probs) + beta*traj_prob（完整方案，
                      默认 alpha=0.7, beta=0.3，因步骤级准确率94% > 轨迹级78%，偏向步骤级）
"""

import os
import sys
import json
import glob
import argparse
import random
import time
from datetime import datetime
import torch

os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/autodl-tmp/huggingface'

sys.path.insert(0, '/root/workspace/qwen-dianjin/DianJin-PRM/src')

from transformers import AutoTokenizer, AutoModel
from model.fin_prm import Qwen3ForProcessRewardModel
from model.fin_config import Qwen3PRMConfig
from peft import PeftModel

BASE_DIR = '/root/workspace/qwen-dianjin/DianJin-PRM/src/data_dpo'
PRE_DIR = os.path.join(BASE_DIR, 'pre')
RANKINGS_DIR = os.path.join(BASE_DIR, 'rankings')

CONFIG_PATH = '/root/workspace/qwen-dianjin/DianJin-PRM/src/model/config.json'
BASE_MODEL_PATH = '/root/autodl-tmp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
PRM_CKPT_BASE = '/root/autodl-tmp/checkpoint_prm_v2'
PRM_CKPT_FALLBACK = '/root/autodl-tmp/checkpoint/checkpoint-1169'
PROMPT_TEMPLATE_PATH = '/root/workspace/qwen-dianjin/DianJin-PRM/src/templates/rollout_prompt.txt'

TEMPERATURE = 2.0
MAX_SAMPLES = None
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
MAX_SEQ_LENGTH = 4096

RANKING_MODES = ['trajectory', 'step_mean', 'fusion_static', 'fusion_weighted']


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    return line


def load_prompt_template():
    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def load_pre_samples(max_samples=None):
    jsonl_files = sorted(glob.glob(os.path.join(PRE_DIR, '*.jsonl')))
    log(f"发现 {len(jsonl_files)} 个 pre/ 文件...")
    samples = []
    for i, fpath in enumerate(jsonl_files):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
                if max_samples is not None and len(samples) >= max_samples:
                    break
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        raise ValueError(f"pre/ 目录下没有可用的样本: {PRE_DIR}")
    log(f"文件读取完成: 共 {len(samples)} 个问题样本")
    total_passes = sum(len(s['answer']) for s in samples)
    log(f"预计 PRM 推理次数: {total_passes} 次")
    return samples


def format_knowledge_items(knowledge_items):
    return [f'{k}: {v}' for k, v in knowledge_items.items() if v is not None]


def build_prm_text(question, knowledge_items, steps_dict, final_answer):
    step_values = [v for v in steps_dict.values() if v is not None]
    trajectory = '<extra_0>'.join(step_values) + '<extra_0>'
    ki_items = format_knowledge_items(knowledge_items)
    return (
        '##Question\n' + question +
        '\n\n##Knowledge\n' + '\n'.join(ki_items) +
        '\n\n##Thinking Trajectory\n' + trajectory +
        '\n\n##Final Answer\n' + final_answer + '<extra_1>'
    )


def build_dpo_answer(steps_dict, final_answer):
    step_values = [v for v in steps_dict.values() if v is not None]
    thought = '\n\n'.join(step_values)
    return (
        f"<|begin_of_thought|>\n{thought}\n<|end_of_thought|>\n"
        f"<|begin_of_solution|>\n{final_answer}\n<|end_of_solution|>"
    )


def build_dpo_prompt(prompt_template, question, knowledge_items):
    ki_items = format_knowledge_items(knowledge_items)
    knowledge_text = '\n'.join(ki_items)
    full_question = f"{question}\n\n参考知识：\n{knowledge_text}"
    return prompt_template.format(question=full_question)


def score_answer(model, tokenizer, text, sep0_id, sep1_id, temperature=TEMPERATURE, device='cuda'):
    """同时提取步骤级(<extra_0>)与轨迹级(<extra_1>)分数。"""
    enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits.float()

    # 步骤级：每个 <extra_0> 位置
    step_positions = (input_ids[0] == sep0_id).nonzero(as_tuple=True)[0]
    step_probs = [
        round(torch.nn.functional.softmax(logits[0, p] / temperature, dim=-1)[1].item(), 6)
        for p in step_positions
    ]

    # 轨迹级：末尾 <extra_1> 位置
    final_positions = (input_ids[0] == sep1_id).nonzero(as_tuple=True)[0]
    if len(final_positions) > 0:
        pos = final_positions[-1].item()
    else:
        pos = input_ids.shape[1] - 1
    prob = torch.nn.functional.softmax(logits[0, pos] / temperature, dim=-1)

    return {
        'step_probs': step_probs,
        'traj_prob': round(prob[1].item(), 6),
    }


def compute_combined(step_probs, traj_prob, mode, alpha, beta):
    """根据排序模式计算综合得分。"""
    if not step_probs:
        step_mean = 0.0
    else:
        step_mean = sum(step_probs) / len(step_probs)
    if mode == 'trajectory':
        return traj_prob
    elif mode == 'step_mean':
        return step_mean
    elif mode == 'fusion_static':
        return 0.5 * step_mean + 0.5 * traj_prob
    elif mode == 'fusion_weighted':
        return alpha * step_mean + beta * traj_prob
    else:
        raise ValueError(f"未知排序模式: {mode}")


def detect_prm_ckpt():
    if os.path.isdir(PRM_CKPT_BASE):
        import re
        ckpt_dirs = glob.glob(os.path.join(PRM_CKPT_BASE, 'checkpoint-*'))
        if ckpt_dirs:
            ckpt_dirs.sort(key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1)))
            return ckpt_dirs[-1]
    return PRM_CKPT_FALLBACK


def load_prm_model(prm_ckpt_path):
    config = Qwen3PRMConfig.from_pretrained(CONFIG_PATH)
    model = Qwen3ForProcessRewardModel(config=config)
    pretrained = AutoModel.from_pretrained(BASE_MODEL_PATH)
    model.model.load_state_dict(pretrained.state_dict(), strict=True)
    del pretrained
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(model, prm_ckpt_path)
    model.eval()
    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_0>', '<extra_1>']})
    sep0_id = tokenizer.encode('<extra_0>', add_special_tokens=False)[0]
    sep1_id = tokenizer.encode('<extra_1>', add_special_tokens=False)[0]
    return model, tokenizer, sep0_id, sep1_id


def score_and_cache(cache_path, max_samples):
    """阶段1：PRM 打分并缓存原始分数。"""
    prompt_template = load_prompt_template()
    samples = load_pre_samples(max_samples)
    prm_ckpt_path = detect_prm_ckpt()
    log(f"PRM 检查点: {prm_ckpt_path}")
    model, tokenizer, sep0_id, sep1_id = load_prm_model(prm_ckpt_path)

    total_samples = len(samples)
    total_inferences = sum(len(s['answer']) for s in samples)
    log(f"开始 PRM 打分: {total_samples} 个问题, 共 {total_inferences} 次推理")

    t_start = time.time()
    score_times = []
    cache_lines = []

    for idx, sample in enumerate(samples):
        question = sample['question']
        knowledge_items = sample.get('knowledge_items', {})
        answers = sample['answer']
        dpo_prompt = build_dpo_prompt(prompt_template, question, knowledge_items)

        candidates = []
        for ans in answers:
            t_infer = time.time()
            steps = ans['steps']
            final_answer = ans['final_answer']
            prm_text = build_prm_text(question, knowledge_items, steps, final_answer)
            score = score_answer(model, tokenizer, prm_text, sep0_id, sep1_id)
            score_times.append(time.time() - t_infer)
            candidates.append({
                'dpo_text': build_dpo_answer(steps, final_answer),
                'step_probs': score['step_probs'],
                'traj_prob': score['traj_prob'],
            })

        cache_lines.append({
            'q_idx': idx,
            'prompt': dpo_prompt,
            'candidates': candidates,
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == total_samples:
            elapsed = time.time() - t_start
            avg_per_sample = elapsed / (idx + 1)
            remaining = avg_per_sample * (total_samples - idx - 1)
            avg_infer = sum(score_times) / len(score_times)
            log(f"  [{idx+1}/{total_samples}] 平均推理 {avg_infer:.2f}s/次 | "
                f"耗时 {elapsed/60:.1f}min | 剩余 ~{remaining/60:.0f}min")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        for line in cache_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    t_total = time.time() - t_start
    log(f"打分完成! 耗时 {t_total/60:.1f} 分钟, 平均 {sum(score_times)/len(score_times):.2f}s/次")
    log(f"缓存已保存: {cache_path} ({len(cache_lines)} 个样本)")


def generate_from_cache(cache_path, ranking_mode, alpha, beta, output_dir):
    """阶段2：从缓存生成指定排序模式的数据集。"""
    samples = []
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    dpo_pairs = []
    for sample in samples:
        prompt = sample['prompt']
        candidates = sample['candidates']

        scored = []
        for c in candidates:
            combined = compute_combined(c['step_probs'], c['traj_prob'], ranking_mode, alpha, beta)
            scored.append({
                'combined': round(combined, 6),
                'step_mean': round(sum(c['step_probs']) / len(c['step_probs']), 6) if c['step_probs'] else 0.0,
                'traj_prob': c['traj_prob'],
                'step_probs': c['step_probs'],
                'dpo_text': c['dpo_text'],
            })

        scored.sort(key=lambda x: x['combined'], reverse=True)

        if len(scored) >= 2:
            margin = scored[0]['combined'] - scored[-1]['combined']
            dpo_pairs.append({
                'prompt': prompt,
                'chosen': scored[0]['dpo_text'],
                'rejected': scored[-1]['dpo_text'],
                'metadata': {
                    'chosen_score': scored[0]['combined'],
                    'rejected_score': scored[-1]['combined'],
                    'margin': round(margin, 6),
                    'num_candidates': len(scored),
                    'all_scores': [a['combined'] for a in scored],
                    'chosen_step_probs': scored[0]['step_probs'],
                    'rejected_step_probs': scored[-1]['step_probs'],
                    'chosen_traj_prob': scored[0]['traj_prob'],
                    'rejected_traj_prob': scored[-1]['traj_prob'],
                }
            })

    random.seed(SEED)
    random.shuffle(dpo_pairs)

    n = len(dpo_pairs)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    splits = {
        'train': dpo_pairs[:n_train],
        'validate': dpo_pairs[n_train:n_train + n_val],
        'test': dpo_pairs[n_train + n_val:],
    }

    for name, data in splits.items():
        out_dir = os.path.join(output_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        filename = f'dpo_{"val" if name == "validate" else name}.jsonl'
        path = os.path.join(out_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        avg_chosen = sum(p['metadata']['chosen_score'] for p in data) / len(data) if data else 0
        avg_rejected = sum(p['metadata']['rejected_score'] for p in data) / len(data) if data else 0
        log(f"  {name}: {path} ({len(data)} 条, {len(data)/n:.1%}), "
            f"avg chosen={avg_chosen:.4f}, rejected={avg_rejected:.4f}")

    log(f"排序模式 [{ranking_mode}] 数据生成完成: {output_dir} (共 {n} 对)")


def main():
    parser = argparse.ArgumentParser(description='构建 DPO 数据集（步骤级+轨迹级评分）')
    parser.add_argument('--max_samples', type=int, default=MAX_SAMPLES,
                        help='阶段1最多处理的样本数（测试用）')
    parser.add_argument('--cache_path', type=str, default=os.path.join(PRE_DIR, 'scored_cache.jsonl'),
                        help='原始分数缓存路径')
    parser.add_argument('--from_cache', action='store_true',
                        help='阶段2：从缓存生成数据集（跳过打分）')
    parser.add_argument('--ranking_mode', type=str, default='fusion_weighted', choices=RANKING_MODES,
                        help='排序模式（阶段2）')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='fusion_weighted 的步骤级权重')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='fusion_weighted 的轨迹级权重')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='阶段2输出目录（默认 rankings/<mode>）')
    args = parser.parse_args()

    if args.from_cache:
        if not os.path.isfile(args.cache_path):
            raise FileNotFoundError(f"缓存文件不存在: {args.cache_path}")
        output_dir = args.output_dir or os.path.join(RANKINGS_DIR, args.ranking_mode)
        log("=" * 60)
        log(f"从缓存生成数据集: mode={args.ranking_mode}, alpha={args.alpha}, beta={args.beta}")
        log(f"  缓存: {args.cache_path}")
        log(f"  输出: {output_dir}")
        log("=" * 60)
        generate_from_cache(args.cache_path, args.ranking_mode, args.alpha, args.beta, output_dir)
    else:
        log("=" * 60)
        log(f"DPO 打分 + 缓存: cache={args.cache_path}, max_samples={args.max_samples}")
        log("=" * 60)
        score_and_cache(args.cache_path, args.max_samples)


if __name__ == '__main__':
    main()
