"""
使用 PRM 模型对 pre/ 样本打分，构建 DPO 强化学习数据集。

流程:
1. 读取 data_dpo/pre/ 下全部 JSONL 文件（支持单个合并文件 dpo_pre.jsonl）
2. 对每个问题的多个回答候选，用 PRM 模型打分
3. 选择最高分作为 chosen，最低分作为 rejected
4. 按 7:2:1 随机拆分到 train/validate/test
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
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validate')
TEST_DIR = os.path.join(BASE_DIR, 'test')

CONFIG_PATH = '/root/workspace/qwen-dianjin/DianJin-PRM/src/model/config.json'
BASE_MODEL_PATH = '/root/autodl-tmp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
# PRM 模型检查点路径 - 使用最新训练的PRM模型
# 如果 checkpoint_prm_v2 存在则使用，否则回退到旧版
PRM_CKPT_BASE = '/root/autodl-tmp/checkpoint_prm_v2'
PRM_CKPT_FALLBACK = '/root/autodl-tmp/checkpoint/checkpoint-1169'
PROMPT_TEMPLATE_PATH = '/root/workspace/qwen-dianjin/DianJin-PRM/src/templates/rollout_prompt.txt'

# 温度参数 - 控制PRM打分的平滑程度
# 温度越高，概率分布越平滑，chosen/rejected之间的margin会更小
# 温度=1.0 为标准softmax，温度>1.0 增加探索性
TEMPERATURE = 2.0

MAX_SAMPLES = None  # None 表示读取全部样本；设为整数可限制样本数量（便于快速测试）
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
MAX_SEQ_LENGTH = 4096


def load_prompt_template():
    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    return line


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
        if (i + 1) % 100 == 0:
            log(f"  已读取 {i+1}/{len(jsonl_files)} 个文件, 当前累计 {len(samples)} 个样本")
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        raise ValueError(f"pre/ 目录下没有可用的样本: {PRE_DIR}")
    log(f"文件读取完成: 共 {len(samples)} 个问题样本, 每个样本 {len(samples[0]['answer'])} 个候选回答")
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


def score_answer(model, tokenizer, text, sep1_id, temperature=TEMPERATURE, device='cuda'):
    enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits.float()

    mask = input_ids[0] == sep1_id
    positions = mask.nonzero(as_tuple=True)[0]
    pos = positions[-1].item() if len(positions) > 0 else (input_ids.shape[1] - 1)
    prob = torch.nn.functional.softmax(logits[0, pos] / temperature, dim=-1)

    return {
        'prob_1': round(prob[1].item(), 6),
        'prob_0': round(prob[0].item(), 6),
    }


def main():
    parser = argparse.ArgumentParser(description='构建 DPO 数据集')
    parser.add_argument('--max_samples', type=int, default=MAX_SAMPLES,
                        help='最多处理的样本数，默认 None 表示全部')
    args = parser.parse_args()

    random.seed(SEED)
    t_start = time.time()

    # Auto-detect PRM checkpoint: use v2 if available, otherwise fallback
    if os.path.isdir(PRM_CKPT_BASE):
        import glob as _glob
        import re
        ckpt_dirs = _glob.glob(os.path.join(PRM_CKPT_BASE, 'checkpoint-*'))
        if ckpt_dirs:
            ckpt_dirs.sort(key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1)))
            prm_ckpt_path = ckpt_dirs[-1]
            log(f"  使用新版PRM检查点: {prm_ckpt_path}")
        else:
            prm_ckpt_path = PRM_CKPT_FALLBACK
            log(f"  新版PRM检查点目录为空，回退到旧版: {prm_ckpt_path}")
    else:
        prm_ckpt_path = PRM_CKPT_FALLBACK
        log(f"  新版PRM检查点不存在，使用旧版: {prm_ckpt_path}")

    log("=" * 60)
    log("DPO 数据集构建开始")
    log(f"  配置文件: {CONFIG_PATH}")
    log(f"  PRM 模型: {prm_ckpt_path}")
    log(f"  PRM 温度: {TEMPERATURE}")
    log(f"  最大样本数: {args.max_samples if args.max_samples is not None else '全部'}")
    log(f"  随机种子: {SEED}")
    log(f"  数据拆分比例: train={TRAIN_RATIO} / val={VAL_RATIO} / test={1-TRAIN_RATIO-VAL_RATIO}")
    log("=" * 60)

    log("[1/5] 加载 prompt 模板...")
    prompt_template = load_prompt_template()
    log("  prompt 模板加载完成")

    log("[2/5] 加载 pre/ 样本...")
    samples = load_pre_samples(args.max_samples)

    log("[3/5] 加载 PRM 模型...")
    log(f"  加载模型配置: {CONFIG_PATH}")
    config = Qwen3PRMConfig.from_pretrained(CONFIG_PATH)
    model = Qwen3ForProcessRewardModel(config=config)
    log(f"  加载基座模型: {BASE_MODEL_PATH}")
    pretrained = AutoModel.from_pretrained(BASE_MODEL_PATH)
    model.model.load_state_dict(pretrained.state_dict(), strict=True)
    del pretrained
    torch.cuda.empty_cache()

    log(f"  加载 LoRA adapter: {prm_ckpt_path}")
    model = PeftModel.from_pretrained(model, prm_ckpt_path)
    model.eval()
    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_0>', '<extra_1>']})
    sep1_id = tokenizer.encode('<extra_1>', add_special_tokens=False)[0]

    # GPU memory info
    gpu_free = torch.cuda.mem_get_info()[0] / (1024**3)
    log(f"  模型加载完成, GPU 剩余显存: {gpu_free:.1f} GB")

    total_samples = len(samples)
    total_inferences = sum(len(s['answer']) for s in samples)
    log(f"[4/5] 开始 PRM 打分: {total_samples} 个问题, 共 {total_inferences} 次推理")
    log(f"  预计单样本耗时 ~2-5s, 总计约 {total_inferences*3//60}-{total_inferences*5//60} 分钟")
    log("-" * 60)

    dpo_pairs = []
    total_scored = 0
    score_times = []  # track per-inference timing

    for idx, sample in enumerate(samples):
        t_sample_start = time.time()
        question = sample['question']
        knowledge_items = sample.get('knowledge_items', {})
        answers = sample['answer']

        dpo_prompt = build_dpo_prompt(prompt_template, question, knowledge_items)

        scored = []
        for ans_idx, ans in enumerate(answers):
            t_infer_start = time.time()
            steps = ans['steps']
            final_answer = ans['final_answer']
            prm_text = build_prm_text(question, knowledge_items, steps, final_answer)
            score = score_answer(model, tokenizer, prm_text, sep1_id)
            t_infer = time.time() - t_infer_start
            score_times.append(t_infer)
            total_scored += 1
            scored.append({
                'prob_1': score['prob_1'],
                'dpo_text': build_dpo_answer(steps, final_answer),
            })

        scored.sort(key=lambda x: x['prob_1'], reverse=True)

        if len(scored) >= 2:
            margin = scored[0]['prob_1'] - scored[-1]['prob_1']
            dpo_pairs.append({
                'prompt': dpo_prompt,
                'chosen': scored[0]['dpo_text'],
                'rejected': scored[-1]['dpo_text'],
                'metadata': {
                    'chosen_score': scored[0]['prob_1'],
                    'rejected_score': scored[-1]['prob_1'],
                    'margin': round(margin, 6),
                    'num_candidates': len(scored),
                    'all_scores': [a['prob_1'] for a in scored],
                }
            })

        # Per-sample progress logging
        t_sample = time.time() - t_sample_start
        avg_infer = sum(score_times[-len(answers):]) / len(answers)

        if (idx + 1) % 1 == 0:  # Log every sample
            elapsed = time.time() - t_start
            avg_per_sample = elapsed / (idx + 1)
            remaining = avg_per_sample * (total_samples - idx - 1)
            scores_str = "/".join([f"{a['prob_1']:.4f}" for a in scored])
            log(
                f"  [{idx+1}/{total_samples}] "
                f"scores=[{scores_str}] "
                f"margin={scored[0]['prob_1']-scored[-1]['prob_1']:.4f} "
                f"| {t_sample:.1f}s "
                f"| 耗时 {elapsed/60:.1f}min "
                f"| 剩余 ~{remaining/60:.0f}min "
                f"| 速率 {avg_per_sample:.1f}s/样本"
            )

        # Summary every 50 samples
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            avg_infer_all = sum(score_times) / len(score_times)
            pairs_so_far = len(dpo_pairs)
            avg_margin = sum(p['metadata']['margin'] for p in dpo_pairs[-50:]) / min(50, len(dpo_pairs))
            log(f"  --- 汇总 [{idx+1}/{total_samples}]: "
                f"已生成 {pairs_so_far} 条偏好对, "
                f"近50条平均margin={avg_margin:.4f}, "
                f"平均推理={avg_infer_all:.2f}s/次, "
                f"总耗时={elapsed/60:.1f}min")

    t_scoring = time.time() - t_start
    log("-" * 60)
    log(f"  打分完成! 耗时 {t_scoring/60:.1f} 分钟")
    log(f"  总推理次数: {total_scored}")
    log(f"  平均推理速度: {sum(score_times)/len(score_times):.2f}s/次")
    log(f"  生成 DPO 偏好对: {len(dpo_pairs)} 条")

    log("[5/5] 拆分和保存数据...")
    random.shuffle(dpo_pairs)

    n = len(dpo_pairs)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    splits = {
        'train': (dpo_pairs[:n_train], TRAIN_DIR, 'dpo_train.jsonl'),
        'validate': (dpo_pairs[n_train:n_train + n_val], VAL_DIR, 'dpo_val.jsonl'),
        'test': (dpo_pairs[n_train + n_val:], TEST_DIR, 'dpo_test.jsonl'),
    }

    for name, (data, out_dir, filename) in splits.items():
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        avg_chosen = sum(p['metadata']['chosen_score'] for p in data) / len(data)
        avg_rejected = sum(p['metadata']['rejected_score'] for p in data) / len(data)
        log(f"  {name}: {path} ({len(data)} 条, {len(data)/n:.1%})")
        log(f"         avg chosen_score={avg_chosen:.4f}, avg rejected_score={avg_rejected:.4f}")

    t_total = time.time() - t_start
    log("=" * 60)
    log(f"DPO 数据集构建完成! 总耗时 {t_total/60:.1f} 分钟")
    log(f"  偏好对总数: {len(dpo_pairs)}")
    log(f"  训练集: {n_train} / 验证集: {n_val} / 测试集: {n - n_train - n_val}")
    log("=" * 60)


if __name__ == '__main__':
    main()
