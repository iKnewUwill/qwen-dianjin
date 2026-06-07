"""
使用 PRM 模型对 pre/ 样本打分，构建 DPO 强化学习数据集。

流程:
1. 读取 data_dpo/pre/ 下前 20 个 JSONL 文件
2. 对每个问题的多个回答候选，用 PRM 模型打分
3. 选择最高分作为 chosen，最低分作为 rejected
4. 按 7:2:1 随机拆分到 train/validate/test
"""

import os
import sys
import json
import glob
import random
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
PRM_CKPT_PATH = '/root/autodl-tmp/checkpoint/checkpoint-1169'
PROMPT_TEMPLATE_PATH = '/root/workspace/qwen-dianjin/DianJin-PRM/src/templates/rollout_prompt.txt'

NUM_FILES = 400
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
MAX_SEQ_LENGTH = 4096


def load_prompt_template():
    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def load_pre_samples(num_files):
    jsonl_files = sorted(glob.glob(os.path.join(PRE_DIR, '*.jsonl')))[:num_files]
    print(f"读取 {len(jsonl_files)} 个文件...")
    samples = []
    for fpath in jsonl_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    print(f"共 {len(samples)} 个问题样本")
    return samples


def build_prm_text(question, knowledge_items, steps_dict, final_answer):
    step_values = [v for v in steps_dict.values() if v is not None]
    trajectory = '<extra_0>'.join(step_values) + '<extra_0>'
    ki_items = [f'{k}: {v}' for k, v in knowledge_items.items()]
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
    ki_items = [f'{k}: {v}' for k, v in knowledge_items.items()]
    knowledge_text = '\n'.join(ki_items)
    full_question = f"{question}\n\n参考知识：\n{knowledge_text}"
    return prompt_template.format(question=full_question)


def score_answer(model, tokenizer, text, sep1_id, device='cuda'):
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
    prob = torch.nn.functional.softmax(logits[0, pos], dim=-1)

    return {
        'prob_1': round(prob[1].item(), 6),
        'prob_0': round(prob[0].item(), 6),
    }


def main():
    random.seed(SEED)

    print("[1/5] 加载 prompt 模板...")
    prompt_template = load_prompt_template()

    print("[2/5] 加载 pre/ 样本...")
    samples = load_pre_samples(NUM_FILES)

    print("[3/5] 加载 PRM 模型...")
    config = Qwen3PRMConfig.from_pretrained(CONFIG_PATH)
    model = Qwen3ForProcessRewardModel(config=config)
    pretrained = AutoModel.from_pretrained(BASE_MODEL_PATH)
    model.model.load_state_dict(pretrained.state_dict(), strict=True)
    del pretrained
    torch.cuda.empty_cache()

    model = PeftModel.from_pretrained(model, PRM_CKPT_PATH)
    model.eval()
    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_0>', '<extra_1>']})
    sep1_id = tokenizer.encode('<extra_1>', add_special_tokens=False)[0]
    print("   模型加载完成")

    print(f"[4/5] 对 {len(samples)} 个样本打分...")
    dpo_pairs = []

    for idx, sample in enumerate(samples):
        question = sample['question']
        knowledge_items = sample.get('knowledge_items', {})
        answers = sample['answer']

        dpo_prompt = build_dpo_prompt(prompt_template, question, knowledge_items)

        scored = []
        for ans in answers:
            steps = ans['steps']
            final_answer = ans['final_answer']
            prm_text = build_prm_text(question, knowledge_items, steps, final_answer)
            score = score_answer(model, tokenizer, prm_text, sep1_id)
            scored.append({
                'prob_1': score['prob_1'],
                'dpo_text': build_dpo_answer(steps, final_answer),
            })

        scored.sort(key=lambda x: x['prob_1'], reverse=True)

        if len(scored) >= 2:
            dpo_pairs.append({
                'prompt': dpo_prompt,
                'chosen': scored[0]['dpo_text'],
                'rejected': scored[-1]['dpo_text'],
                'metadata': {
                    'chosen_score': scored[0]['prob_1'],
                    'rejected_score': scored[-1]['prob_1'],
                    'num_candidates': len(scored),
                    'all_scores': [a['prob_1'] for a in scored],
                }
            })

        if (idx + 1) % 20 == 0:
            print(f"   已处理 {idx + 1}/{len(samples)} 个样本...")

    print(f"   生成 {len(dpo_pairs)} 条 DPO 偏好数据")

    print("[5/5] 拆分和保存数据...")
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
        print(f"  {name}: {path} ({len(data)} 条, {len(data)/n:.1%})")

    print("\nDPO 数据集构建完成!")


if __name__ == '__main__':
    main()
