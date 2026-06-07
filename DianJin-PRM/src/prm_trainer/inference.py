import os, glob, json, torch, sys
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/autodl-tmp/huggingface'
sys.path.insert(0, '/root/workspace/qwen-dianjin/DianJin-PRM/src')

from transformers import AutoTokenizer, AutoModel
from model.fin_prm import Qwen3ForProcessRewardModel
from model.fin_config import Qwen3PRMConfig
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict


def load_jsonl_dataset(data_dir):
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, '*.jsonl')))
    datasets = [load_dataset('json', data_files=f, split='train') for f in jsonl_files]
    return concatenate_datasets(datasets)


def build_text(sample, tokenizer, max_tokens=4096):
    steps = sample['steps']
    step_contents = [v for v in steps.values() if v is not None]
    trace = '<extra_0>'.join(step_contents) + '<extra_0>'
    final_part = '\n\n##Final Answer\n' + sample['final_answer'] + '<extra_1>'

    ki = sample.get('knowledge_items', {})
    ki_items = [f'{k}: {v}' for k, v in ki.items()]

    def make_text(kt):
        header = '##Question\n' + sample['question'] + '\n\n##Knowledge\n' + kt
        body = '\n\n##Thinking Trajectory\n' + trace + final_part
        return header + body
    
    full_text = make_text('\n'.join(ki_items))
    full_len = len(tokenizer.encode(full_text))
    if full_len <= max_tokens:
        return full_text
    
    fixed_len = len(tokenizer.encode(make_text('')))
    budget = max_tokens - fixed_len - 20
    
    if budget <= 0:
        return full_text[:max_tokens * 4]
    
    lo, hi = 0, len(ki_items)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        kt = '\n'.join(ki_items[:mid])
        candidate = make_text(kt)
        if len(tokenizer.encode(candidate)) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    
    kt = '\n'.join(ki_items[:lo])
    return make_text(kt)


def score_sample(model, tokenizer, text, sep0_id, sep1_id, device='cuda', max_length=4096):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits.float()

    mask = (input_ids[0] == sep0_id) | (input_ids[0] == sep1_id)
    positions = mask.nonzero(as_tuple=True)[0].tolist()
    scores = []
    for pos in positions:
        prob = torch.nn.functional.softmax(logits[0, pos], dim=-1)
        scores.append({
            'position': pos,
            'logit_0': round(logits[0, pos, 0].item(), 4),
            'logit_1': round(logits[0, pos, 1].item(), 4),
            'prob_0': round(prob[0].item(), 6),
            'prob_1': round(prob[1].item(), 6),
        })
    return scores


def main():
    config_path = '/root/workspace/qwen-dianjin/DianJin-PRM/src/model/config.json'
    base_path = '/root/autodl-tmp/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
    ckpt_path = '/root/autodl-tmp/checkpoint/checkpoint-1169'

    config = Qwen3PRMConfig.from_pretrained(config_path)
    model = Qwen3ForProcessRewardModel(config=config)
    pretrained = AutoModel.from_pretrained(base_path)
    model.model.load_state_dict(pretrained.state_dict(), strict=True)
    del pretrained
    torch.cuda.empty_cache()

    model = PeftModel.from_pretrained(model, ckpt_path)
    model.eval()
    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_0>', '<extra_1>']})
    sep0_id = tokenizer.encode('<extra_0>', add_special_tokens=False)[0]
    sep1_id = tokenizer.encode('<extra_1>', add_special_tokens=False)[0]

    for split_name, data_dir in [
        ('validate', '/root/workspace/qwen-dianjin/DianJin-PRM/src/data/validate'),
        ('test', '/root/workspace/qwen-dianjin/DianJin-PRM/src/data/test'),
    ]:
        ds = load_jsonl_dataset(data_dir)
        total = len(ds)
        results = []
        step_acc = {'correct': 0, 'total': 0, 'prob_1_correct': 0, 'prob_1_wrong': 0}

        for i, sample in enumerate(ds):
            text = build_text(sample, tokenizer)
            scores = score_sample(model, tokenizer, text, sep0_id, sep1_id)

            step_labels = sample['step_labels']
            trajectory_label = sample['trajectory_label']

            step_results = []
            for j, s in enumerate(scores[:-1]):
                pred = 1 if s['prob_1'] >= 0.5 else 0
                true = step_labels[j] if j < len(step_labels) else None
                correct = 'correct' if (true is not None and pred == true) else ('wrong' if true is not None else 'N/A')
                step_results.append({
                    'step_key': f'Step {j+1}',
                    'true_label': true,
                    'pred_label': pred,
                    'prob_1': s['prob_1'],
                    'result': correct,
                })
                if true is not None:
                    step_acc['total'] += 1
                    if pred == true:
                        step_acc['correct'] += 1
                    if true == 1:
                        step_acc['prob_1_correct'] += s['prob_1']
                    else:
                        step_acc['prob_1_wrong'] += s['prob_1']

            final_score = scores[-1] if scores else None
            final_pred = 1 if (final_score and final_score['prob_1'] >= 0.5) else 0
            final_correct = 'correct' if final_pred == trajectory_label else 'wrong'

            results.append({
                'index': i,
                'question': sample['question'][:100] + '...',
                'step_labels': step_labels,
                'trajectory_label': trajectory_label,
                'step_scores': step_results,
                'final_score': {
                    'prob_1': final_score['prob_1'] if final_score else None,
                    'pred': final_pred,
                    'true': trajectory_label,
                    'result': final_correct,
                },
            })

            if (i + 1) % 100 == 0:
                print(f'  Processed {i+1}/{total} samples...')

        print(f'\n{split_name} results (total={total}):')
        print(f'  Step-level accuracy: {step_acc["correct"]}/{step_acc["total"]} = {step_acc["correct"]/step_acc["total"]:.4f}' if step_acc['total'] > 0 else '  N/A')
        if step_acc['prob_1_correct'] > 0:
            print(f'  Avg prob_1 for correct steps: {step_acc["prob_1_correct"]/max(1, sum(1 for r in results for s in r["step_scores"] if s["true_label"]==1)):.4f}')
        if step_acc['prob_1_wrong'] > 0:
            print(f'  Avg prob_1 for wrong steps: {step_acc["prob_1_wrong"]/max(1, sum(1 for r in results for s in r["step_scores"] if s["true_label"]==0)):.4f}')

        out_path = f'/root/autodl-tmp/inference_{split_name}.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'  Results saved to {out_path}')


if __name__ == '__main__':
    main()
