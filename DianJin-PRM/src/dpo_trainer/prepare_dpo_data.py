"""
DPO 偏好数据准备脚本
====================
根据 实验方案.md 阶段二设计：
  1. 从 CFLUE 金融问答数据集提取问题
  2. 使用 rollout prompt template 格式化提示
  3. 基于 ground truth 答案构造 chosen（正确推理）和 rejected（错误推理）响应
  4. 保存为 TRL DPO 标准格式（prompt/chosen/rejected）

注意：本脚本不修改原始数据（requirement #8）
"""

import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional

# ============ 配置 ============

SYSTEM_PROMPT = "你是一位专业的金融领域分析师，请通过逐步推理来回答问题。"

ANSWER_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# ============ 推理轨迹生成 ============


def parse_choices(choices_data) -> Dict[str, str]:
    """解析 choices，支持 dict 或 str 格式"""
    if isinstance(choices_data, dict):
        return choices_data
    if isinstance(choices_data, str):
        try:
            return json.loads(choices_data.replace("'", '"'))
        except (json.JSONDecodeError, AttributeError):
            pass
    return {}


def generate_correct_reasoning(question: str, choices: Dict[str, str],
                               answer: str, analysis: str) -> str:
    """
    基于正确的 analysis 生成正确的推理轨迹。
    格式：Thought（逐步推理）+ Solution（正确答案）
    """
    # 构建推理步骤
    reasoning_steps = []
    choices_text = "；".join([f"{k}: {v}" for k, v in choices.items()])

    # 步骤1：理解问题
    reasoning_steps.append(f"首先，我需要理解这个问题。题目询问的是：{question[:100]}...")

    # 步骤2：分析选项
    reasoning_steps.append(f"分析各个选项：{choices_text}")

    # 步骤3：使用 analysis 构建推理过程
    if analysis:
        reasoning_steps.append(f"逐步分析：{analysis}")
    else:
        # 如果没有 analysis，构造一个通用推理
        reasoning_steps.append(f"经过分析，正确答案是 {answer}。")

    # 步骤4：得出结论
    correct_option_text = choices.get(answer, answer)
    reasoning_steps.append(
        f"综上所述，正确答案是 {answer} ({correct_option_text})。"
    )

    # 组装为 Thought + Solution 格式
    thought = "\n\n".join(reasoning_steps)
    solution = f"正确答案是 {answer}。"

    response = (
        f"<|begin_of_thought|>\n{thought}\n<|end_of_thought|>\n"
        f"<|begin_of_solution|>\n{solution}\n<|end_of_solution|>"
    )
    return response


def generate_wrong_reasoning(question: str, choices: Dict[str, str],
                             answer: str, analysis: str) -> str:
    """
    生成错误的推理轨迹。
    策略：选择一个错误答案，构造看似合理但包含错误的推理过程。
    """
    # 选择一个错误答案
    wrong_answers = [k for k in choices.keys() if k != answer]
    if not wrong_answers:
        # 如果没有其他选项，使用一个不同的答案
        wrong_answer = f"非{answer}"
    else:
        wrong_answer = random.choice(wrong_answers)

    choices_text = "；".join([f"{k}: {v}" for k, v in choices.items()])
    wrong_option_text = choices.get(wrong_answer, wrong_answer)
    correct_option_text = choices.get(answer, answer)

    reasoning_steps = []

    # 步骤1：理解问题（可以正确）
    reasoning_steps.append(f"首先分析问题：{question[:100]}...")

    # 步骤2：错误地分析选项
    # 构造一个看似合理但指向错误答案的推理
    if analysis:
        # 从正确的 analysis 中"错误解读"
        corrupted_analysis = _corrupt_analysis(analysis, answer, wrong_answer)
        reasoning_steps.append(f"考虑各选项的特征：{corrupted_analysis}")
    else:
        reasoning_steps.append(
            f"分析认为，选项 {wrong_answer} ({wrong_option_text}) 是最符合题意的。"
        )

    # 步骤3：错误的计算/推理
    reasoning_steps.append(
        f"因此，我判断正确答案应该是 {wrong_answer} ({wrong_option_text})，"
        f"而非 {answer} ({correct_option_text})。"
    )

    # 组装
    thought = "\n\n".join(reasoning_steps)
    solution = f"正确答案是 {wrong_answer}。"

    response = (
        f"<|begin_of_thought|>\n{thought}\n<|end_of_thought|>\n"
        f"<|begin_of_solution|>\n{solution}\n<|end_of_solution|>"
    )
    return response


def _corrupt_analysis(analysis: str, correct_answer: str,
                      wrong_answer: str) -> str:
    """对正确的 analysis 进行局部篡改以支持错误答案"""
    sentences = re.split(r'[。；!?]', analysis)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= 2:
        # 篡改结论句
        for i, sent in enumerate(sentences):
            if correct_answer in sent:
                sentences[i] = sent.replace(correct_answer, wrong_answer)
                break
    else:
        # 简单替换
        analysis = analysis.replace(correct_answer, wrong_answer)

    return "。".join(sentences) + "。"


# ============ 主流程 ============


def prepare_dpo_data(args):
    """主数据准备流程"""
    random.seed(42)

    # 1. 加载原始 CFLUE 数据（不修改）
    print(f"[1/5] 加载原始数据: {args.input_path}")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"      共 {len(raw_data)} 条原始样本")

    # 2. 加载 prompt template
    print(f"[2/5] 加载提示模板: {args.prompt_template}")
    with open(args.prompt_template, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # 3. 过滤和抽样
    print(f"[3/5] 过滤和抽样数据")
    # 过滤掉没有 choices 或 answer 的样本
    valid_data = []
    for item in raw_data:
        choices = item.get('choices', {})
        answer = item.get('answer', '')
        if choices and answer:
            # 确保 answer 是选项字母
            parsed_choices = parse_choices(choices)
            if any(k == answer for k in parsed_choices.keys()):
                valid_data.append(item)

    print(f"      有效样本: {len(valid_data)}/{len(raw_data)}")

    # 按比例切分 train/test
    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * (1 - args.test_ratio))
    train_items = valid_data[:split_idx]
    test_items = valid_data[split_idx:]

    print(f"      训练集: {len(train_items)}, 测试集: {len(test_items)}")

    # 4. 生成 DPO 格式数据
    print(f"[4/5] 生成 DPO 偏好数据")

    def process_items(items: List[Dict]) -> List[Dict]:
        dpo_data = []
        for i, item in enumerate(items):
            question = item['question']
            choices = parse_choices(item['choices'])
            answer = item['answer']
            analysis = item.get('analysis', '')

            # 格式化 prompt
            question_with_choices = question
            if choices:
                choices_str = "\n".join(
                    [f"{k}. {v}" for k, v in choices.items()]
                )
                question_with_choices = f"{question}\n\n选项：\n{choices_str}"

            prompt = prompt_template.format(question=question_with_choices)

            # 生成 chosen（正确推理）
            chosen = generate_correct_reasoning(
                question, choices, answer, analysis
            )

            # 生成 rejected（错误推理）
            rejected = generate_wrong_reasoning(
                question, choices, answer, analysis
            )

            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "data_source": "cflue",
                    "subject": item.get('科目', ''),
                    "chapter": item.get('章节', ''),
                    "correct_answer": answer,
                }
            })

            if (i + 1) % 2000 == 0:
                print(f"        已处理 {i + 1}/{len(items)} 条...")

        return dpo_data

    train_dpo = train_items
    test_dpo = test_items

    # 限制规模（可选）
    if args.max_train_samples and len(train_dpo) > args.max_train_samples:
        train_dpo = train_dpo[:args.max_train_samples]
    if args.max_test_samples and len(test_dpo) > args.max_test_samples:
        test_dpo = test_dpo[:args.max_test_samples]

    train_dpo = process_items(train_dpo) if train_dpo else []
    test_dpo = process_items(test_dpo) if test_dpo else []

    # 5. 保存
    print(f"[5/5] 保存 DPO 数据")
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "dpo_train.jsonl")
    test_path = os.path.join(args.output_dir, "dpo_test.jsonl")

    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_dpo:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_dpo:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"      训练数据: {train_path} ({len(train_dpo)} 条)")
    print(f"      测试数据: {test_path} ({len(test_dpo)} 条)")
    print("      DPO 数据准备完成!")


# ============ 入口 ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备 DPO 偏好数据")
    parser.add_argument(
        "--input_path",
        default="/root/autodl-tmp/data/tongyi_dianjin/CFLUE/knowledge/train.json",
        help="原始 CFLUE 数据路径"
    )
    parser.add_argument(
        "--prompt_template",
        default="/root/workspace/qwen-dianjin/DianJin-PRM/src/templates/rollout_prompt.txt",
        help="rollout prompt template 路径"
    )
    parser.add_argument(
        "--output_dir",
        default="/root/workspace/qwen-dianjin/DianJin-PRM/src/dpo_trainer/data",
        help="DPO 数据输出目录"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="测试集比例"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="最大训练样本数（用于快速测试）"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=200,
        help="最大测试样本数"
    )
    args = parser.parse_args()
    prepare_dpo_data(args)
