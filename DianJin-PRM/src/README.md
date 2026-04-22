# 目标
基于新的数据样式训练PRM模型

---

# 数据解释
包含多个样本，每个样本下有多个子样本。样例数据见src/data/task_20260420_191514_7e437cee.jsonl
每个子样本包含：
```json
{
    question: str
    knowledge_items: Dict[str, Optional[float]] = Field(default_factory=dict)
    steps: Dict[str, str] = Field(default_factory=dict)
    final_answer: str
    step_labels: List[int]
    trajectory_label: int
}
```