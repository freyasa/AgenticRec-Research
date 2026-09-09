# Paper Index

| Paper | 动机 | 方法 | 什么有用 | Year | Venue | Link |
|---|---|---|---|---|---|---|
| MARS | 平铺式文本 memory 将短期行为与稳定偏好混合，且缺少明确的 memory 生命周期管理。 | 构建 Event → Preference → Profile 的层次化 belief-state memory；使用 LLM planner 调度 extract/boost/demote/merge/forget/synthesize 操作，冻结 LLM ranker 使用 profile 与近期事件进行推荐。 |  | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.14401) · [Notes](papers/MARS.md) |
| AMEM4Rec | Agentic 推荐系统过度依赖语义 memory，难以利用跨用户协同信号。 | 将 LLM 生成的行为模式 memory 转换为 SBERT 向量，构建全局 memory pool，通过 memory 演化与检索增强 LLM reranking。 |  | 2026.02 | arXiv | [Paper](https://arxiv.org/abs/2602.08837) · [Notes](papers/AMEM4Rec.md) |
| CoVeMem | 文本 memory 需要频繁重写，并且将 embedding 语言化会丢失细粒度协同几何信息。 | 使用 LightGCN 学习并冻结 user/item 状态作为向量 memory，通过 projector 转换为 LLM soft token，再使用语义对齐和排序损失训练 projector 与 LoRA reader。 |  | 2026.08 | arXiv | [Paper](https://arxiv.org/abs/2608.26895) · [Notes](papers/CoVeMem.md) |
| AgenticRec | 工具增强推理 trajectory 与推荐反馈存在错位，限制模型学习细粒度用户偏好。 | RTA 使用 GRPO 和 ranking reward 训练 Think–Act–Observation–Recommendation trajectory；PPR 从排序错误中挖掘 self-bootstrapped hard pair，并继续进行 GRPO 优化。 |  | 2026.03 | arXiv | [Paper](https://arxiv.org/abs/2603.21613) · [Notes](papers/AgenticRec.md) |
| RRCM | 固定 retrieval/RAG pipeline 无法根据不同推荐实例判断需要什么信息，可能造成无效 context 使用。 | 将 retrieval 视为 agent action，让 LLM 学习是否检索、选择 Collaborative Memory 或 Meta Memory；通过 SFT warmup 学习工具调用，再使用 GRPO 根据 ranking reward 优化 retrieval policy。 |  | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.07129) · [Notes](papers/RRCM.md) |

> `什么有用` 暂不维护，研究判断保留在详细笔记中，保证索引表易于维护。
