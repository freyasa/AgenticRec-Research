# Paper Index

| Paper | 动机 | 方法 | 什么有用 | Year | Venue | Link |
|---|---|---|---|---|---|---|
| MARS | 平铺式文本 memory 将短期行为与稳定偏好混合，并且没有memory生命周期 | 提出了分层的memory，分别包含最近历史记录，短期/长期兴趣。每次inference的时候会先通过Extract整理memory，同时根据memory进行推荐。![2026-09-10_ MARS](./papers/imgs/2026-09-10_MARS.png) |  | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.14401) · [Notes](papers/MARS.md) |
| AMEM4Rec | 现有的工作只制作了user自己的memory，没有考虑到协同的东西 | 创建了一个memory池。让llm能够使用当前user的memory去检索内存池中的shared memory。shared memory创建方法为生成所有的user memory，相似度最高的一对送入llm生成合并后的memory。![2026-09-10_ AMEM4Rec](./papers/imgs/2026-09-10_ AMEM4Rec.png) |  | 2026.02 | arXiv | [Paper](https://arxiv.org/abs/2602.08837) · [Notes](papers/AMEM4Rec.md) |
| CoVeMem | 如果像MemRec一样去做text memory是很浪费token的 | 使用embedding代替text去做memory：1. 首先用CF训练了一个embedding表征user和item；2. 把这个embedding对齐到LLM emb的宽度；3. 继续训练projector+LLM lora（使用了LLM listwise loss）![2026-09-10_CoVeMem](./papers/imgs/2026-09-10_CoVeMem.png) |  | 2026.08 | arXiv | [Paper](https://arxiv.org/abs/2608.26895) · [Notes](papers/CoVeMem.md) |
| AgenticRec | 不训练的LLM agent是没法自己根据情况调用工具的 | ReAct Agent: 使用GRPO去做两阶段训练：1. 在所有样本生成不同trajectory做GRPO；2. 在hard-neg sample上生成不同trajectory做GRPO![2026-09-10_AgenticRec](./papers/imgs/2026-09-10_AgenticRec.png) |  | 2026.03 | arXiv | [Paper](https://arxiv.org/abs/2603.21613) · [Notes](papers/AgenticRec.md) |
| RRCM | 固定 retrieval/RAG pipeline 无法根据不同推荐实例判断需要什么信息，可能造成无效 context 使用。 | 将 retrieval 视为 agent action，让 LLM 学习是否检索、选择 Collaborative Memory 或 Meta Memory；通过 SFT warmup 学习工具调用，再使用 GRPO 根据 ranking reward 优化 retrieval policy。 |  | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.07129) · [Notes](papers/RRCM.md) |

> `什么有用` 暂不维护，研究判断保留在详细笔记中，保证索引表易于维护。
