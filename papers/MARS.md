# MARS — Agentic Recommender System with Hierarchical Belief-State Memory

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2605.14401
- **Drive note:** https://docs.google.com/document/d/1QTHsldUIZ5ly8BCXxsjum1LDGz19hstkTptDroAwARk/edit

## Motivation

现有 memory-augmented LLM recommender 往往把短期行为、长期偏好和整体画像混在一段扁平文本里。这样会导致：

- 一次偶然点击容易污染长期画像；
- 每次更新都可能重写整段 memory；
- 缺少明确的强化、弱化、合并、遗忘等生命周期；
- LLM 很难区分“最近发生了什么”和“用户稳定喜欢什么”。

MARS 将推荐建模为部分可观测问题，用层次化 belief state 持续估计用户状态。

## Core Method

每个用户维护三层状态：

### 1. Event Memory

保存最近的原始行为事件，形式可写为：

`e = (user, item, action, metadata, timestamp)`

实验中使用有界 FIFO 队列，最多保留最近 15 条行为。新行为先进入这一层，不立即上升为长期偏好。

### 2. Preference Memory

维护可独立修改的细粒度偏好单元：

`p = (category, text, strength, evidence)`

其中 `strength ∈ [-1, 1]` 表示喜欢/不喜欢及强度，`evidence` 记录支持或反驳次数。新证据只更新相关 preference chunk，而不是重写整个用户表示。

### 3. Profile Memory

把所有 preference chunks 合成为 150–300 词左右的自然语言用户画像：

`Profile = Synthesizer(Preference Memory, Previous Profile)`

最终排序器主要读取 **Profile + recent Event**。Preference Memory 更像中间状态，用于更新与合成，而不是把三层全部塞进 ranking prompt。

## Memory Lifecycle

MARS 明确定义六种操作：

1. **Extract**：从新事件抽取偏好；
2. **Boost**：增强已有偏好；
3. **Demote**：根据反证弱化偏好；
4. **Merge**：合并重复或近似 preference；
5. **Forget**：删除已经衰退/被反驳的偏好；
6. **Synthesize**：重新生成 Profile。

Planner 根据当前状态决定执行哪些操作，也可以直接 skip。实验中每积累 3 条待处理行为检查一次状态；发生若干次 memory 变化后自动重建 Profile。

## Agent Structure

更准确地说，MARS 是 **single-agent orchestration framework**，不是 multi-agent system。Extractor、Synthesizer、Planner、Ranker 是同一个系统里的不同 LLM 调用角色。

- **Planner**：决定 memory update action；
- **Extractor**：从 pending events 提取或修改 preference；
- **Synthesizer**：生成整体 Profile；
- **Ranker**：根据 Profile、recent events、instruction 和候选完成排序。

## Training

MARS 本身基本 **不训练 LLM 参数**。

- Planner / Extractor / Synthesizer / Ranker 都通过 prompt 调用 frozen instruction LLM；
- memory 在 JSON / text 层面动态更新；
- 变化的是外部状态，不是模型权重；
- Planner 也没有通过 RL 学习 reward。

因此要区分：**memory evolution ≠ model training**。

## Inference Example

如果用户连续阅读《Dune》《Foundation》《The Three-Body Problem》：

1. 原始行为先进入 Event Memory；
2. Planner 触发 Extract；
3. Preference Memory 可能产生“喜欢科幻”“偏好文明/哲学主题”“喜欢复杂叙事”等结构化偏好；
4. Synthesizer 把它们合成长期 Profile；
5. Ranker 用 Profile + 最近事件对候选重排。

论文设定是候选重排序，不是从完整 item catalog 中做召回。

## Relation to MemRec / SAGER

- **MemRec**：主要解决“参考谁的经验”，强调跨用户/物品 collaborative memory；
- **SAGER**：主要增加每用户 Policy Skill，尝试表达“应该怎样判断”；
- **MARS**：主要解决“用户记忆如何组织和更新”，强调 Event → Preference → Profile 的纵向层次。

三者关注点不同，MARS 的核心贡献不是 collaborative retrieval，而是显式的用户状态层次与 memory lifecycle。

## Discussion Notes

我们讨论中一个关键判断是：真正的 **Intent** 应是当前状态，而 **Policy** 应是条件化决策规则。若所谓 Policy 只是“用户喜欢科幻、重视主题匹配”，它实际上仍然是 preference/memory。MARS 的分层结构对避免这种概念混淆很有帮助：短期状态、长期 preference 和整体 profile 被明确拆开。
