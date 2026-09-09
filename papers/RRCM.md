# RRCM — Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2605.07129
- **Drive note:** https://docs.google.com/document/d/1cdckM8IHyouFadvwE9tnBZdTQNx1aCcNWxako1d_lNU/edit

## Motivation

很多 LLM recommender 的 context construction 是固定的：

- 每次都把完整历史塞进 prompt；
- 或者每次都固定执行 RAG / collaborative retrieval / metadata retrieval。

RRCM 认为这不合理，因为不同实例真正缺失的 evidence 不一样。有些用户历史已经足够明确，不需要额外检索；有些稀疏历史、冷门 item 或模糊候选则需要额外 collaborative 或 metadata 信息。

因此核心问题变成：

> **能不能让 retrieval 本身成为 agent 的 decision/action，并直接用 ranking outcome 来学习 retrieval policy？**

## Core Idea

LLM agent 根据当前 recommendation context 动态选择：

1. 不检索，直接推荐；
2. retrieve Collaborative Memory；
3. retrieve Meta Memory；
4. 两种 memory 都检索；
5. 基于得到的 evidence 完成最终 ranking。

所以 RRCM 的重点不是提出一种新的 memory representation，而是：

> 把 retrieval 从固定 pipeline 变成可优化的 agent action。

## Two Memories

### 1. Collaborative Memory

提供跨用户行为证据，即“和当前用户具有相似交互模式的人还喜欢什么”。

它把 collaborative signal 转成 LLM 可以阅读的自然语言 evidence。

### 2. Meta Memory

提供 item-side metadata / semantics，例如：

- category
- description
- attributes
- functional information

它回答的是“这个 item 是什么”。

## Are the Memories Dynamic?

不是。

RRCM 的 memory corpus 在离线阶段构建，并建立 retrieval index；训练和推理阶段 memory 内容保持固定。

动态变化的是：

- LLM 的 retrieval policy；
- 是否检索；
- 检索哪一种 memory；
- 如何利用 retrieved evidence。

因此可以把它理解为：

- **Memory = external fact/evidence store**
- **Agent policy = learned controller for reading the store**

这和 MARS / MemRec 那种 memory 本身持续演化的路线明显不同。

## Training Pipeline

`Base LLM → SFT warmup → GRPO → RRCM agent`

它不是单纯对完整 trajectory 做普通 NTP/SFT。

## Stage 1: SFT Warmup

SFT warmup 主要用于让 base LLM 先学会“像一个 agent 一样工作”，包括：

- 输出基本 agent 格式；
- 生成 `<think>` / tool call / answer；
- 判断何时需要 retrieval；
- 生成 retrieval query；
- 读取和使用 retrieval result；
- 完成推荐任务的基本交互流程。

所以 SFT 的作用更像 **behavior initialization / format & tool-use bootstrapping**，而不是直接把最终 ranking metric 优到最好。

SFT 之后再进入 RL，让模型真正学习哪些 retrieval decisions 对 ranking 有帮助。

## Stage 2: GRPO

一个可能的 trajectory：

```text
User History
→ Think: 当前信息不足
→ Action: Retrieve Collaborative Memory
→ Observation: similar users liked ...
→ Think: evidence now sufficient
→ Final Ranking
```

RRCM 用最终 ranking outcome 给 trajectory reward，并通过 GRPO 更新 policy。

Reward 与推荐质量直接相关，例如：

- Recall
- NDCG
- Hit Rate

若某类 retrieval strategy 更容易带来更好的最终 ranking，未来模型就更倾向选择类似 action。

所以这里真正优化的是：

> `retrieval decision → acquired evidence → final ranking quality`

而不是某个局部 token 的 cross-entropy accuracy。

## Difference from Standard RAG

### Standard RAG

`Query → Retriever → LLM`

Retriever 基本固定执行，LLM 只消费结果。

### RRCM

`Context → LLM Agent → decide whether/what to retrieve → retrieve → reason → rank`

LLM 先判断自己缺什么 evidence，再决定是否调用哪类 memory。

因此 RRCM 学的是 **retrieval policy**，而不仅是 retrieval content。

## Relation to AgenticRec / MemRec / SAGER

### MemRec

重点：如何构建与维护 memory，尤其 collaborative semantic memory。

### SAGER

重点：如何在 memory 之上维护 user policy / ranking skill。

### AgenticRec

重点：通过工具增强 trajectory + ranking reward 学习工具使用策略，并使用 ranking violations 构造 hard pairs。

### RRCM

重点：让 agent 主动判断当前 recommendation 缺少什么 evidence，并把 retrieval 作为 trajectory 核心 action。

## Discussion Point: Why RAG Is Explicitly Mentioned

RRCM 把 RAG 当作重要对照，是因为作者的核心批评对象就是 **fixed retrieval pipeline**。

普通 RAG 默认“retrieval 一定发生”；RRCM 则认为 recommendation 场景中的 evidence demand 是 instance-specific 的，应该由 agent 学习：

- 是否 retrieve；
- retrieve collaborative 还是 metadata；
- 是否需要两者；
- 什么时候停止检索并开始 ranking。

所以论文强调 RAG 不是随便拿来类比，而是和其核心 motivation 直接对应。

## Key Distinction from Memory-Evolution Methods

RRCM 不试图让 LLM 不断重写用户记忆。

它的思路更接近：

> 保持 evidence store 相对稳定，把学习能力放在“什么时候读什么”上。

这使得它和“不断生成更抽象 user policy / user profile”的路线形成明显对比。

## Final Takeaway

RRCM 的主要贡献可以压缩成一句话：

> **用 ranking reward 学习一个 adaptive evidence-acquisition policy，让 LLM 推荐 agent 主动决定是否以及如何读取 collaborative / metadata memory。**

这里真正被优化的是 information acquisition strategy，而不是 memory corpus 本身。
