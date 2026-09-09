# AMEM4Rec — Leveraging Cross-User Similarity for Memory Evolution in Agentic LLM Recommenders

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2602.08837
- **Drive note:** https://docs.google.com/document/d/1mELwwpYXBjRUTa8KBBkkauFBAeDV5B7Y7A3OnI-IqWU/edit

## Motivation

很多 agentic LLM recommender 依赖自然语言用户记忆，但主要利用语义信息，缺少推荐系统中关键的跨用户 collaborative signal。AMEM4Rec 希望不依赖预训练 CF backbone，而是让不同用户的行为模式在一个全局 memory pool 中相互连接、合并和演化。

## Core Idea

AMEM4Rec 保存的不是传统 user/item embedding，而是 **LLM 从用户历史窗口总结出的行为模式文本**。

一个 history window 先被 LLM 转成结构化 pattern，例如：

- Behavior Explanation
- Pattern Description

随后整段 pattern 由 Sentence-BERT 编码：

`e_k = SBERT(p_k)`

最终 memory item 为：

`m_k = (pattern text p_k, embedding e_k)`

也就是说，memory 同时包含：

- 可供 LLM 阅读的自然语言模式；
- 可供 retriever 做余弦相似度匹配的向量。

## Memory Construction and Evolution

每产生一条新的 pattern memory，就用其 embedding 在全局 memory pool 中寻找 top-k 相似旧 memory。

这里是 **memory-to-memory matching**：两端都是 LLM 生成的行为模式文本，因此抽象层级比较一致。

检索后还有两层判断：

1. **Similarity Validator**：根据向量相似度判断 save / update-and-save / update；
2. **Semantic Validator**：让 LLM 阅读文本，进一步判断两个 memory 是否真的应该连接。

确认连接后，旧 memory 会吸收新模式并被重写，从而在多个用户之间积累共享行为规律。

## Inference Retrieval

推理阶段和建库阶段不同。

当前用户近期历史主要由 item title / category 组成，先编码为 query embedding：

`q_u = SBERT(history text)`

然后计算：

`cos(q_u, e_j)`

检索 top-k memory，再把：

- 原始用户历史；
- 检索到的 behavior-pattern memories；
- 当前候选列表；

一起交给 LLM 做 reranking。

## Important Representation Gap

这是我们讨论 AMEM4Rec 时最关键的问题：

### 建库阶段

`behavior-pattern text → SBERT → behavior-pattern memory`

两边文本形式一致，属于相对自然的同构匹配。

### 推理阶段

`raw item titles/categories → SBERT → abstract behavior-pattern memory`

两端抽象层次不同，而且论文没有明确的 task-specific history-memory alignment training。

因此推理阶段存在明显的 **representation gap**。

## Why It Can Still Work

Sentence-BERT 的通用语义空间可以识别较粗的主题关系。例如历史里出现 Dark Souls、Elden Ring、Action RPG，而 memory 写的是“偏好高难度战斗、幻想世界和探索”，两边可能因为共享 action RPG / fantasy / combat / exploration 语义而匹配。

但这种成功主要依赖预训练文本 encoder 的语义泛化，而不是模型真正从 recommendation supervision 中学会了“哪段历史应该检索哪条 memory”。

## Failure Modes

### 1. Semantic similarity ≠ recommendation relevance

两段历史属于相同主题，不代表它们具有相同 next-item transition。

### 2. Sequence information can be lost

例如：

- camera → memory card → tripod
- camera → photography book → photo printer

两者都充满摄影词汇，但行为转移意图不同；一个 pooled sentence embedding 很容易忽略顺序差异。

### 3. Semantic similarity ≠ collaborative relation

传统 CF 能捕捉“文本并不相似，但经常被同一批用户共同消费”的关系。AMEM4Rec 把协同规律翻译成自然语言后，可能主要留下主题信息，而损失细粒度 co-occurrence geometry。

## How to Interpret Its CF Claim

AMEM4Rec 的确进行了跨用户聚合：相似用户行为 pattern 会进入同一个 global memory pool 并被反复融合。

因此它不是纯 per-user profile method。

但更准确的描述是：

> **cross-user aggregated semantic behavior patterns**

而不是保留完整 collaborative geometry 的传统 CF。

## Potential Improvements Discussed

1. 推理时也先把 history 总结成与 memory 相同格式的 behavior pattern，再做 retrieval；
2. 训练双塔 history encoder / memory encoder，用正负 history-memory pairs 做 contrastive alignment；
3. 做 candidate-aware retrieval，让本次候选集参与 query 构造，而不是只检索一般用户兴趣。

## Final Judgment

AMEM4Rec 最自然的部分是 memory-to-memory evolution；真正薄弱的是 inference 时 raw history → abstract memory 的匹配。它在主题清晰、候选差异大的情况下可能有效，但在 hard-negative reranking、序列转移、互补关系和细粒度协同场景中存在较明显的错配风险。
