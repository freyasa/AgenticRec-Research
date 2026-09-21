# From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents

- **Year:** 2026
- **Venue:** COLM 2026
- **Paper:** https://arxiv.org/abs/2602.17913
- **Code:** https://github.com/FreedomIntelligence/Tiermem



## TLDR



![2026-09-16_From Lossy to Verified](<./imgs/2026-09-16_From Lossy to Verified.png>)



## Abstract

长时间运行的 Agent 通常会把历史交互压缩成 summary，以控制 memory 长度。

但这里有一个根本问题：**summary 是在未来 query 出现之前写的**。当系统总结历史时，它并不知道未来的问题到底会依赖哪个细节。因此，summary 很可能把未来真正关键的信息删掉。比如“用户对花生严重过敏”可能被压缩成比较泛化的饮食偏好，之后 Agent 就无法基于可靠证据回答相关问题。

一种解决方法是始终保留并使用完整的 raw history，因为 raw log 不会丢失信息。但这样代价又很高：很多问题其实只看 summary 就能回答，如果每次都处理完整历史，会显著增加 token 消耗和延迟。

因此作者提出 **TierMem**：一个带 provenance 的两层 memory 系统。默认先使用便宜的 summary memory；只有当 summary 不足以回答当前问题时，才访问完整 raw log。同时，从 raw log 中找到的重要信息会重新写回 summary，并且记录它来自哪段原始历史。

在 LoCoMo 上，TierMem 达到 **0.851 accuracy**，接近始终使用 raw history 的 **0.873**，同时减少 **54.1% 输入 token** 和 **60.7% latency**。



## Motivation

现在基本有两种内存存储方法：

1. summary：每次新信息到来就更新一次内存。**Summary 很便宜，但是会丢信息。**
2. raw：把所有历史信息都写入内存。**raw太贵。**

于是作者就组合了一下这两个策略



## Method

背景：所有任务共用memory，而不是rec那种1user-1mem。

可以把作者的方法理解成一个很简单的 **“summary cache + raw history + router”** 系统。**<u>其中最重要的是router，因为判断信息量够不够很难</u>**。



### 1. 先同时存两份 memory

#### Tier-2

每段对话不断往 raw history 里加。作者默认每大约 **1000 tokens** 切成一个 page：

```
Page 1: 原始对话
Page 2: 原始对话
Page 3: 原始对话
...
```

#### Tier-1

```
Tier-1 memory unit
= summary
+ embedding
+ raw page ID
```



让 LLM 对 Tier-2 的 page 做一个 summary：

```
Page 17 raw:
用户：我最近开始重新看 graphic novels...
用户：我比较喜欢人物塑造深入的作品...
...

↓ LLM summary

Summary:
用户喜欢人物塑造深入的 graphic novels
```

然后把这个 summary <u>用专用的embedding模型</u>做 embedding，放进向量数据库。作者实验里直接用了 **Mem0 + text-embedding-3-small**。



### 2. 来一个 query，先只搜 summary

先把query embedding和tier-1的embedding做top-k匹配。然后**交给一个 Router。**

Router 做二分类：

```
S = summary 已经够了
R = summary 不够，需要查 raw
```



### 3. 如果 summary 不够，就回 raw history 查

如果检索到的答案回答不了用户的问题。那么就读检索到的summary对应的raw history <u>page</u>。

如果page里有想要的东西，让Memory Manager写回tier-1。方法有`UPDATE、SKIP、ADD`。



### 4. 如果raw history也不够，就deepresearch

搜索方法有两种

```
MEM0_SEARCH
= 在 summary 上做 semantic/vector search

KEYWORD_SEARCH
= 在 raw pages 上做 BM25 keyword search
```



### 5. 关于router

需要一个额外的router来判断tier-1的memory能否回答query。

```
输入：query和tier-1memory

输出：yes/no
```

<u>但其实router没有针对于这种二分类任务去进行学习。只是去调用llm。甚至和推荐模型是一个llm。</u>
