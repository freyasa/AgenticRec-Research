# MemRec: Collaborative Memory-Augmented Agentic Recommender System

- **Year:** 2026
- **Venue:** ACL2026
- **Paper:** https://arxiv.org/abs/2601.08816
- **Code:** https://github.com/rutgerswiselab/MemRec



## TLDR

![2026-09-14_MemRec](./imgs/2026-09-14_MemRec.png)

## Motivation

1. 现有的LLM/Agentic Rec注重于孤立的memory，**没有体现协同的信息**。
2. 不能简单的把所有的collaborative information都丢给LLM。



## Abstract

推荐系统正在从传统的协同过滤逐渐发展到基于大语言模型的 Agent 推荐系统。这类系统通常依赖语义化的用户记忆和物品记忆来进行预测。然而，现有的 Agent 往往彼此独立地维护这些记忆，因此忽略了重要的协同信号，例如用户与物品之间的共同交互，以及社区中相似用户之间的关系。这限制了系统发现用户隐藏偏好和准确理解用户需求的能力，尤其是在用户数据较少的情况下。

为了解决这一问题，我们提出了 **Collaborative Memory（协同记忆）**：将原本彼此独立的语义记忆连接起来，从而能够共享关系信息。但是，如果直接把大量协同记忆全部交给 LLM，会带来严重的上下文过载、噪声以及很高的计算成本。

因此，我们提出 **MemRec**，在架构上将“记忆管理”和“推荐推理”分离。MemRec 使用一个专门的、较轻量的语言模型 **LM_Mem**，在后台维护并总结一个动态的 collaborative memory graph；然后只把经过压缩的、高价值的信息提供给更强的 **LLM_Rec**，由它完成最终的推荐。四个 benchmark 上的实验表明，MemRec 达到了 SOTA 性能。 



## Method

### 利用不同的user交互过同一个item形成一个graph

原始

```
User A --点击过--> Item 1
User A --点击过--> Item 2

User B --点击过--> Item 2
User B --点击过--> Item 3
```

变成

```
User A -- Item 1
   |
 Item 2
   |
User B -- Item 3
```

并且**每一个user都有一个memory。**

那么A的neighbor是

```
Item 1
Item 2
User B
```



### 从刚才那个 graph 里，找出对当前 user 最有用的邻居

**这个图太大，直接检索可能上下文太长**。因此需要sample出top-k邻居。

这一步要从上面的neighbor中进行筛选，**打分过程完全不依赖LLM**。但是**打分规则是依赖LLM进行生成的**。例如：

```
metadata overlap > 0.6 			 → ×2.5
共同交互 > 3        			   → ×1.8
memory similarity > 0.5			→ 再 boost
太久以前的 interaction				→ decay
```

然后根据分数选择top-k，另外需要保证top-k里至少有x个user和y个item。



### LLM把user + 邻居信息进行总结（Stage-R）

输入

```
User A memory:
喜欢科幻电影

Neighbors:
User B: 最近看了 Interstellar、Arrival
User C: 最近看了 Dune、Blade Runner
Item X: Interstellar
...
```

输出包含了每一个facets，具体包括`facet, confidence, supporting_neighbors`。

```
{
  "facets": [
    {
      "facet": "喜欢科幻题材",
      "confidence": 0.95,
      "supporting_neighbors": [
        "User-B",
        "User-C",
        "Item-X"
      ]
    },
    {
      "facet": "偏好太空探索类电影",
      "confidence": 0.82,
      "supporting_neighbors": [
        "User-B",
        "Item-X"
      ]
    },
    {
      "facet": "可能喜欢严肃、世界观较强的科幻",
      "confidence": 0.75,
      "supporting_neighbors": [
        "User-C"
      ]
    }
  ],
  "support_edges": [
    {
      "from": "User-B",
      "to": "User-A",
      "w": 0.9
    },
    {
      "from": "User-C",
      "to": "User-A",
      "w": 0.8
    },
    {
      "from": "Item-X",
      "to": "User-A",
      "w": 0.85
    }
  ]
}
```



### 打分

输入

```
1. 前一步得到的 preference facets
2. 每个 facet 的 confidence
3. candidate items
4. candidate 的 item memory
5. 如果有的话，再加当前 user instruction
```

输出每个candidate的score和reason

```
{
  "scores": [
    {
      "item_id": 10,
      "score": 0.94,
      "rationale": "Strong match with science fiction and space exploration preferences."
    },
    {
      "item_id": 11,
      "score": 0.30,
      "rationale": "Does not strongly match the user's science fiction preferences."
    },
    {
      "item_id": 12,
      "score": 0.88,
      "rationale": "Strong science fiction match, though less focused on space exploration."
    }
  ]
}
```



### 更新memory（stage-w）

更新ground truth item和user的memory

```
这次点击
+
刚才得到的 preference facets
+
User A 当前 memory
+
Item 10 当前 memory
+
相关 neighbors
```
