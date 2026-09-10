# CoVeMem — When Memory Takes Gradients: Collaborative Vector Memory for Agentic Recommender Systems

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2608.26895
- **Drive note:** https://docs.google.com/document/d/1TQPJxBMLm5Xmj-4DKqFX6Mfo-g766hEGD9eZtFyqQQk/edit



## TLDR

直接把 CF 模型学到的 user/item embedding 当成 memory，然后训练 LLM 学会“读”这种 memory。而不是使用Text去做Memory。



![2026-09-10_ CoVeMem](./imgs/2026-09-10_ CoVeMem.png)

## Motivation

现有 agentic recommender 多把 memory 写成自然语言。作者认为这种 text memory 有两个核心问题：

1. 新交互到来后需要额外 LLM 调用去重写/维护 memory，成本高；
2. user–item collaborative geometry 一旦被压成少量文本 preference，很容易丢失连续、细粒度的协同关系。

CoVeMem 的思路是：**语义偏好继续用文字表达，而协同核心直接保留成向量 memory。**



## Method

就是先从cf模型中拿到user和item的emb，然后对齐到LLM的embedding上，再train LLM，拿到logits做softmax



### 先训练 collaborative model

- 默认是 LightGCN。
- 用用户–物品交互图训练，论文里用 BPR loss。
- 训练完以后，user/item embedding 就构成 vector memory bank。
- 后续阶段 **LightGCN 和这些 embedding 都被冻结**。



### 训练 Projector —— “怎么把 CF 向量变成 LLM 能接收的东西”

- 把 64 维左右的 collaborative embedding 映射（**<u>gated MLP</u>**）到 LLM 的 embedding space。

**<u>label来源于原始的 title + category token 的 input embedding 平均值</u>。**

然后在in batch上做了双向info nce。



### 训练 LLM 的 LoRA + projector

- Qwen 主干参数冻结。
- **<u>只训练 attention 上的 LoRA，以及 projector</u>**。
- 用 candidate ranking supervision，让 LLM 真正学会使用这些 vector soft tokens。
- 论文还会随机 mask 一部分 candidate 文本，防止 LLM 只靠标题语义做题、完全忽略 vector memory。

**candidate ranking supervision：**

输入了user info和10candidate，根据每一个logits得到10个分数

loss
$$
L = \sum_j max(0, m - (z^+ - z_j))
$$
m代表margin



## Inference

训练是 10 candidates together + A–J logits；推理却改成 pointwise Yes/No：

- 每个 candidate 单独构造 prompt；
- 读取 answer position 的 `Yes` logit；
- 对所有候选分数排序。

因此<u>存在明显的 train–inference mismatch</u>。



## Experiments

四个 InstructRec 数据集：Books、Goodreads、MovieTV、Yelp。

每个测试实例：1 positive + 9 random negatives。

论文所说“19/20”是相对 MemRec：18 个指标更高、1 个相同、1 个更低；不是在 19 个指标上超过所有 baseline。

一个非常关键的消融：Goodreads 上去掉 LoRA 后 H@1 约 0.10，接近 10 候选随机水平。这说明：

> ### 只把 collaborative vector 经过 projector 塞给 frozen LLM 并不够；LLM reader 必须被训练。
