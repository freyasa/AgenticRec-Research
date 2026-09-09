# CoVeMem — When Memory Takes Gradients: Collaborative Vector Memory for Agentic Recommender Systems

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2608.26895
- **Drive note:** https://docs.google.com/document/d/1TQPJxBMLm5Xmj-4DKqFX6Mfo-g766hEGD9eZtFyqQQk/edit

## Motivation

现有 agentic recommender 多把 memory 写成自然语言。作者认为这种 text memory 有两个核心问题：

1. 新交互到来后需要额外 LLM 调用去重写/维护 memory，成本高；
2. user–item collaborative geometry 一旦被压成少量文本 preference，很容易丢失连续、细粒度的协同关系。

CoVeMem 的思路是：**语义偏好继续用文字表达，而协同核心直接保留成向量 memory。**

## One-Sentence Understanding

先训练传统推荐模型得到 user/item embeddings，再通过 projector 把这些 embedding 映射成 LLM input space 里的 soft tokens，并通过 LoRA 训练 LLM 学会读取这些向量。

因此它不是 API-only agent，而是：

> collaborative vector memory + textual profile + trainable LLM reader

## Stage 0: Build Collaborative Memory

论文使用三层 LightGCN，在训练集交互图上用 BPR 目标训练 50 epochs，得到：

- user state `s_u ∈ R^64`
- item state `v_i ∈ R^64`

训练完成后，整个 memory bank 冻结：

`M_vec = {s_u} ∪ {v_i}`

validation/test target 不进入交互图。

每个用户另外有一段一次性生成的短文本 profile，用于显式语义偏好。

## Candidate-Conditioned Retrieval

当前候选集为 `C_u` 时，先计算候选 item vectors 的中心：

`c_bar = (1 / |C_u|) Σ v_c`

然后在可用历史中寻找与 `c_bar` 点积最大的 K 条历史，默认 `K=5`。

所以它不是取“最近 5 条”，而是取“与当前候选最相关的 5 条历史”。

这是一种 candidate-aware memory retrieval。

## Soft-Token Injection

一个 gated MLP projector `φ` 将 64 维 collaborative vector 映射到 Qwen2.5-7B 的 3584 维 input embedding space。

- `φ(s_u)`：user soft token
- `φ(v_h)`：historical item soft tokens
- `φ(v_c)`：candidate item soft token

这些向量不是离散 token ID，而是直接通过 `inputs_embeds` 进入 Transformer。

Prompt 同时包含：

- textual user profile
- selected historical titles
- history soft tokens
- user soft token
- candidate title
- candidate soft token

## Training Stage 1: Semantic Anchor Alignment

第一阶段只训练 projector。

对 item `i`，使用其标题/类别对应的 Qwen input embeddings 构造固定 semantic anchor `y_i`，再令：

`h_i = φ(v_i)`

使用 batch 内对称 contrastive objective：

`S_ij = normalize(h_i)^T normalize(y_j) / τ`

`L_align = 1/2 [CE(S, diagonal) + CE(S^T, diagonal)]`

目标是让 collaborative item vector 投影后靠近自己的文本语义锚点，而远离其他 item。

这一阶段：

- LightGCN frozen
- Qwen backbone frozen
- Qwen input embeddings frozen
- **only projector is trained**

## Training Stage 2: Ranking Co-Training

每条训练事件包含：

- prefix history
- 1 positive
- 9 negatives
- 10 candidates shuffled into one prompt

LLM 在 answer position 输出 A–J option logits。

论文使用的不是标准 NTP loss，也不是 ListNet/ListMLE，而是：

`L_rank = average_j max(0, m - (z_pos - z_j))`

即：

> listwise candidate context + one-positive-vs-all pairwise hinge loss

### Candidate Masking

训练时每个 candidate title 以 `p=0.5` 独立 mask；mask 后只剩 soft token，迫使模型真正读取 vector memory。

若正样本被 mask，则只与同样被 mask 的负样本比较，避免信息不对称。

### Trainable Parameters

Stage 2 更新：

- projector `φ`
- Qwen attention 中 `Wq/Wk/Wv/Wo` 上的 rank-4 LoRA

冻结：

- Qwen original parameters
- LM head
- LightGCN embeddings

Projector + LoRA 总计约 6.8M trainable parameters，不到总模型 0.1%。

## How Ranking Loss Backpropagates

假设正样本 logit `z+ = 2.5`，负样本 `z- = 2.3`，margin=1：

`loss = max(0, 1 - (2.5 - 2.3)) = 0.8`

0.8 是 loss 数值，不是梯度。

激活时：

- `∂L/∂z+ = -1`
- `∂L/∂z- = +1`

梯度路径为：

`L_rank → option logits → answer hidden state → self-attention → soft-token embeddings → projector`

LoRA 位于 attention projection 中，因此也会获得梯度。

LightGCN vectors 虽然数学上仍在计算图上游，但实现中已 freeze/detach，因此梯度不会更新 memory bank 本身。

## Why It Does Not Collapse Easily

- 所有 logits 相同会产生非零 hinge loss；
- projector 输出相同向量无法降低 Stage-1 contrastive loss；
- candidate order 被随机打乱，不能靠位置 shortcut；
- soft-token gate、dropout、weight decay、gradient clipping 进一步稳定训练。

更现实的风险不是数值 collapse，而是 shortcut learning：模型可能依赖 title、popularity 或 collaborative backbone，而不是形成真正的 memory reasoning。

## Inference

训练是 10 candidates together + A–J logits；推理却改成 pointwise Yes/No：

- 每个 candidate 单独构造 prompt；
- 读取 answer position 的 `Yes` logit；
- 对所有候选分数排序。

因此存在明显的 train–inference mismatch。

## Experiments

四个 InstructRec 数据集：Books、Goodreads、MovieTV、Yelp。

每个测试实例：1 positive + 9 random negatives。

论文所说“19/20”是相对 MemRec：18 个指标更高、1 个相同、1 个更低；不是在 19 个指标上超过所有 baseline。

一个非常关键的消融：Goodreads 上去掉 LoRA 后 H@1 约 0.10，接近 10 候选随机水平。这说明：

> 只把 collaborative vector 经过 projector 塞给 frozen LLM 并不够；LLM reader 必须被训练。

## Relation to API-Only Agentic Recommendation

MemRec 等文本 memory 方法可以使用黑盒/API LLM；CoVeMem 不行，因为它需要：

- input embedding access
- `inputs_embeds`
- model parameters
- output logits
- projector training
- LoRA training

所以严格来说，它更接近 **LLM-based hybrid neural recommender/scorer**，而不是 training-free agent。

## Key Clarification

“Memory Takes Gradients”并不意味着 Stage 2 的 ranking loss 在线更新每个 user/item LightGCN state。

真正发生的是：

- collaborative memory content 由 LightGCN/BPR 离线学习；
- ranking gradient 学习的是 **如何翻译和读取这些 memory states**（projector + LoRA）。

## Full Pipeline

`Train LightGCN → freeze user/item states → contrastively align projector → jointly train projector + LoRA with ranking loss → pointwise Yes-logit inference`
