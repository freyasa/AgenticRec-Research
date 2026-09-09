# AgenticRec — A Recommendation-Oriented Agentic Framework with Progressive Tool-Integrated Reasoning Optimization

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2603.21613
- **Drive note:** https://docs.google.com/document/d/1kY43TuZrvWZYTMqGa7b9nC35IMX-nF-wAQIRbaKqYRg/edit

## Motivation

现有 recommender agent 即使能够调用工具，其 Think–Act–Observation trajectory 也未必和最终 ranking feedback 对齐。模型可能会“看起来在推理、也调用了工具”，但这些行为并没有真正帮助把真实下一交互 item 排得更靠前。

AgenticRec 的核心目标是：

> 直接用推荐排序反馈训练 agent 的工具使用与推理轨迹，再从模型自己的 ranking error 中挖掘 hard pairs，进一步细化 preference boundary。

## Task Setting

实验使用 Amazon Reviews 2023 的四个子集：

- CDs and Vinyl
- Musical Instruments
- Office Products
- Video Games

时间范围为 2022-10 到 2023-10，按时间 8:1:1 划分 train/valid/test，历史长度最多 10。

每个实例：

- 1 positive
- 19 random negatives
- 20 candidates shuffled
- 输出 Top-10

因此本质仍是 sampled-candidate reranking，而不是 full-catalog retrieval。

## Agent Structure

系统中心是一个 **Qwen3-4B-Instruct-2507**，属于 single-agent ReAct-style framework。

一次推荐中可以反复执行：

`Think → Act → Observation → Think → Act → Observation → ... → Recommendation`

四类外部工具：

1. **User Profile Tool**：长期用户画像；
2. **Item Information Tool**：候选 metadata 与集合分析；
3. **Behavioral Statistics Tool**：近期类别、session interest、rating statistics；
4. **Collaborative Information Tool**：通过训练集上的 SASRec embeddings 检索相似用户/物品。

这些 tool 不是独立 agent，只是外部函数/服务。

## Training Preparation

在 RL 前先准备：

- 在 Amazon train split 上训练 SASRec，建立 collaborative retrieval space；
- 用同系列 Qwen 离线生成用户 profile；
- 建立 metadata query tool；
- 建立行为统计 tool；
- 把交互转换成 1 positive + 19 negatives 的 ranking task。

## Stage 1: RTA — Recommendation-Oriented Trajectory Activation

### Rollout

对同一个 request 采样 `G=8` 条 trajectory。不同 rollout 可以：

- 调不同工具；
- 形成不同 Think；
- 得到不同 Top-10。

单条 trajectory 最多调用 10 次工具。

### Ranking Reward

若格式合法且 positive 进入 Top-10，则 reward 使用 positive 所在位置对应的 NDCG@10。

例如只有一个 positive 时：

- rank 1 → reward 1.0
- rank 2 → ~0.631
- rank 3 → 0.5

若 positive 不在 Top-10：`-0.5`；格式非法或 tool budget 超限：`-1`。如果 positive rank=1 且至少调用一次工具，还有额外 `+0.1`。

### GRPO

同一 request 的 8 条 trajectory 计算组内 baseline，简化优势可写为：

`A(g) = R(g) - mean(R)`

高于组均值的轨迹被增强，低于组均值的被抑制。RTA 训练 3 epochs。

关键点：这里优化的是 **整条 agent trajectory 的生成概率**，不是一个静态 ranking classifier。

## Stage 2: PPR — Progressive Preference Refinement

RTA 之后，用当前 policy 重新跑训练实例并查看自己的 ranking errors。

### Hard-Negative Mining

如果 positive `c+` 不是第 1：

- 将排在 `c+` 前面的 candidates 作为 hard-negative pool；
- 若 `c+` 不在 Top-10，则将整个 Top-10 作为 pool；
- 从中采样一个 `c-`，形成 hard pair `(c+, c-)`。

这是一种 self-bootstrapped / on-policy hard-negative mining：

`current policy → ranking violation → hard pair → further training`

论文实际报告的是一轮 PPR（1 epoch），并没有持续多轮“重挖—再训练”。

### Bidirectional Preference Task

同一个 hard pair 被改写成两个 prompt：

- positive direction：哪个更可能被用户选择？正确答案 `c+`
- negative direction：哪个更不可能被用户选择？正确答案 `c-`

两个任务仍允许 Think + tools + Observation。

### PPR Still Uses GRPO

PPR 没有直接计算 BPR / hinge / logistic loss。

它仍然采样 trajectory，并用二元 reward：

- correct = 1
- wrong = 0

然后继续 GRPO 更新 LLM policy。

论文理论里出现的 pairwise logistic expression 主要是解释双向训练，不是实现中的直接 supervised loss。

## How Multi-Step Trajectory Is Backpropagated

这是我们讨论里一个很重要的问题。

### Rollout Is Interactive

真实执行不是一次性生成：

1. LLM 生成 `Think1 + Act1`；
2. 程序执行工具并插入 `Obs1`；
3. 再把已有 transcript 输入 LLM；
4. 继续生成 `Think2 + Act2`；
5. 插入 `Obs2`；
6. 最后生成 `Recommendation`。

### Training Reconstructs One Transcript

rollout 完成后，系统已有完整 transcript：

`[Prompt, Think1, Act1, Obs1, Think2, Act2, Obs2, Rec]`

训练时将其拼成普通 causal sequence，但使用 action mask：

- Prompt: mask 0
- Observation: mask 0
- Think / Act / Rec: mask 1

因此可以通过一次 teacher-forcing forward 重算所有 policy token 的 log probability。

轨迹概率只包含 LLM 自己生成的部分：

`log πθ(τ) = Σ_{t:mask=1} log πθ(z_t | z_<t)`

Observation 属于 environment，只作为后续 token 的 context，不参与 policy log-prob，也不会把梯度传回 SASRec/tool。

## Why NDCG Can Train the LLM

NDCG 不需要可导，因为这里是 policy gradient。

简化目标：

`L ≈ - Σ_g A(g) Σ_{t:mask=1} log πθ(z_t^(g) | z_<t^(g))`

ranking metric 只作为 detach 的 scalar reward/advantage：

- `A > 0`：提高整条自生成 trajectory 的概率；
- `A < 0`：降低它的概率。

因此底层仍然依赖 next-token log-probability 获得梯度，但这不是普通 SFT/NTP，而更接近 **reward-weighted next-token policy optimization**。

## Credit Assignment Limitation

整条 trajectory 通常共享一个最终 advantage，所以模型只能统计性学到“哪些工具调用/推理模式更经常带来高 reward”。它不能严格知道某一句 Think 或某一个 tool call 的独立因果贡献。

因此是 trajectory-level credit assignment，而不是精确 step-level attribution。

## Experiments

论文报告在四个数据集、20 个指标中 19 个最好；唯一未第一的是 Office H@1。

消融也显示：未经训练的工具调用并不天然有用。某些数据集上 Frozen tool-integrated reasoning 甚至低于 frozen reasoning-only；经过 RTA 后工具增强才稳定变好。

PPR 在多个数据集的 H@1 上进一步提升。

## Important Open Questions

论文尚未充分拆解：

- 四种工具各自贡献多大；
- hard negative 相对 random negative 的独立贡献；
- bidirectional PPR 是否真的优于 positive-only PPR；
- 训练采用 full-parameter GRPO 还是 LoRA/QLoRA；
- sampled 20-candidate setting 对真实 hard retrieval candidate 的泛化。

## Key Takeaway from Our Discussion

这篇论文最值得单独记住的不是“用了 agent”本身，而是两个技术点：

1. **用最终 listwise ranking feedback 直接优化工具增强 trajectory**；
2. **从当前 policy 的 ranking violation 中自挖 hard negatives，再继续 refinement**。

而所谓 bidirectional preference reasoning 的独立贡献仍缺少 positive-only ablation。
