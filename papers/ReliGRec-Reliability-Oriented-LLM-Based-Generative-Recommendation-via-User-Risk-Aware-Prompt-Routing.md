# ReliGRec: Reliability-Oriented LLM-Based Generative Recommendation via User-Risk-Aware Prompt Routing

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2609.16560



## TLDR

背景：SID + Graph + LLM Rec<br />1. 根据用户历史生成用户兴趣向量<br />2. 根据不同时间窗口的CF信息生成不同时间段用户兴趣向量<br />3. 根据以上两个信息生成一个分数选择Prompt<br />4. history SID + graph的emb + Prompt 做 inference

![2026-09-18_ReliGRec](./imgs/2026-09-18_ReliGRec.png)

## Abstract

现实世界推荐系统中的用户行为具有很强的异质性。一些用户表现出较为一致的偏好，而另一些用户则可能出现兴趣突然变化、短时间内集中交互、过度重复，或与协同邻域中的用户行为不一致等现象。这些偏离既可能来自正常的兴趣变化，也可能来自包括刷榜攻击（shilling attack）在内的操纵行为，但仅凭这些偏离并不能说明用户具有恶意。

现有的鲁棒推荐方法通常在训练阶段，通过样本重加权或图聚合等方式利用用户风险信号；而在基于 LLM 的生成式推荐中，如何根据估计得到的用户级弱风险来动态调整生成过程，目前仍缺乏研究。

本文提出 **ReliGRec（Reliability-oriented Generative Recommendation）**，一个弱监督的生成式推荐框架。其中，“Reliability”表示模型的设计目标，而不是一个直接监督得到的可靠性变量。

ReliGRec 首先利用部分用户的评论反馈信号构造用户级的**弱风险代理标签（weak-risk proxy labels）**，并分别使用 **Behavior Token** 和**时序 Graph Tokens** 表示用户的序列行为和协同上下文。随后，一个 **Dual-View Weak-Risk Estimator** 融合这两类表示，得到用户级弱风险分数，并在推理时根据该分数选择 **Simple Prompt** 或 **Cautious Prompt**。

其中，Cautious Prompt 会引导模型更多关注稳定且得到协同信息支持的证据，同时减少对孤立、短期或重复交互行为的过度依赖。Behavior Token 主要通过风险估计和 Prompt 路由间接影响生成，而聚合后的 Graph Token 则直接为下一物品的 Semantic ID 生成提供协同信息。

因此，ReliGRec 将“弱风险估计”从一个辅助预测任务转化为了一个**生成阶段的控制信号**。实验结果表明，该方法在推荐性能和弱风险代理标签预测上具有竞争力；同时，作者还分析了基于弱风险的 Prompt 路由在推荐质量和推理成本方面的表现。



## Motivation

用户的行为有时候会出现和其他用户行为不一致的现象，并且用户的兴趣会随着时间变化。因此需要对时间窗口去做切分，得到不同时间段用户的兴趣。

## Method

SID based LLM Generative Rec

**SID部分**：LETTER的RQ-VAE

**Rec部分**：


$$
\text{History}\rightarrow \begin{cases} \text{Transformer}\rightarrow \text{Behavior Token}\\ \text{Temporal Graph}\rightarrow \text{Graph Token} \end{cases} \rightarrow \text{Risk Estimator} \rightarrow \text{Prompt Routing}
$$



$$
\text{History + Prompt + Graph Token} \rightarrow \text{LLM} \rightarrow \text{Next Item SID}
$$
LLM 用的是 **Qwen2.5-1.5B-Instruct + LoRA**。整个Rec模型是e2e的。梯度都连在一起。



### 构建Behavior Token

有点像sasrec，把history对应到embedding送入一个transformer里面，拿最后一个hidden state。



### 构建Temporal Graph Token

把整个数据集按照时间patch不同子数据集。在子数据集里构建u-i图。对图跑graph encoder得到每个item和user的embedding。再去计算整个窗口内的所有item emb的pooling。

因此这个时间的\[user\_embedding=concat(u_{emb}, pooling(i_{imbs}) \]。其实意思就是\[\text{user embedding} + \underbrace{\text{窗口内全局 item embedding 的平均}}_{\text{一个向量}}\]

每个时间段user都有一个embedding。把这一组embedding送入一个transformer里面，拿最后一个hidden state（和Behavior Token Transformer不共享权重）。



### 选择不同的Prompt

拿到上述的 \[B_u \quad \text{Behavior Token}\] 和 \[\bar g_u \quad \text{Temporal Graph Token}\] 之后。需要根据这两个Token输出一个score，根据score选择两套Prompt。

**先做投影**

\[z_u^{B}=W_B B_u\]

\[z_u^{G}=W_G \bar g_u\]

\[z_u=[z_u^B;z_u^G]\]



**再过一个MLP**

\[s_u=\sigma(\mathrm{MLP}(z_u))\]



**选择Prompt**

\[s_u < \tau \Rightarrow \text{Simple Prompt}\]

\[s_u \ge \tau \Rightarrow \text{Cautious Prompt}\]



这两种Prompt是死的，只能选择不能更新。



### 做Rec

那会拿到的Temporal Graph Token过一个Projector，变成能输入到LLM的维度。被叫做Graph Emb。

然后：\[\boxed{ \text{History SID + Graph Emb + Prompt} \rightarrow \text{LoRA Qwen} \rightarrow \text{Next Item SID} }\]



## Experiments（由LLM生成，论文的卖点不显著，懒得看了）

实验主要分成四块：**整体推荐效果、weak-risk 预测、prompt routing、graph ablation**。数据集只用了 **Amazon Beauty 和 Yelp**，都是按时间排序后做 leave-one-out；Beauty 有 22,363 个用户、12,101 个 item，Yelp 有 30,545 个用户、34,139 个 item。

整体推荐对比了 LightGCN、SASRec、LETTER-LC-Rec、LETTER-TIGER、GraphRfi、PGT4Rec。Beauty 上 ReliGRec 最好，例如：

| Model        | H@10        | NDCG@10     |
| ------------ | ----------- | ----------- |
| SASRec       | 0.04914     | 0.02503     |
| LETTER-TIGER | 0.05755     | 0.03100     |
| **ReliGRec** | **0.06269** | **0.03698** |

所以 Beauty 上提升还是比较明显的。但 **Yelp 基本没有赢 LETTER-TIGER**：ReliGRec 的 NDCG@10 是 0.01701，LETTER-TIGER 是 0.01714，几乎一样。

然后他们单独验证 weak-risk estimator。Beauty 上：

$\text{Behavior only AUPRC}=0.2484$$\text{Graph only}=0.2090$$\boxed{\text{Behavior + Graph}=0.2631}$

说明两种表示合起来预测 weak-risk 确实更好。Yelp 上 ReliGRec 的 AUPRC 也从 GraphRfi / PGT4Rec 的约 0.20 提升到 **0.2990**。不过这里预测的是他们自己构造的 **proxy risk label**，不是实际 malicious user。

比较有意思的是 **Prompt Routing 实验其实不成功**。Beauty：

| Routing           | NDCG@10    |
| ----------------- | ---------- |
| **All-Simple**    | **0.0379** |
| All-Cautious      | 0.0360     |
| Random            | 0.0366     |
| Weak-Risk Routing | 0.0370     |

也就是说：

$\boxed{\text{All-Simple} > \text{Weak-Risk Routing}}$

甚至 Weak-Risk routing 跟随机 routing 的差异都没有明确证据。作者自己也承认，目前实验**不能证明 risk-guided routing 比统一 Simple Prompt 或随机 routing 更好**。

最后是我觉得和你最相关的 **Graph ablation**：

| Input            | H@10       | NDCG@10    |
| ---------------- | ---------- | ---------- |
| History only     | 0.0632     | 0.0365     |
| **Static Graph** | 0.0634     | **0.0384** |
| Temporal Graph   | **0.0642** | 0.0379     |

所以 **Graph Token 确实有用**，history-only → graph 后 NDCG 有提升。可是很关键的是：

$\boxed{\text{Temporal Graph 并没有稳定优于 Static Graph}}$

NDCG@10 甚至是 static graph 更高，只是 Temporal Graph 的 H@10 稍高。不过 Temporal Graph 在 **risk prediction** 上比 static graph 明显好：AUPRC 0.2631 vs 0.2313。

所以我会把这篇实验结论概括成：

> **真正被实验支持的是“加入 collaborative Graph Token 有帮助”；“Temporal 比 Static 更好”证据不足；最核心的 risk-aware prompt routing 反而没有证明有效。**
