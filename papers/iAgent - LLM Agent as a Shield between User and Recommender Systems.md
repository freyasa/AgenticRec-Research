# iAgent: LLM Agent as a Shield between User and Recommender Systems

- **Year:** 2025
- **Venue:** ACL 2025 Findings
- **Paper:** https://arxiv.org/abs/2502.14662 / https://aclanthology.org/2025.findings-acl.928.pdf
- **Code:** https://github.com/WujiangXu/iAgent



## TLDR

![2026-09-21_iAgent](./imgs/2026-09-21_iAgent.png)

## Abstract

传统推荐系统通常采用“用户—平台”范式，即用户直接暴露在平台推荐算法的控制之下。然而，这种范式可能使用户处于较为被动的位置。一方面，许多推荐模型主要围绕平台的商业目标进行设计，可能无法充分保护和捕捉用户的真实兴趣；另一方面，这些模型通常基于所有用户的数据进行优化，因此可能忽略个体用户的独特偏好。

因此，传统推荐系统可能带来用户缺乏控制、平台操纵、回音室效应，以及低活跃用户个性化不足等问题。已有一些工作使用 LLM Agent 模拟用户行为，但主要仍然服务于提升平台侧的推荐性能。

为此，本文提出一种新的 **User-Agent-Platform** 范式，让 Agent 作为用户与推荐系统之间的“保护层”。作者首先构建了四个带有用户自然语言指令的推荐数据集 **InstructRec**。随后提出 **Instruction-aware Agent（iAgent）**，通过理解用户指令，并调用工具从外部环境获取相关知识。进一步提出 **Individual Instruction-aware Agent（i²Agent）**，加入动态记忆机制，根据个体用户的反馈持续进行个性化。

实验表明，i²Agent 在四个数据集的排序指标上相较 SOTA baseline 平均提升 16.6%，同时能够缓解回音室效应以及低活跃用户所受到的模型偏差。



## Motivation

不是“现有 memory 更新不好”，而是“现有推荐由平台主导，缺少真正站在单个用户侧、理解当前显式需求并根据个人反馈自适应的 agent”。（感觉没什么说服力）

## Method

\[iAgent\]是包含外部工具的train-free的Agentic Rec。\[i^2Agent\]是在前者的基础上加了一个user的memory。



1. 用户需要输入一个陈述，LLM把这个陈述翻译成一段描述 ，LLM通过这个instruction判断是否需要 tool → 单次外部检索
2. history + candidate + knowledge (+ memory) 进行排序
3. history + old profile + pos/neg → prediction → reveal GT/feedback → update profile**（这个只有\[i^2Agent\]才有）**



**这里的外部工具指的是Google Search API**，search的内容是生成的instruction（<u>instruction是用户的目的，也就相当于看看网上相同目的的用户想要什么东西，拿回来做个参考</u>）。

**<u>这个数据没有用户输入的陈述，是通过gpt-4omini把ground-truth的用户评价反向写成一个陈述的。</u>**感觉有点数据泄露的嫌疑在。
