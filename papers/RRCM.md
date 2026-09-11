# RRCM — Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2605.07129
- **Drive note:** https://docs.google.com/document/d/1cdckM8IHyouFadvwE9tnBZdTQNx1aCcNWxako1d_lNU/edit

## TLDR

ReAct Agent + 双Memory，使用GRPO训练



![2026-09-10_ RRCM](./imgs/2026-09-10_ RRCM.png)

## Motivation

1. 作者认为LLM没有CF信息，因为transformer没法显示学到CF信息
2. 如果把user/item所有信息塞入prompt，context就会爆炸；如果seq只放title，llm有可能不能理解这个item。



## Method

标准的 \[Think→Search→Observe→Think→Search→Observe→Answer\] 结构。再加两个memory。这里Agent能调用的工具只有两个memory



### 两个Memory

#### Collaborative Memory

就是user的history sequence。但是只有item title组成sequence



#### Meta Memory

就是item的metadata



### GRPO

按照上述的ReAct结构，采样多个trajectory，去做GRPO。
