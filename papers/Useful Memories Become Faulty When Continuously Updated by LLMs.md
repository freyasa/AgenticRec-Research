# Useful Memories Become Faulty When Continuously Updated by LLMs

- **Year:** 2026
- **Venue:** arXiv
- **Paper:** https://arxiv.org/abs/2605.12978

## TLDR

作者先发现了Memory越更新越差的现象，然后根据这个现象去做了多个实验。结论是：

1. 无论何种方式，memory越更新越差
2. 原始信息保留效果最好
3. 非要更新的话，可以把相同的类别送到一起保留

## Abstract

从过去的经验中学习，通常依赖两种互补形式的记忆：一种是**情景记忆（episodic traces）**，即记录“发生了什么”的原始轨迹；另一种是**整合后的抽象记忆（consolidated abstractions）**，它从大量 episode 中提炼出可复用的、类似 schema 的经验与规律。

近期的 agent memory 系统越来越关注后一种形式：使用 LLM 将过去的交互轨迹重写为一个文本形式的 memory bank，并随着新的交互不断更新这些记忆。这样的设计试图在**不更新模型参数**的情况下，让 agent 实现持续的自我改进。

然而，我们发现，即使这些整合后的记忆来源于本来有用的经验，当前 LLM 所生成的 consolidated memory 也经常存在错误。随着 consolidation 持续进行，memory 的效用通常会先上升，随后下降，甚至可能降低到**不使用 memory 的 baseline 以下**。

更令人意外的是，即使 consolidation 使用的是**真实正确的 ground-truth solutions**，GPT-5.4 在一组它原本能够在不使用 memory 的情况下正确解决的 ARC-AGI 问题上，仍然有 **54% 出现失败**。

我们的分析表明，这种性能退化主要来自 **consolidation 过程本身，而不是底层 experience 的质量**：对于完全相同的一组 trajectories，仅仅采用不同的 memory update schedule，就会得到质量上明显不同的 memory。与此同时，一个简单的 **episodic-only baseline**——仅仅保留这些原始 trajectories，而不进行抽象整合——其表现仍然能够与我们测试的 consolidation 方法保持竞争力。

在一个受控的 **ARC-AGI Stream** 环境中，我们允许 agent 执行三种 memory 操作：**Retain（保留）、Delete（删除）和 Consolidate（整合）**。结果发现，当 agent 可以自主选择 memory 操作时，它们默认倾向于保留原始 episode，并且准确率达到“强制每次进行 consolidation”的 agent 的大约 **2 倍**。进一步地，如果完全关闭 consolidation，仅进行 episodic memory 的管理，其性能与上述自动选择策略基本相当。

从实际系统设计的角度来看，一个更加鲁棒的 agent memory 系统应该把**原始 episode 当作一等公民式的证据（first-class evidence）保留下来**，并且显式地控制何时进行 consolidation，而不是在每一次 interaction 后都自动触发 consolidation。

展望未来，要实现真正可靠的 agentic memory，需要 LLM 具备一种新的能力：**在进行 memory consolidation 的同时，不覆盖或破坏这些抽象记忆所依赖的原始证据。**



## Motivation

现有 agent memory 默认认为“<u>把过去 experience 持续总结成更抽象的文本 memory</u>”会让 agent 越来越聪明，但作者怀疑这个前提其实没有被认真验证。



## Method

他做了**四组层层递进的实验**，目的都是在回答一个问题：

> **memory 变差，到底是因为 experience 不好，还是因为 consolidation / rewrite 本身有问题？**



### 实验一：memory持续更新会不会变坏

![2026-09-15_Useful Memories Become Faulty When Continuously Updated by LLMs_Fig1](<./imgs/2026-09-15_Useful Memories Become Faulty When Continuously Updated by LLMs_Fig1.png>)

图a：横轴是更新的step数。随着step增加，memory的用处先上升后下降。另外不同颜色代表最大保存memory条数，适中最好。

图b：横轴是使用多少个成功的trajectory更新memory（这个案例是所有的交互共享一组memory，而不是1user-1mem）。raw trajectory指的是把成功的路径原封不动存进去；AWM就是让LLM压缩原始的trajectory。结论：<u>如果持续更新和重写，可能越来越错；反而 raw trajectories 原样保留更稳定。</u>

右图：BT（Bad Trajectory）。用原始有错误的trajectory去inference。<u>结论：如果原始 trajectory 本身有很多错误，把这些 bad trajectories 再总结成 AWM，不仅没有“从经验中学会”，反而把性能从 16% 降到了 12%。</u>



### 实验二：优质数据多轮更新memory对结果的影响

![2026-09-15_Useful Memories Become Faulty When Continuously Updated by LLMs_Fig2](<./imgs/2026-09-15_Useful Memories Become Faulty When Continuously Updated by LLMs_Fig2.png>)



背景：先挑出 19 个 GPT-5.4 在完全不用 memory 时能 100% 做对的问题。然后每次 memory 更新时，作者都提供这些问题的 ground-truth solution

- **Static：代表把所有19个问题都处理完之后在去做memory的总结（consolidation）。**

- **Stream：代表每处理一个问题，就马上改一次 memory。**

- **R10、R50代表整个19个问题循环x轮之后的结果。**

**<u>结论：反复改写内存会导致结果变坏。（根本原因是现在的LLM改写机制不好，越写越丢失细节信息）</u>**



### 实验三：相同数据使用不同memory批处理策略

![2026-09-15_Useful Memories Become Faulty When Continuously Updated by LLMs_Fig3](<./imgs/2026-09-15_Useful Memories Become Faulty When Continuously Updated by LLMs_Fig3.png>)

背景：把不同的任务中每一条 trajectory 单独转换成一个 structured summary/abstract，里面有 `contextual_description`、`refined_trajectory`、`strategic_guidelines` 三部分。

- No Memory：直接让LLM去inference

- Summary：每个history item分别总结，再concat。
- Grouped：按 任务类别分组，<u>每组内部再 总结（consolidation），最后把各组结果拼起来</u>
- Ungrp.：所有history一次性总结
- Stream：每次都有不同任务的一部分trajectory，每次都和原始的一个memory做更新

结论：越频繁更新结果就越差，更新的时候越无序就越差



### 实验四：在最好的memory上进行stream更新

![2026-09-15_Useful Memories Become Faulty When Continuously Updated by LLMs_Fig4](<./imgs/2026-09-15_Useful Memories Become Faulty When Continuously Updated by LLMs_Fig4.png>)

背景：先用 GPT-5.4 和 ALFWorld 的成功 trajectories，按照前面效果较好的 Static-Group 方式，**构建一个高质量的初始 memory**：然后不重新从零构建，而是拿 GPT-5-nano 继续在同一个 trajectory pool 上一批批**<u>使用Stream更新</u>**它

- **Homogeneous batch**：一个 batch 里的 trajectories 来自同一种 task family
- **Heterogeneous batch**：一个 batch 里混不同 task

结论：只要是经常更新，尽管最初有一个好的memory最终也会坏。



### ARC-AGI Stream（本章由ai生成，想知道细节可以到时候再问ai）

作者构造了一个可控的 memory-management 环境，用于分析 continual consolidation 到底哪里出问题。

Memory 被拆成两部分：

- **Episodic memory**：保留原始 problem–solution episode
- **Abstract memory**：由多个 episode consolidation 得到的抽象规则

每来一个新 episode，agent 可以执行：

- **Retain**：保留原始 episode
- **Delete**：删除 episode
- **Consolidate**：将 episode 总结成 abstract memory

主要比较：

- **Force**：每次都强制 consolidation
- **Auto**：agent 自己决定 Retain / Delete / Consolidate
- **Episodic-only**：只允许 Retain / Delete，不做 consolidation
- **Abstract-only**：只使用 abstract memory

主要结论：

1. <u>**Auto 明显优于 Force**：说明每步强制 consolidation 不可靠。</u>
2. <u>Auto 模式下，agent 更倾向于 **保留 episodic memory，而不是频繁 consolidation**。</u>
3. <u>**Episodic-only ≈ Auto**，说明主要有效信息来自原始 episode。</u>
4. <u>**Abstract-only 接近 no-memory baseline**，表明 consolidation 后的 abstraction 本身并没有稳定保留有效信息。</u>

**Takeaway：**
当前 LLM memory system 中，raw episodic evidence 比持续重写的 abstract memory 更可靠；consolidation 应该按需触发，而不是每一步都执行。
