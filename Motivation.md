# MainLine

a. **模型依靠LLM进行打分的问题**

1. **LLM更适合listwise而不是pointwise的打分。LLM没法解释为什么给一个item打80分（假说不成立）**

   详情：listwise的优点就是能够相互比较，最后根据结果排序。但当样本为hard-negative的时候，LLM不一定能通过大片的prompt分析出哪个更符合user，反而有可能相互影响。从而导致排序结果不如pointwise。**<u>但也不是绝对的</u>**

   解决方案：

2. **如果给定了candidate，会不会出现顺序导致打分有偏的问题（未验证）**

   解决方案：



b. **Agent Memory更新相关问题**

1. **<u>简单的一个Memory会融合不同方面，尺度的信息，导致memory出现宽泛的问题（未验证）</u>**

​	解决方案：设置层级的memory（<u>MARS</u>、<u>TierMem</u>）



2. **<u>LLM未能准确更新memory的内容（有些memory带有权重，更新权重较为准确，但是权重很低时会替换memory内容，同样存在以上问题），导致更新后的memory越来越同质化、没有代表性、宽泛而没有重点</u>（未验证）**

   Reason：

   - 本质还是text的信息太稀疏了，没法像embedding一样融很多东西进去。最终解法就是c.1；
   - 另外emb的更新是 \[\boxed{\theta_{k+1}=\theta_k+\text{gradient from original observations}}\] 的，但是LLM是 \[\boxed{M_{t+1}=LLM(M_t,\text{new observation})}\] 的。这样LLM容易丢失信息。总结就是：<u>Embedding-based model 也会不断更新 representation，但这种更新是 loss-grounded 的；LLM memory evolution 则通常是 generation-grounded 的</u>。
   - 现有的模型基本都是使用短句/句子去做memory的，句子的信号就更稀疏了。

   *可研究的问题节点：*

   - *上述第 2 点：text-based memory 的更新不是 loss-grounded 的。*

   Solution：

   - ❗️根据文章 <u>Useful Memories Become Faulty When Continuously Updated by LLMs</u>，能不能设置一个cache，保留用户一些近期序列信息放到一起进行压缩，这样能保留信息。（不知道有没有这样做的）**<u>实验结论：在所有简单方法中，这样的结果一般，Append效果要比改写好。</u>**
   - 使用单词/短语作为memory。方便concat，<u>压缩可能麻烦点</u>。**<u>实验结论：多个词汇词义相近，信息密度低。</u>**
   - <u>更新不使用LLM，直接在外部用小模型更新。查看新的item和history以及CF的相似度（FPMC）</u>

*<u>**什么是窄化，作时间序列图，最开始分散后面集中。**</u>*

*<u>**anchor**</u>*

**<u>Alpha edit</u>**

> **先证明自己的机制解释是对的，再证明基于这个解释设计的方法有效。**

\[\boxed{ \begin{aligned} &1.\ \text{Repeated LLM update causes memory narrowing.}\\ &2.\ \text{Define and measure what “narrowing” means.}\\ &3.\ \text{Identify which information is systematically lost.}\\ &4.\ \text{Show narrowing predicts recommendation degradation.}\\ &5.\ \text{Use controlled studies to isolate its cause.}\\ &6.\ \text{Derive a minimal intervention from that cause.}\\ &7.\ \text{Show the intervention improves MemRec / other Agentic Rec.} \end{aligned} }\]



先验证现象是否存在
- 最新的memory更新的rec模型改进分类

- 有了别人的经验设计一个实验框架。（其他相同类型工作的作者都认可的setting）
- 产物：一个方法/机制，不耦合于代码。像lora一样能够整合到其他模型当中。



下一步干啥：

1. 分析不同模型的内存更新偏好，以及每个内存更新的优缺点
2. 画一个memory迭代图
3. 



**Analysis-driven paper**总体的步骤是：

1. 先提出一个新的分析视角，并用理论推导 + 一系列机制实验验证它
2. 根据这个分析，解决方案非常直接，因此方法本身很小
3. 再证明这个小改动确实有效，而且具有一定通用性



3. **未能充分利用数据集信息（包括未充分利用协同信息、历史交互）（未验证）**

​	解决方案：



4. **不同 LLM 之间能否直接迁移 memory/skill/policy（LLM生成，未验证）**

​	解决方案：*<u>zhaoziyi</u>*



5. <u>**现在多数模型都不太考虑时序信息，或只使用长短期记忆这种弱时序信号（再叠加问题2），导致无法感知细粒度的长短期信息（未验证）**</u>

   TODO：查看一下CF的发展历程，当时CF发展到下一阶段是如何加上时序的：TimeSVD++ 在原始的CF矩阵上加一个马尔可夫链体现时序信息。**实验结论：没有参考价值。emb能直接相加，text没办法**
   
   



c. **其他问题**

1. **如何在有限的token下给Frozen LLM更多的信息（LLM生成，未验证）**

   Solution：

   - 用word替代sentence



现有的工作

| 工作                                    | 是否真正连续更新                                             | 是否开源                                                     | 对你研究的价值                                               |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **AgentCF, WWW 2024**                   | **是**。每轮 user/item interaction 后根据正负 item 和预测结果做 reflection，更新 user/item memory；论文明确写 `Collaborative Reflection and Memory Update`，而且 interaction/reflection 会迭代进行。([OpenReview](https://openreview.net/pdf?id=knAtuqBXDV&utm_source=chatgpt.com)) | **是**，官方代码公开。([GitHub](https://github.com/RUCAIBox/AgentCF?utm_source=chatgpt.com)) | **非常高。** 最标准的“旧 textual memory → LLM reflection → 新 textual memory”，非常适合研究 repeated rewriting 是否 narrowing。 |
| **i²Agent / iAgent, ACL Findings 2025** | **是，而且特别干净。** 用户 profile 满足 $F^T=M(F^{T-1},\text{new feedback},\dots)$，上一轮生成的 profile 直接进入下一轮，再根据最新正样本/feedback 重写。([ResearchGate](https://www.researchgate.net/publication/394269751_iAgent_LLM_Agent_as_a_Shield_between_User_and_Recommender_Systems?utm_source=chatgpt.com)) | **是**，代码和 InstructRec 数据都公开。([GitHub](https://github.com/WujiangXu/iAgent?utm_source=chatgpt.com)) | **我最推荐先做。** 它基本就是你想分析的 recursive user-profile rewriting，没有 collaborative graph 等复杂因素。 |
| **AgentCF++, SIGIR 2025**               | **是**。继承 AgentCF 的逐 interaction memory update，同时加入 single-domain / cross-domain user memory。代码中明确循环处理 interaction，读取旧 memory、LLM 生成新 memory、覆盖保存。([GitHub](https://github.com/jhliu0807/AgentCF-plus/blob/main/AgentCF%2B%2B.py)) | **是**。([GitHub](https://github.com/jhliu0807/AgentCF-plus?utm_source=chatgpt.com)) | **非常高。** 更妙的是代码已经有 `save_old_memory`、`save_new_memory`，并且每处理 10% interaction 就 `save_memory` 做 checkpoint，天然适合你做 memory trajectory analysis。([GitHub](https://github.com/jhliu0807/AgentCF-plus/blob/main/AgentCF%2B%2B.py)) |
| **MemRec, ACL 2026**                    | **是**。论文明确写 $M_u^t,M_i^t\leftarrow LM_{Mem}(M_u^{t-1},M_i^{t-1},\dots)$，用户每次与 item 交互后更新当前 user/item memory，同时向 neighbor propagation。([ResearchGate](https://www.researchgate.net/publication/399754091_MemRec_Collaborative_Memory-Augmented_Agentic_Recommender_System?utm_source=chatgpt.com)) | **是，完整 repo 已公开**，包含 memory manager、pruner、dataset conversion 和运行脚本。([GitHub](https://github.com/rutgerswiselab/memrec)) | **高，但适合第二阶段。** 它比前三个复杂，因为 narrowing 可能同时来自 self-rewrite 和 collaborative propagation。可以研究“isolated narrowing vs collaborative contamination”。 |
| **Agent4Rec, SIGIR 2024**               | **memory 会随模拟过程积累/反思**，包括 factual memory、emotional memory 和 reflection，但不是最纯粹的“不断重写同一个 preference profile”。([arXiv](https://arxiv.org/abs/2310.10108?utm_source=chatgpt.com)) | **是**。([GitHub](https://github.com/LehengTHU/Agent4Rec?utm_source=chatgpt.com)) | **辅助参考。** 更像 user simulator，适合作为不同 memory substrate 的对照，不建议作为你的第一个分析对象。 |



# 其他探索

1. 长短期记忆真的偏好很大吗（未验证）





做一个统一的东西：**LLM未能准确把历史信息转化为能够帮助Rec model进行决策的memory**
有两个原因：

1. llm更新导致信息熵下降

   解决方法：

   - 硬性：设立一个指标（如TF-IDF），强迫模型更新后的信息密度更大。
   - 软性：设计prompt让llm更新
   - 修改模型：分层/memory池

2. 现有的模型没有直接输入时序信息

   解决方法：

   - 软注入：每次新进入item就更新memory，越久的东西被总结的次数越多，越新的东西被总结的次数越少。
   - 硬注入：分层/显示使用item的时间信息写入memory。
   - 加在协同层：做一个时序图。



**sample amazon yelp 60 user<u>随机负样本</u>的实验**

做法：使用trainset每个item初始化memory。在test上listwise打分。

实验结论：

1. **<u>memrec</u>**好于**<u>memory为全历史总结+最近20个title</u>**好于**<u>全部商品名称</u>**好于**<u>逐个item改写</u>**好于<u>**group改写**</u>、**<u>双memory</u>**、**<u>多slot memory</u>**（不排除prompt原因）。因此随机负样本无法凸显memory的改进，<u>后续需要用hard-negative去做</u>。

2. 不同长度的history适合不同的方法。10和70len使用**<u>全历史总结+最近20个title</u>**最好。20和50len **<u>逐个item改写</u>**最好。100len用**<u>全部商品名称</u>**更好。
3. 使用TF-IDF分析：一个user的memory每个word，组成一个向量；把这个向量和其他59user计算取平均，得到**<u>一个用户间cosine分数</u>**。结果是**<u>NDCG和这个指标无关</u>**。
4. 在**<u>本数据集上</u>**，**<u>sample negative使用listwise打分更好；而 hard-negative上使用pointwise更好</u>**。（后续实验表明并不一定，hard也可能listwise稍强）

   ~~强行~~解释：pointwise每次只看到一个candidate，不会被其他candidate生成的模式所影响，让user memory主导评测。而在没那么强的negative sample上反而可以用listwise进行item之间的对比。
   
5. 使用no llm的基于分词匹配的方案效果远高于任何模型（包括MemRec、SAGER）
