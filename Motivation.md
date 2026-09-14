# MainLine

a. **模型依靠LLM进行打分的问题**

1. **LLM更适合listwise而不是pointwise的打分。LLM没法解释为什么给一个item打80分（未验证）**

   解决方案：

2. **如果给定了candidate，会不会出现顺序导致打分有偏的问题（未验证）**

   解决方案：



b. **Agent Memory更新相关问题**

1. **<u>简单的一个Memory会融合不同方面，尺度的信息，导致memory出现宽泛的问题（未验证）</u>**

​	解决方案：设置层级的memory



2. **<u>LLM未能准确更新memory的内容（有些memory带有权重，更新权重较为准确，但是权重很低时会替换memory内容，同样存在以上问题），导致更新后的memory越来越没有代表性，宽泛而没有重点</u>（未验证）**

   Reason：

   - 本质还是text的信息太稀疏了，没法像embedding一样融很多东西进去。最终解法就是c.1；

   - 另外emb的更新是 \[\boxed{\theta_{k+1}=\theta_k+\text{gradient from original observations}}\] 的，但是LLM是 \[\boxed{M_{t+1}=LLM(M_t,\text{new observation})}\] 的。这样LLM容易丢失信息。总结就是：<u>Embedding-based model 也会不断更新 representation，但这种更新是 loss-grounded 的；LLM memory evolution 则通常是 generation-grounded 的</u>。
   - 现有的模型基本都是使用短句/句子去做memory的，句子的信号就更稀疏了。

   Solution：

   - 使用单词/短语作为memory。方便concat，<u>压缩可能麻烦点</u>。



3. **未能充分利用数据集信息（包括未充分利用协同信息、历史交互）（未验证）**

​	解决方案：



4. **不同 LLM 之间能否直接迁移 memory/skill/policy（LLM生成，未验证）**

​	解决方案：



5. <u>**现在多数模型都不太考虑时序信息，或只使用长短期记忆这种弱时序信号（再叠加问题2），导致无法感知细粒度的长短期信息（未验证）**</u>

   TODO：查看一下CF的发展历程，当时CF发展到下一阶段是如何加上时序的？



c. **其他问题**

1. **如何在有限的token下给Frozen LLM更多的信息（LLM生成，未验证）**



# 其他探索

1. 长短期记忆真的偏好很大吗（未验证）
