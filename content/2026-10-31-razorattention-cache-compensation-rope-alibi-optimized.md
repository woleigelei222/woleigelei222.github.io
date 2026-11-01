---
title: "RazorAttention 精读续篇：分头缓存、补偿 Token 与 RoPE/ALiBi"
slug: "2026-10-31-razorattention-cache-compensation-rope-alibi"
kind: "paper"
date: "2026-10-31"
summary: "复盘 RazorAttention 的预先 Head 诊断、KV 缓存分配、GQA 约束、Compensation Token 数量修正，以及 ALiBi 和 RoPE 不同的检索头识别依据。"
tags: ["论文精读", "RazorAttention", "KV Cache", "Attention", "RoPE", "ALiBi"]
featured: false
published: true
example: false
---

# RazorAttention 精读续篇：从 Head 识别到真正压缩 KV Cache

> **学习时间：2026-10-31。** 本文依据当天学习记录重新组织，保留原有研究重点。下述数值演算为教学示例，不是本人已完成的模型实验。涉及论文实现的细节应以原文及代码为最终依据。

## 01｜延续昨天的问题：识别并不是每个 Decode Step 都重新运行

在前一天的学习中，我梳理了 RazorAttention 如何使用重复的随机 Token 序列，计算 Echo 与 Induction 特征分数，并筛选 Retrieval Heads。今天首先明确两点：

1. **Head 识别是预先进行的行为诊断。** 正常生成时复用已识别的 Head 分类，不需要每个新 Query 到来就重新运行整套重复序列测试。
2. **Echo/Induction 得分是代理指标，而不是正确率保证。** 一个 Head 在合成重复序列上具有强 Induction 行为，不代表它能在所有现实任务中可靠检索远距离信息。

一个容易混淆的层次是：计算分数的原始数据来自 **Query 与历史 Key 的 Attention 权重**，但最终被诊断和分类的对象是 **Head**，并不是给当前 Query 本身贴标签。

## 02｜Retrieval 与 Non-retrieval：缓存保留规则

识别结束后，作者对两类 Head 采取差异化缓存策略。

| Head 类型 | 对历史 KV 的处理 | 设计目的 |
|---|---|---|
| Retrieval Head | 保留完整历史 K/V | 支持远距离信息检索 |
| Non-retrieval Head | 保留开头 Sink、最近窗口，并构造补偿 KV | 降低存储开销，缓解直接截断造成的误差 |

对于 Non-retrieval Head，笔记记录的最近窗口为：

\[
L_h=\max(4000,N/5)
\]

其中 \(N\) 是序列长度。这是特定配置下的规则，而不是任何输入长度都可以超额保留 4000 个 Token；短序列时受实际长度约束。开头还保留 4 个 Sink Tokens。

### 手算：10,000 Token 时存储了什么？

设 \(N=10000\)，则

\[
L_h=\max(4000,2000)=4000.
\]

假设 Sink 与 Recent 不重叠，缓存分区为：

- Sink：位置 1～4，共 4 个；
- 待淘汰历史：位置 5～6000，共 \(N_d=5996\) 个；
- Recent：位置 6001～10000，共 4000 个。

作者将中间待淘汰部分聚合成 **一个补偿 K/V 条目**。因此仅按 KV 条目数量计算，压缩后存储约为

\[
4+1+4000=4005.
\]

对应这一个 Non-retrieval KV 缓存的条目减少比例为

\[
1-4005/10000=59.95\%.
\]

**边界：** 这不是整模型的显存压缩率；整模型还包括 Retrieval Heads、GQA 共享、元数据等成本。

## 03｜GQA：为什么某个 Query Head 重要，会影响整组 KV？

在 Grouped-Query Attention（GQA）中，多个 Query Heads 共享同一个 KV Head。例如：

\[
Q_0,Q_1,Q_2,Q_3\longrightarrow KV_0.
\]

如果诊断结果为：

| Query Head | 诊断分类 |
|---|---|
| Q0 | Non-retrieval |
| Q1 | Non-retrieval |
| Q2 | Retrieval |
| Q3 | Non-retrieval |

那么不能因为多数 Query Heads 是 Non-retrieval，就直接淘汰共享 \(KV_0\) 的中间历史：**Q2 仍要借助同一套 KV 访问完整历史。** 因而整组共享 KV 需要受到保护。

这里的逻辑是实际缓存资源按 **KV Head** 共享，而不是对每个 Q Head 建立独立的 Key/Value 存储。

## 04｜Compensation Token：均值只来自被淘汰的 KV

如果简单删除 Non-retrieval Head 的全部中间历史，它对模型输出的影响也会随之消失。因此论文引入 Compensation Token，对待淘汰部分的 Key、Value **分别求均值**：

\[
\hat k=\frac{1}{N_d}\sum_{m\in D}k_m,
\qquad
\hat v=\frac{1}{N_d}\sum_{m\in D}v_m.
\]

这里 \(D\) 仅表示被淘汰的位置集合，**不包含完整保留的 Sink 与 Recent**。

### 一组二维 KV 的手算

\[
\begin{aligned}
k_1&=[1,3],&v_1&=[2,0],\\
k_2&=[3,5],&v_2&=[4,2],\\
k_3&=[5,7],&v_3&=[6,4].
\end{aligned}
\]

于是

\[
\hat k=\left[\frac{1+3+5}{3},\frac{3+5+7}{3}\right]=[3,5],
\]

\[
\hat v=\left[\frac{2+4+6}{3},\frac{0+2+4}{3}\right]=[4,2].
\]

原本三个不同历史位置的 K/V，现在用 \(([3,5],[4,2])\) 这个代表性条目近似保存；**均值并不保留每个位置的差异**。

## 05｜为什么均值之后还要乘被淘汰数量 \(N_d\)？

如果只把补偿条目当作一个普通 Token 参与 Softmax，原来 \(N_d\) 个历史位置的整体注意力质量可能被低估。论文采用的数量修正思想可写成（省略或吸收标准 Attention 缩放因子）：

\[
\hat O\approx
\frac{N_d e^{q\hat k^\top}\hat v+\sum_{j\in R} e^{qk_j^\top}v_j}
{N_d e^{q\hat k^\top}+\sum_{j\in R} e^{qk_j^\top}},
\]

其中 \(R\) 是完整保留的位置集合。也就是说，补偿条目的 Softmax 指数权重变成 \(N_de^{q\hat k^\top}\)，等价于把它的 Logit 增加 \(\log N_d\)。

### 极简手算：不乘和乘 \(N_d\) 差多少？

假设被压缩的三个 Value 均为 4，另一个完整保留 Value 为 10，四个 Token 的原始 Attention Logit 都为 0。

**未压缩：**

\[
O_{full}=\frac{4+4+4+10}{4}=5.5.
\]

**压缩成平均值但不计数量：**

\[
O_{no\ count}=\frac{4+10}{2}=7.
\]

**压缩且修正 \(N_d=3\)：**

\[
O_{count}=\frac{3\times4+1\times10}{3+1}=5.5.
\]

这个特例能够恢复原输出，是因为被压缩部分分数相同且 Value 也相同。**一般情况下修正数量不意味着无损恢复**：如果原本某一 Key 与 Query 特别匹配，均值仍可能抹掉其检索特征。

## 06｜为什么论文要分开讨论 ALiBi 与 RoPE？

两者都引入位置相关信息，却不以同一种方式改变 Attention Score。

### ALiBi：显式距离惩罚

对于当前 Query 位于 \(m\)、历史 Key 位于 \(n\)，可写成：

\[
S_{m\to n}=q_mk_n^\top-l_h(m-n),\qquad m\ge n.
\]

\(l_h\) 是 Head 相关的距离惩罚斜率。**只考虑内容分数相同时**，远距离位置会承受更大的负偏置。论文在有界性等假设下推导与斜率相关的有效关注范围：一般而言 \(l_h\) 越大，这一范围越短。

不能将“距离惩罚”误解为任意两条真实 Token 的 Attention 权重必然随距离严格单调减少：它们的内容相关性也不同。单个位置 Attention 权重的上界也不自动等于整体输出误差上界。

### RoPE：旋转 Q/K，不直接加线性距离惩罚

RoPE 根据 Token 位置对 Q、K 实施旋转，可抽象为

\[
q_m=R_mq,\qquad k_n=R_nk.
\]

RoPE 的点积依赖相对旋转角度，但**没有 ALiBi 那样的 \(-l_h(m-n)\) 线性扣分项**。一个简化二维例子中，相关性可随角度呈现余弦式振荡，因此不能直接套用 ALiBi 的有效范围证明。

对于 RoPE 模型，RazorAttention 因而采用此前学习过的 **Echo / Induction 行为诊断**来识别应完整保留 KV 的 Heads；这是经验性诊断，并不是 ALiBi 定理在 RoPE 上的推论。

## 07｜这次复盘带来的理解变化

此前容易把 KV Cache 压缩理解成“把注意力分数低的旧 Token 删除”。今天进一步认识到，RazorAttention 至少有三个相互联系但功能不同的决策层次：

1. **Head 层：** 先确定哪些 Heads 需要完整历史；
2. **缓存层：** 非检索 Head 用 Sink、Recent 和补偿条目减少存储；
3. **Attention 层：** 通过 \(N_d\) 修正补偿条目的数量权重，降低把多条历史缩成一条带来的偏差。

同时，GQA 决定了实际缓存保护必须尊重共享关系；ALiBi 与 RoPE 的不同位置编码机制也决定了 Head 识别的理论依据不能混用。

## 08｜尚待验证与下一步

本次记录的是理论阅读及手算梳理，**未记录代码复现、模型实测或加速结果**。后续可以核对论文中 ALiBi 有效关注范围的定理假设与完整符号定义，并比较 Compensation Token 在非均匀 Attention 分数下的输出近似误差。进一步阅读其他 KV 压缩论文时，应持续区分 **Head 识别、Token 选择和预算分配** 三个层面的优化。

*整理说明：本篇以个人学习记录为主体，公式排版、示例展开与语言结构经 AI 辅助整理；手算是解释性演示，不代表实际复现实验。*
