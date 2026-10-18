---
title: "从 GQA 到语义检索头：CompressKV 的 Token 选择机制学习记录"
slug: "compresskv-srh-token-selection-2026-10-18"
kind: "paper"
date: "2026-10-18"
summary: "初读 CompressKV，理解 GQA 中共享 KV 的淘汰冲突、Streaming Head 与 Retrieval Head 的差异，以及 SRH 识别与重要 Token 选择流程。"
tags: ["边缘计算", "CompressKV", "GQA", "Retrieval Head", "Semantic Retrieval Head", "KV Cache", "论文精读"]
featured: false
published: true
example: false
---

# 从 GQA 到语义检索头：CompressKV 的 Token 选择机制学习记录

## 今天学了什么？

今天开始阅读新论文 **CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference**。

此前我主要阅读 KV Cache 的分层存储、数据传输和近存储评分，而这篇论文把问题进一步转向：**在压缩 KV Cache 时，怎样避免把长文本中间真正重要的证据一起删除？**

## 一、先理解 GQA：共享 KV 为什么会影响淘汰决策？

在普通多头注意力（MHA）中，每个 Query Head 通常有自己对应的 K/V Head。GQA（Grouped-Query Attention）则允许**多个 Query Heads 共享一个 KV Head**，从而降低 KV Cache 存储与读取带宽需求。

例如，在 8 个 Query Heads 的教学配置中，可以让 Q1～Q4 共用 KV Group 1，Q5～Q8 共用 KV Group 2。这样 Q Head 数量仍为 8，但 KV Head 数量只有 2。

共享之后出现一个问题：**不同 Query Heads 可以关注不同的历史 token，但它们所共享的 KV 缓存保留决策会影响同组 Head 的访问。**

### 1.1 简单聚合为什么可能淘汰关键证据？

沿用今天的教学示例，某个 GQA 相关组中有三个 Heads，对四个历史位置评分如下：

| Head | t1：开头 | t2：背景 | t3：重要证据 | t4：结尾 |
|---|---:|---:|---:|---:|
| Streaming Head 1 | 8 | 1 | 2 | 8 |
| Streaming Head 2 | 9 | 1 | 1 | 9 |
| Retrieval-like Head | 1 | 1 | 10 | 1 |
| **简单求和** | **18** | **3** | **13** | **18** |

如果缓存预算只允许保留两个位置，简单求和会优先选择 t1 和 t4，而把包含关键证据的 t3 淘汰。

这组分数是教学假设，用来说明 **Streaming Heads 的聚合分数可能掩盖少数检索头的信号**；并非 CompressKV 原文实验的注意力数据。

## 二、Streaming Head、Retrieval Head 与 SRH 的区别

- **Streaming Head（SH）**：常表现为对序列开头和末尾较高的关注。
- **Retrieval Head（RH）**：表现出从上下文中检索相关信息的能力。传统识别方式往往重视正确答案 token 是否出现在注意力 Top-1 或 Top-k 位置。
- **Semantic Retrieval Head（SRH）**：CompressKV 提出的识别标准，不只考虑单个词上的尖锐峰值，还统计对**完整标注答案片段**的注意力总量。

例如正确答案由两个 token 组成，某个 Head 可能分别给它们 0.25 和 0.20 的权重，却没有让任何一个位置成为全局 Top-1。严格的 Top-1 检索标准可能忽略这个 Head，但答案片段总注意力已经达到 0.45。

要谨慎理解：**SRH 并不是自动把所有语义相关的邻近词都当作答案**。论文的评分利用标注的答案 token 片段，在正确答案生成步骤中进行统计。

## 三、SRH 公式：计算的到底是什么？

论文给出的识别评分为：

\[
S_{\mathrm{SRH}}(h)=\sum_{t=1}^{N}
\mathbf{1}\{y_t\in\mathcal A\}
\sum_{j\in\mathcal A}a_{t,j}^{(h)}
\]

其中：

- \(h\)：待评估的 Query Attention Head。
- \(t\)：生成步骤；\(y_t\)：当前生成的 token。
- \(\mathcal A\)：标注的答案 token 片段。
- \(a_{t,j}^{(h)}\)：Head \(h\) 在生成步骤 \(t\) 对答案位置 \(j\) 的注意力权重。
- \(\mathbf 1\)：判断该生成步骤是否符合答案条件的指示函数。

我的理解是：**内层求和计算一个生成步骤对整个答案片段的关注总量；指示函数筛选正确答案相关步骤；外层求和把这些步骤的贡献累计为该 Head 的评分。**

特别注意：**这个公式首先输出的是 Head 的评分，不是直接输出 KV Cache，也不是直接选择 token。**

## 四、从选 Head 到实际保留 KV：两层选择不要混淆

在这篇论文中，有两个先后发生的过程。

### 4.1 离线选择 SRH

首先使用校准数据，为不同层的 Q Heads 计算 SRH 分数。论文实验为每层选择 **Top-4 SRH**，得到用于指导重要性评分的 Head 集合。

例如：

```text
全部 Query Heads
      ↓ SRH 离线评分
选出 H2、H4、H5、H7
```

**这里选到的是 Head，而不是 4 个历史 token，也不是 4 组独立的 Q/K/V。** GQA 中多个 Q Heads 本来就可能共用同一组 K/V。

### 4.2 在线对历史 token 评分并保留 KV

在具体输入的 Prefill 阶段，作者借鉴 SnapKV 的思路，观察 Prompt 尾部的 **Observation Window**。例如假设 Prompt 有 1000 个 token，观察最后 8 个 Query 位置，则可对某个历史位置 \(j\) 汇总其得到的注意力：

\[
s_j^{(h)}=\sum_{t=993}^{1000}a_{t,j}^{(h)}
\]

这个表达式用于说明窗口聚合；实际流程还会在 token 维度进行池化。

随后，系统对选出的 SRH 的评分向量取平均，得到各历史 token 的重要性排序，并按照**该层实际 KV Budget** 选 Top-N 个 token。最终保留它们对应的 Key 和 Value，淘汰其余不在集合内的历史 KV。

```text
离线：全部 Q Heads → Top-4 SRH
在线：选中 SRH 的窗口注意力
          ↓
       窗口求和与池化
          ↓
       Head 间平均评分
          ↓
       按当前层 Budget 选 Top-N token
          ↓
       保留这些 token 的 K/V
```

论文采用层内统一的 token 位置集合，便于 GQA 场景中的共享 KV 管理。**未被选为 SRH 的 Q Heads 仍可正常参与 Attention；SRH 选择只决定谁指导缓存淘汰。**

## 五、今天的理解变化

**原本的疑问：** 既然模型已经计算 Attention，为什么直接汇总所有 Head 分数还可能出错？SRH 评分找出来的到底是 Head 还是 KV？

**现在的理解：** 不同 Head 承担的功能不同，统一聚合可能掩盖少数检索头对中间证据的关注。CompressKV 先离线选出有用的评分者，再利用它们针对当前输入选择需要保留的历史 token。**“选 Head”与“选 token/KV”不是同一层次。**

**理解发生变化的原因：** 在 GQA 下，Q Head 的数量和 KV Head 的数量不同。对共享 KV 作淘汰决定时，更需要考虑哪些 Q Heads 的注意力行为真正有助于发现重要证据，而不能默认 Head 的贡献完全相同。

## 六、仍待进一步学习

今天重点理解了 CompressKV 的**第一项创新：SRH 识别与 token 选择**。以下内容仍需要继续阅读，不作为今天已经掌握或验证的成果：

1. 论文如何基于各层 Attention 输出误差分配不同 KV Cache 预算？
2. 为什么每层选择 Top-4 SRH，而不是更多或更少？
3. 在长答案、多轮对话或输入分布变化时，离线识别的 SRH 是否稳定？
4. 如何通过代码对比全 Head 聚合与 SRH 引导选择的差异？

**未记录个人代码复现、实验数据或模型训练。** 下一次可以优先学习误差感知的 Layer-Adaptive Cache Allocation，并用一个多层预算分配的例子手算。

> **今日结论：** KV Cache 压缩不只是“留下多少 token”的问题，还需要先确定“由哪些 Head 判断 token 重要性”，防止关键证据在不恰当的评分聚合中被丢弃。
