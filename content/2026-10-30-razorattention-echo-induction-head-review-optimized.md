---
title: "10 月 30 日学习记录：RazorAttention 的 Echo Head、Induction Head 与免训练检索头识别"
slug: "2026-10-30-razorattention-echo-induction-head-review"
kind: "paper"
date: "2026-10-30"
summary: "精读 RazorAttention 的研究动机与 Echo/Induction Head 识别方法，通过重复随机 token 的教学评分示例理解如何免训练选择重要 Heads，并对比 DuoAttention。"
tags: ["论文精读", "RazorAttention", "KV Cache", "Retrieval Head", "Echo Head", "Induction Head", "ICLR 2025"]
featured: false
published: true
example: false
---

# 10 月 30 日学习记录：RazorAttention 的免训练 Head 识别

> 本文根据 10 月 30 日原始学习记录整理。当天重点是 **RazorAttention 的研究问题、Echo/Induction Head 的含义以及 Head 得分与筛选过程**。文中的 10-Head 手算为辅助理解的**教学模拟**，不是模型实际输出。

## 一、为什么在 DuoAttention 之后读 RazorAttention？

今天开始阅读 **RazorAttention: Efficient KV Cache Compression Through Retrieval Heads**，这篇论文发表于 **ICLR 2025**，作者团队来自 Huawei Technologies。

它提出的问题与之前的 DuoAttention 很接近：**是否可以减少长上下文 KV Cache，同时尽量保留未来未知问题所需的远距离检索能力？**

两篇论文对“哪些 Head 需要完整历史”给出了不同答案：

- **DuoAttention：** 构造 Passkey 长距离检索任务，通过隐藏状态蒸馏与 L1 正则训练门控 \(\alpha\)，再划分 Retrieval/Streaming Heads。
- **RazorAttention：** 构造重复随机 token 序列，直接统计 Head 的 Echo/Induction 行为，通过打分排序选出 Retrieval Heads，**不需要再训练原模型或额外门控参数**。

我今天重点理解的是第二条路线：**怎样利用重复序列，给每个 Head 打出可比较的分数？**

---

## 二、首先分清 Echo Head 与 Induction Head

### 2.1 Echo Head：找历史中相同的 token

假设模型处理到：

```text
位置：1  2  3  4  5
Token A  B  C  A  B
                 ↑ 当前 B
```

Echo Head 倾向于让当前的 \(B_5\) 关注历史中相同的 \(B_2\)：

\[
\boxed{B_5\longrightarrow B_2}
\]

它更像是在回答“我之前在哪里见过同一个 token？”

### 2.2 Induction Head：关注历史重复模式中的后继信息

仍然使用：

```text
历史模式： A₁ → B₂ → C₃
当前模式： A₄ → B₅ → ?
```

如果模型想借助历史模式预测下一个 token，就可以让当前位置的 \(B_5\) 关注历史中 \(B_2\) **后面的** \(C_3\)：

\[
\boxed{B_5\longrightarrow C_3}
\]

这反映的是“之前出现过 B，B 后面跟着什么？”

| 特征 | Echo Head | Induction Head |
|---|---|---|
| 当前 token | B | B |
| 关注位置 | 历史 B | 历史 B 的后继 C |
| 教学例子 | \(B_5\to B_2\) | \(B_5\to C_3\) |
| 直观用途 | 重复匹配 | 序列后继模式补全 |

**这里需要纠正容易混淆的一点：** Induction 的目标不是“当前 B 前面的某个位置”，而是论文定义下历史中与当前 token 匹配的那个 token **紧接着的后继位置**。这是一种对 Attention 行为的诊断，不代表模型只靠一个 Head 就一定复制出 C。

---

## 三、作者如何构造免训练诊断数据？

论文 Section 3.3 使用的办法是：

1. 生成一段随机 token 序列，例如 \(K=2500\) 个 token；
2. 将这段序列重复 **4 次**，形成可预测的重复结构；
3. 让原本训练好的模型正常做前向计算；
4. 根据每个 Head 在指定位置的 Attention 权重，统计 Echo Score 和 Induction Score；
5. 分别排序，选出需要保护的 Heads。

由于输入是随机 token，能减少自然语言语义关系对观察结果的干扰；重复结构则让 Echo 和 Induction 有明确的目标位置。

**免训练不等于不做计算。** 这里仍需运行模型前向、收集 Attention 权重、统计分数，但不进行反向传播，也不更新 \(W_Q,W_K,W_V\) 等模型权重。

---

## 四、手算示例：模拟 H0～H9 的评分全过程

> **全部为教学数据。** 为便于人工计算，将原论文的长序列简化为四个 token 的重复片段，并只使用两个 Query 位置评分。真实论文采用更长、更充分的统计。

### 4.1 构造输入

\[
S=[A,B,C,D]
\]

在此演示里把它重复 3 次：

```text
位置： 1  2  3  4 | 5  6  7  8 | 9  10 11 12
Token：A  B  C  D | A  B  C  D | A   B  C  D
```

只看第三段开头两个位置：\(t=9\) 的 A 和 \(t=10\) 的 B。

| 当前 Query 位置 | Echo 目标 | Induction 目标 |
|---|---|---|
| \(t=9\)，当前 A | 位置 5 的 A | 位置 6 的 B |
| \(t=10\)，当前 B | 位置 6 的 B | 位置 7 的 C |

因此，需要从每个 Head 的 Attention 矩阵中提取：

\[
A_{9,5}^{(h)},\quad A_{9,6}^{(h)},\quad A_{10,6}^{(h)},\quad A_{10,7}^{(h)}
\]

### 4.2 H0 的具体计算

假设 H0 的四个 Attention 权重是：

| Query 位置 | Echo 权重 | Induction 权重 |
|---|---:|---:|
| 9 | \(A_{9,5}=0.04\) | \(A_{9,6}=0.88\) |
| 10 | \(A_{10,6}=0.06\) | \(A_{10,7}=0.94\) |

本教学示例采用两个位置的平均值：

\[
S_{echo}^{(H0)}=\frac{0.04+0.06}{2}=\boxed{0.05}
\]

\[
S_{ind}^{(H0)}=\frac{0.88+0.94}{2}=\boxed{0.91}
\]

这个结果说明：**在人工设定的两次观测中，H0 明显偏向 Induction 目标位置。**

### 4.3 H2 的具体计算

假设 H2 的观测权重为：

| Query 位置 | Echo 权重 | Induction 权重 |
|---|---:|---:|
| 9 | 0.88 | 0.03 |
| 10 | 0.92 | 0.02 |

得到：

\[
S_{echo}^{(H2)}=\frac{0.88+0.92}{2}=\boxed{0.90}
\]

\[
S_{ind}^{(H2)}=\frac{0.03+0.02}{2}=\boxed{0.025}
\]

因此 H2 的 Echo 特征更强。

### 4.4 汇总十个 Head 的模拟得分

| Head | Echo Score | Induction Score | 观察 |
|---|---:|---:|---|
| H0 | 0.05 | **0.91** | Induction 特征强 |
| H1 | 0.08 | **0.83** | Induction 特征强 |
| H2 | **0.90** | 0.025 | Echo 特征强 |
| H3 | 0.19 | 0.22 | 两者不突出 |
| H4 | 0.11 | 0.40 | 有一定 Induction 偏向 |
| H5 | 0.17 | 0.34 | 有一定 Induction 偏向 |
| H6 | 0.15 | 0.12 | 两者不突出 |
| H7 | 0.08 | 0.50 | 有一定 Induction 偏向 |
| H8 | 0.14 | 0.26 | 两者不突出 |
| H9 | 0.12 | 0.09 | 两者不突出 |

本示例每一个 Query 位置的 Echo 和 Induction 权重之和均不超过 1，其余权重留给其他可访问位置，因此不违反 Attention Softmax 的基本归一化要求。

**重要边界：** 上述“指定两个位置后取平均”的公式是教学化处理，目的是解释论文所说的统计流程，并非作者公布的完整评分实现。

### 4.5 排序与选 Head

按照 Induction Score 排序：

\[
H0(0.91)>H1(0.83)>H7(0.50)>\cdots
\]

按照 Echo Score 排序：

\[
H2(0.90)>H3(0.19)>H5(0.17)>\cdots
\]

为避免 10 个 Head 按百分比选取时的取整问题，**教学示例规定 Induction Top 2 + Echo Top 1**：

\[
\mathcal H_I=\{H0,H1\},\qquad \mathcal H_E=\{H2\}
\]

取并集：

\[
\boxed{\mathcal H_R=\mathcal H_I\cup\mathcal H_E=\{H0,H1,H2\}}
\]

最终 H0、H1、H2 为本次模拟的 Retrieval Heads；H3～H9 作为 Non-Retrieval Heads。

注意：**这不是每来一个 Query 就做一次的 Top-k token 筛选，而是对模型中哪些 Head 应当长期保留完整缓存的一次诊断分类。**

---

## 五、论文实际使用怎样的 Head 保护策略？

论文 Table 2 给出的常用配置是：

| 参数 | 论文配置 | 作用 |
|---|---:|---|
| Induction Head | Top 14% | 完整保护 Induction 得分最高的 Heads |
| Echo Head | Top 1% | 完整保护 Echo 得分最高的 Heads |
| Sink Tokens | 4 | 保留序列最前面的位置 |
| Non-Retrieval 缓存窗口 | \(\max(4000,N/5)\) | 保存最近一段 KV |

**14% + 1% 不是严格保证最终恰好保护 15% 的物理 KV Heads。** 两类集合可能重叠；对 GQA 模型，还需要考虑多个 Query Heads 共享一个 KV Head 的情况。论文说明，只要共享组中有一个 Query Head 被划为 Retrieval，就需要保护相应的共享 KV 缓存。

### 5.1 分类完成以后怎么处理缓存？

| Head 类型 | KV 管理方式 |
|---|---|
| Retrieval Head | 继续保留完整历史 KV |
| Non-Retrieval Head | 保存 Sink、Recent Window，并使用 Compensation Token 表示被删历史的一部分信息 |

作者并非简单地认定非检索 Head 的远程历史完全没用；因此后续还提出了 **Compensation Token**，弥补直接丢弃远程 KV 的信息损失。

---

## 六、和昨天 DuoAttention 的区别

| 对比问题 | DuoAttention | RazorAttention |
|---|---|---|
| 如何构造识别任务？ | 合成 Passkey 长距离召回 | 随机 token 序列重复 |
| 如何判断 Head 的作用？ | 训练门控 \(\alpha\)，比较隐藏状态 | 统计 Echo/Induction Attention 权重 |
| 需要反向传播吗？ | 需要，训练门控 | 不需要 |
| 如何部署？ | 阈值二值化后划分 Full/Streaming | 根据评分排名保护 Retrieval Heads |
| 非检索 Head 的缓存 | Sink + Recent | Sink + Recent + Compensation Token |

我现在的理解是：**DuoAttention 更接近“任务输出敏感性驱动的分类”，RazorAttention 更接近“注意力机制行为诊断驱动的分类”。** 二者试图达到相近的目的，但提供分类依据的方式不同。

---

## 七、今天最重要的理解变化

### 原先容易混淆的地方

容易把 Echo Head 和 Induction Head 都理解成“从历史中找到相同 token”，也容易误以为 Head 的得分就是根据当前 Query 临时计算的 token 重要性。

### 现在的理解

1. **Echo** 关注重复 token 本身；**Induction** 更关注重复模式中的后继位置。
2. 评分是根据预先构造的重复序列，在多个 token 位置统计 Attention 行为。
3. 得分按 **Head** 排序，决定哪些 Head 需要完整历史 KV。
4. **免训练是指不更新原模型权重**，并非省掉诊断阶段的前向计算。

## 八、尚未弄懂的问题

1. Echo/Induction 得分在真实代码中是怎样汇总多个重复位置和多次匹配的？
2. 为什么少量 Echo Heads 会对 Induction Heads 的长距离检索表现产生明显影响？
3. 当 Non-Retrieval Head 丢弃大量 KV 后，一个 Compensation Token 为什么还要结合丢弃数量 \(N_d\) 调整注意力权重？
4. 不同上下文长度或不同任务下，固定的 Top 14%/1% 保护比例是否需要重选？

这些是后续研究问题，不代表已经得到实验验证。

## 九、下一步最小行动

下一次重点精读论文第 3.2 节的 **Compensation Token**：先手算被淘汰 \(K,V\) 的平均值，再理解为什么作者在 Attention 近似公式里引入丢弃 token 数量 \(N_d\)，最后核对 Figure 8 的消融实验。

## 十、记录说明

- **今天实际记录的学习：** RazorAttention 的研究动机、Echo/Induction 区别、重复随机序列与 Head 评分流程。
- **教学模拟：** H0～H9 的权重、得分和保护集合由解释需要构造，不是实际模型跑出来的。
- **未记录：** 没有可核实的训练、复现、代码文件、GitHub commit 或实验指标，因此不将其写成个人成果。
