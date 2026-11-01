---
title: "Ada-KV 精读：Attention 淘汰误差上界与跨 Head 自适应预算"
slug: "2026-11-01-adakv-adaptive-budget-safeguard"
kind: "paper"
date: "2026-11-01"
summary: "梳理 Ada-KV 为什么优化 Head-wise KV 预算，推导 L1 淘汰损失上界，手算全局 Top-B 与 safeguard，并复盘 Ada-SnapKV 从观察窗口到缓存压缩的流程。"
tags: ["论文精读", "Ada-KV", "NeurIPS 2025", "KV Cache", "Top-k", "Attention"]
featured: false
published: true
example: false
---

# Ada-KV 精读：为什么 KV Cache 预算不应该在 Heads 之间平均分配？

> **学习时间：2026-11-01。** 阅读论文 *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference*（NeurIPS 2025）。本文依据当天的原始笔记整合推导，示例数值为学习手算，不能作为已完成实验报告。

## 01｜研究动机：相同预算不代表相同收益

传统的 Top-k KV Cache 淘汰方法通常在同一层的不同 Attention Heads 之间均匀分配缓存名额。但每个 Head 的注意力分布并不相同：有的 Head 只集中关注少量历史 Token，有的则把权重分散在更多位置。如果一律给各 Head 相同的预算，前者可能浪费名额，后者却可能丢失大量有用信息。

Ada-KV 的核心不是单纯换一个 Token 重要性评分公式，而是**先给整层设总预算 \(B\)，再让各 Head 的候选权重参与同一次全局 Top-B 竞争**，由入选数量决定各自获得多少历史 KV 名额。

与此前阅读的 RazorAttention 不同，这里的重点是 **Head-wise budget allocation（按 Head 分配缓存预算）**，而不是仅把 Head 二分为完整保留或压缩。

## 02｜为什么 KV 淘汰会改变 Attention 输出？

论文在多头情形下将 Attention 输出写为：

\[
y=\sum_{i=1}^{h}A_iV_iW_i^O.
\]

当部分 KV 被删除后，Softmax 只在剩余位置上重新归一化，因此不仅被删除项不再贡献输出，**其余位置的 Attention 权重也会变化**。

### 手算：一个 Head 淘汰一个 KV

取教学示例：

\[
A=[0.6,0.3,0.1],\quad V=[10,20,30],\quad W^O=1.
\]

完整 Attention 输出：

\[
y=0.6(10)+0.3(20)+0.1(30)=15.
\]

淘汰第三个 KV，保留掩码 \(I=[1,1,0]\)。剩余权重之和为 \(0.9\)，故新权重为

\[
\hat A=\left[\frac{0.6}{0.9},\frac{0.3}{0.9}\right]=\left[\frac23,\frac13\right].
\]

输出与 L1 淘汰损失：

\[
\hat y=\frac23(10)+\frac13(20)=13.3333,
\]

\[
\mathcal L_{eviction}=\|y-\hat y\|_1=|15-13.3333|\approx1.6667.
\]

这个数字是**Attention 输出的实际变化**，不是分类准确率或训练交叉熵。

## 03｜Theorem 3.1：为什么关注“淘汰了多少 Attention 权重”？

定义 \(I_{ij}=1\) 表示保留第 \(i\) 个 Head 的第 \(j\) 个 KV；\(I_{ij}=0\) 表示淘汰。论文构造的 L1 淘汰误差上界可写为：

\[
\mathcal L_{eviction}\le\epsilon
=2hC-2C\sum_{i=1}^{h}\sum_{j=1}^{n}I_{ij}A_{ij},
\]

其中 \(C\) 是由 \(V_iW_i^O\) 行向量范数定义的统一上界常数；每个 Head 的注意力权重之和为 1。

因此 \(\sum_{i,j}A_{ij}=h\)，可整理为

\[
\boxed{\epsilon=2C\sum_{i=1}^{h}\sum_{j=1}^{n}(1-I_{ij})A_{ij}}.
\]

这意味着：**理论上界等于常数 \(2C\) 乘上所有被淘汰位置的原始 Attention 权重之和。**

仍以 \(A=[0.6,0.3,0.1]\) 为例，若取有效界常数 \(C=30\)，淘汰第三个位置：

\[
\epsilon=2\times30\times0.1=6,
\qquad
\mathcal L_{eviction}\approx1.6667\le6.
\]

注意区分：**上界是可优化的理论代理目标，不是每次淘汰后的真实误差。** 上界下降并不必然保证所有样本上的实际误差严格下降。

## 04｜Theorem 3.2：给定每个 Head 的预算，为什么取 Top-k？

因为

\[
\epsilon=2hC-2C\sum_{i,j}I_{ij}A_{ij},
\]

固定总常数后，要让上界尽量小，就要让**保留的原始注意力权重之和尽量大**。

在每个 Head 的预算 \(B_i\) 已经固定的前提下，该 Head 选权重最大的 \(B_i\) 个 KV 就能使上述上界最小。这解释了传统 Top-k Eviction 的理论依据，但问题仍在于：**\(B_i\) 为什么必须相等？**

## 05｜核心手算：全局 Top-B 与均匀预算对比

假设有两个 Heads，各有三个历史 KV：

\[
A_0=[0.90,0.05,0.05],\qquad
A_1=[0.40,0.35,0.25].
\]

整层总预算 \(B=4\)。

### 方案 A：均匀分配 \((2,2)\)

H0 取 \(0.90,0.05\)，H1 取 \(0.40,0.35\)。

\[
M_{uniform}=0.90+0.05+0.40+0.35=1.70.
\]

两个 Head 的 Attention 权重总和为 2，被淘汰的权重质量：

\[
D_{uniform}=2-1.70=0.30.
\]

### 方案 B：全局 Top-4

将六个权重合并并排序：

\[
[0.90,0.40,0.35,0.25,0.05,0.05].
\]

选择前四个：H0 取得 \(0.90\) 一个名额；H1 取得 \(0.40,0.35,0.25\) 三个名额。因此：

\[
B^{adaptive}=(1,3),\qquad M_{adaptive}=1.90,
\]

\[
D_{adaptive}=2-1.90=0.10.
\]

在相同 \(B=4\) 下，保留权重质量从 1.70 增加到 1.90，理论上界从 \(2C(0.30)=0.60C\) 降至 \(2C(0.10)=0.20C\)。

| 分配方式 | H0 预算 | H1 预算 | 保留权重和 | 淘汰权重和 |
|---|---:|---:|---:|---:|
| 均匀分配 | 2 | 2 | 1.70 | 0.30 |
| Ada-KV 全局选择 | 1 | 3 | 1.90 | 0.10 |

这个例子体现出：**注意力高度集中的 Head 不一定需要与分散 Head 相同数量的 KV。**

## 06｜Safeguard：为什么要对理论最优预算再做修正？

全局 Top-B 是依据当前观测得到的 Attention 权重分配预算。然而，压缩后的缓存还要用于后续 Query；若一个 Head 在当前观测中获得接近零的名额，未来它可能缺少需要访问的历史 KV。

论文 Algorithm 2 加入固定平滑系数 \(\alpha\)：

\[
\boxed{B_i^{safe}=\alpha B_i^{adaptive}+(1-\alpha)\frac Bh},\qquad \alpha=0.2.
\]

这表示 **20% 自适应预算 + 80% 均匀预算**；\(\alpha\) 是固定 safeguard 超参数，**不是 DuoAttention 中通过蒸馏训练的 Head 门控**。

### 手算：三个 Heads 的 Safeguard

设总历史预算 \(B=6\)，Head 数 \(h=3\)，均匀预算 \((2,2,2)\)，而全局 Top-B 产生 \((4,2,0)\)。

\[
\begin{aligned}
B_0^{safe}&=0.2(4)+0.8(2)=2.4,\\
B_1^{safe}&=0.2(2)+0.8(2)=2.0,\\
B_2^{safe}&=0.2(0)+0.8(2)=1.6.
\end{aligned}
\]

于是连续预算为 \((2.4,2.0,1.6)\)，总和仍为 6。原本零预算的 H2 得到了非零配额，降低了极端分配的风险。

**工程限制：** 真实 KV 条目数必须为整数，需要兼顾配额和总预算的整数化实现。原始笔记没有记录论文此处唯一指定的取整规则，因此不能把某种教学取整方案当作原文唯一做法。

## 07｜Ada-SnapKV 的实际流程：观察窗口、评分、预算、裁剪

今天的笔记还整理了 Ada-SnapKV/Ada-Pyramid 的落地顺序，可以和论文 Algorithm 2 对应：

1. **在 Prefill 后**得到完整 KV Cache，取最近观察窗口的 Query；论文默认观察窗口大小为 32。
2. 让这些 Query 关注窗口外的历史 Key，计算观测 Attention 分数。
3. 沿历史位置进行局部 max-pooling，随后对观察 Query 的分数取均值，形成每个历史 KV 的重要性估计；主实验池化核大小为 7。
4. 因观察窗口中的 KV 要强制保留，先从整层预算扣除 \(winsize\times h\)（这里采用论文算法中的 Head 记号），余下的才是可分配历史预算。
5. 对所有 Head 的历史评分做 **全局 Top-B**，统计每个 Head 入选个数，得到 \(B_i^{adaptive}\)。
6. 用 \(\alpha=0.2\) 进行 Safeguard 修正，得到稳健分配额度。
7. **在各 Head 内**按修正预算对历史 KV 进行 Top-k 保留；将强制保留的观察窗口 KV 拼回，形成最后的压缩缓存。

简化记忆：

\[
\text{Observation Q}
\rightarrow\text{历史 KV 评分}
\rightarrow\text{全局 Top-B 分名额}
\rightarrow\text{Safeguard}
\rightarrow\text{Head 内 Top-k}
\rightarrow\text{拼回窗口}.
\]

特别要区分：**跨 Head 的全局 Top-B 决定“每个 Head 有几个名额”；Head 内 Top-k 决定“具体保留哪些 Token”。**

## 08｜本次学习的理解变化与待解决问题

学习 Ada-KV 前，我主要从“重要 Token 是否被保留”的角度看 KV 压缩。今天更清楚地认识到：在固定总体容量下，**预算如何跨 Head 分配本身就是一个独立且可以理论化优化的问题**。

理论上，作者构造了 L1 Attention 输出误差上界，把复杂的输出差异问题转化为保留注意力权重的优化；工程上，又利用最近观察窗口提供重要性估计，并用 Safeguard 减轻未来 Query 不确定性带来的风险。

仍有值得进一步检查的细节：一是 Safeguard 连续预算在真实实现中的整数处理；二是 GQA 共享 KV 时评分和预算的合并方式；三是注意力分布变化时，全局 Top-B 的理论收益能否稳定转化为实际任务精度改善。论文还讨论了变长缓存布局与 FlashAttention，但当天记录尚未涉及详细系统复现。

**今日实际完成：** 论文方法梳理、公式理解、上界与预算分配教学手算。**未记录：** 自己运行 CUDA、复现模型指标或提交代码。下一步可对照原文 Algorithm 2 与代码，验证 Safeguard 预算的离散化与 GQA 兼容细节。

*整理说明：本文基于当天原始阅读笔记，公式组织、逻辑衔接和教学手算表达由 AI 辅助整理，所有例子均未冒充实际测量结果。*
