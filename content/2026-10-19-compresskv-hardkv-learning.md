---
title: "从逐层误差预算到动态 KV 管理：CompressKV 与 HARD-KV 学习记录"
slug: "2026-10-19-compresskv-hardkv-learning"
kind: "paper"
date: "2026-10-19"
summary: "继续理解 CompressKV 的误差感知逐层 KV Cache 预算分配，并开始阅读 HARD-KV，思考 Head 自适应压缩与 GPU 实际执行效率之间的矛盾。"
tags: ["边缘计算", "LLM 推理", "KV Cache", "CompressKV", "HARD-KV", "Attention", "学习笔记"]
featured: false
published: true
example: false
---

# 从逐层误差预算到动态 KV 管理：CompressKV 与 HARD-KV 学习记录

> **记录日期：2026 年 10 月 19 日**  
> **阅读主题：** CompressKV 的误差感知逐层预算分配；HARD-KV 的研究动机与 Cascade Cache 中的 Dense Cache。  
> **记录范围：** 本文整理当天实际记下的学习内容，不代表已经完成 HARD-KV 全文精读或论文实验复现。

## 一、今天的学习主线

今天先接着理解了 CompressKV 的误差感知逐层 KV Cache 预算分配，随后开始阅读一篇新论文 HARD-KV。两篇文章把我的关注点从“**如何选出重要的历史 token**”进一步带到了“**如何决定不同位置的缓存预算，以及动态压缩能否真的在 GPU 上高效执行**”。

我记录下来的两个问题是：

1. Transformer 的不同层对 KV Cache 压缩的敏感程度可能不同，为什么要强制每层保留一样多的 token？
2. 即使根据注意力分布动态选择出重要 KV，如果不同 Head 需要的缓存长度不断变化，GPU 推理系统是否就能自动获得性能收益？

## 二、CompressKV：误差感知的逐层 KV Cache 预算分配

### 2.1 为什么不能让每层使用相同的缓存预算？

假设一个模型包含 32 层 Transformer。若每层都被分配相同数量的 KV token，管理方式比较简单，但不同层的信息处理功能与压缩敏感性可能并不相同。

对某些层而言，删去部分历史 KV 后，Attention 输出的变化可能很小；对另外一些层而言，相同程度的压缩却可能带来明显变化。因此，**平均分配预算不一定能让有限的 KV 空间发挥最大价值**。

CompressKV 的思路是：先观察某层使用完整 KV 与压缩 KV 时的 Attention 输出差异，再用这个差异估计该层的压缩敏感程度。变化更明显的层，倾向于获得更多缓存预算。

### 2.2 比较完整与压缩后的 Attention 输出

在第 $l$ 层、第 $t$ 个 Decode 步骤，完整 KV 对应的输出为：

$$
O_{\mathrm{full},t}^{(l)}
= W_O^{(l)}\,\operatorname{Attn}\!\left(
Q_t^{(l)},K_{\mathrm{full}}^{(l)},V_{\mathrm{full}}^{(l)}
\right).
$$

采用压缩 KV 后，对应输出为：

$$
O_{\mathrm{comp},t}^{(l)}
= W_O^{(l)}\,\operatorname{Attn}\!\left(
Q_t^{(l)},K_{\mathrm{comp}}^{(l)},V_{\mathrm{comp}}^{(l)}
\right).
$$

两者对照的关键是：在相应校准条件下，比较使用**完整 KV**与使用**筛选后的 KV**所产生的 Attention 输出。式中的 $W_O^{(l)}$ 是输出投影矩阵，因此比较对象不只是某个单独的注意力权重。

可以先写出差值：

$$
\Delta O_t^{(l)} = O_{\mathrm{comp},t}^{(l)} - O_{\mathrm{full},t}^{(l)}.
$$

如果差异较小，说明在这些校准样本与压缩设置下，该层的输出相对稳定；如果差异较大，则提示这层可能需要更多 KV 预算。**这是一项敏感性估计，不等于对任意输入的最终回答质量提供严格保证。**

### 2.3 为什么使用 Frobenius 范数？

Attention 输出是向量或张量，不适合只比较其中一个坐标。因此，论文用 Frobenius 范数衡量整体差异：

$$
\|A\|_F=\sqrt{\sum_{i,j}A_{ij}^{2}}.
$$

对向量而言，它就退化为各元素平方和再开平方。

不过，只比较绝对误差仍可能不公平。比如 A 层本来输出幅度很大，绝对误差为 2；B 层本来输出幅度较小，绝对误差只有 1。**绝对差值更大的 A 层，不一定有更严重的相对扰动。**

因此，论文采用相对误差并在多个 Decode 步骤上累计：

$$
e^{(l)}
=\sum_{t=1}^{T}
\frac{\left\|O_{\mathrm{comp},t}^{(l)}-O_{\mathrm{full},t}^{(l)}\right\|_F}
{\left\|O_{\mathrm{full},t}^{(l)}\right\|_F+\epsilon}.
$$

其中：

- $l$ 表示 Transformer 层；$t$ 表示统计的 Decode 步骤；
- 分子是压缩前后的输出差异；
- 分母是完整输出的尺度；
- $\epsilon$ 是用于避免分母为零的小常数。

接着可以在层间归一化误差权重：

$$
\widetilde e^{(l)} = \frac{e^{(l)}}{\sum_k e^{(k)}}.
$$

这种归一化把不同层的误差放到共同尺度上，供后续缓存预算分配参考。**它并不意味着只要按误差比例机械划分即可；实际方案还要考虑总预算及每层上下限。**

### 2.4 一个用于帮助理解的数值例子

以下是**教学假设**，并非当天运行的实验：

| 层 | 完整输出范数 | 压缩前后差值范数 | 相对误差 |
|---|---:|---:|---:|
| Layer A | 100 | 2 | 约 0.02 |
| Layer B | 5 | 1 | 约 0.20 |

虽然 A 层的绝对误差更大，但 B 层的相对误差更大。它提示我：**需要关注压缩扰动相对于原始输出的影响，不能只看一个绝对差值。**

### 2.5 这一机制与 SRH 的分工

我还梳理了 CompressKV 的两个不同决策层次：

| 机制 | 解决的问题 | 决策结果 |
|---|---|---|
| SRH 引导的 token 选择 | 这一层保留**哪些**历史 token？ | token 索引集合 |
| 误差感知的层间预算分配 | 这一层允许保留**多少**历史 token？ | 每层缓存预算 |

SRH 可以改善重要 token 的排序，但 CompressKV 的预算主要在**Transformer 层级**分配，并没有让每个 Query Head 拥有完全独立的物理 KV 存储决策。尤其在 GQA 中，多个 Query Heads 会共享 KV Heads，因此更细粒度的独立选择还会牵涉共享索引、缓存并集和实际访存开销。

### 2.6 目前看到的局限

我的记录中最关注两点：

- **离线校准与在线负载之间可能存在差异。** 预先计算的层级敏感性，未必能准确代表所有新输入与生成阶段。
- **预算粒度仍有进一步研究空间。** 层级预算有助于管理，但是否可以在不增加过多存储与访问代价的前提下，进一步考虑 KV Group 或 Head 的差异，仍需要实验判断。

这两点是阅读后形成的分析和疑问，并非已经完成验证的研究结论。

## 三、开始阅读 HARD-KV：动态压缩为什么不一定带来高效执行？

今天还开始了下一篇论文：

> *HARD-KV: Head-Adaptive Regularization for Decoding-time KV Compression*（ICML 2026）

这篇论文提出了一个值得继续追问的问题：**算法能够为不同 Head 动态选择 KV，不代表真实 GPU 推理引擎就能高效执行这些不规则的选择结果。**

### 3.1 固定 Top-k 与动态 Top-p

首先复习两种筛选方式：

- **Top-k：** 按评分保留固定数量 $k$ 个 token，预算容易控制，但不同 Head 的重要性分布可能不同。
- **Top-p：** 按注意力概率从高到低累加，达到阈值 $p$ 时停止，因此不同 Head 可能需要不同数量的 token。

Top-p 更适应分数集中或分散的情况，但由此产生的缓存数量会动态变化。如果推理执行系统偏好相对固定、规则的索引与缓存布局，那么算法上的灵活性就可能转化为系统上的管理开销。

### 3.2 Cascade Cache：按生命周期区分缓存状态

HARD-KV 不把所有历史 KV 视作同一种状态，而是采用 Cascade Cache 的三级逻辑管理方式：**Dense、Sparse、Condensed**。

在今天的原始笔记里，我已经着重记录了第一层：

**Tier 1 — Dense Cache（稠密缓存）**

这一层保存近期生成 token 对应的完整 KV，并采用连续追加的方式维护近期窗口。它不在刚生成时就立即对这些 token 做稀疏淘汰，目的是保护局部依赖、维持近期完整注意力，并减少频繁调整新写入缓存的成本。

另外两层的含义，作为下一步精读的导航：

- **Tier 2 — Sparse Cache：** 关注历史 KV 的动态稀疏选择。
- **Tier 3 — Condensed Cache：** 关注更老 KV 的紧凑化、合并或淘汰，以适配实际的缓存约束。

**说明：** 后两层及其具体执行过程并未在当天原始记录中展开；这里只列出后续要学习的框架，暂不将其写成已完成的精读内容。Dense/Sparse/Condensed 是**逻辑缓存状态**，不等同于 GPU/DRAM/SSD 三种物理设备。

## 四、今天最重要的理解变化

根据今天实际写下的内容，我可以把学习思路整理为三个阶段：

**最初关注的判断：** 如果要压缩 KV Cache，关键是找出重要 token，删掉不重要的 token。

**通过 CompressKV 进一步看到：** 仅选对 token 还不够，还要决定**每层的容量分配**是否合理。压缩同样数量的 KV，对不同层造成的影响可能不同。

**开始阅读 HARD-KV 后意识到：** 即使算法上能够动态地选择 KV，后续仍然要面对缓存布局、索引和推理引擎执行效率的问题。**算法节省的存储量与最终的系统加速不是同一个指标。**

这也是今天两篇论文能够衔接起来的原因：前者让我进一步思考“预算怎样分配”，后者开始让我思考“动态选择怎样真正落地”。

## 五、仍待弄清的问题

以下是从当天记录中自然延伸的**待研究问题**，尚未形成实验结论：

1. CompressKV 用离线相对输出误差分配层级预算时，如何衡量其在不同在线任务上的适用性？
2. 如果进一步细化到 KV Group 或 Query Head 的预算，会不会因共享 KV 的并集及索引管理而抵消压缩收益？
3. HARD-KV 如何具体组织 Sparse、Condensed 两层，并在动态 Top-p 选择下维持规则的缓存执行结构？

## 六、下一步学习计划

下一步准备继续阅读 HARD-KV，优先完成 Cascade Cache 的 **Sparse Cache、Condensed Cache** 与 Dense Cache 的联系，再研究后续的 Logits Calibration 和索引规整机制。

目前记录的是论文理解与问题梳理；**尚未记录独立代码实现、性能测试、GitHub Commit 或实验复现数据**。

---

## 参考论文

1. *CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference*，arXiv:2606.24467。
2. *HARD-KV: Head-Adaptive Regularization for Decoding-time KV Compression*，ICML 2026，PMLR 306。
