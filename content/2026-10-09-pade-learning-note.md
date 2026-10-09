---
title: "从动态稀疏到预测-执行融合：我对 PADE 三个核心机制的理解"
slug: "pade-predictor-free-sparse-attention-2026-10-09"
kind: "paper"
date: "2026-10-09"
summary: "阅读 PADE 后，我开始理解动态稀疏 Attention 的新瓶颈并不只是计算本身，还包括独立稀疏预测器带来的计算与内存开销。PADE 通过 BUI-GF、BS-OOE 和 ISTA，将安全剪枝、PE 利用率和 I/O 优化结合起来，并通过 Scoreboard、Scheduler 与数据布局落到硬件执行。"
tags: ["边缘计算", "Transformer", "Sparse Attention", "PADE", "BUI-GF", "BS-OOE", "ISTA", "硬件加速"]
featured: false
published: true
example: false
---

# 今天读了什么？

今天阅读了一篇新的论文：

**PADE: A Predictor-Free Sparse Attention Accelerator via Unified Execution and Stage Fusion**

这篇论文继续沿着我前面学习的“稀疏 Attention 如何真正落到硬件上”这条路线展开。

我这次重点关注的是一个新的问题：

> **现有动态稀疏 Attention 虽然可以通过预测重要 Q-K 对来减少正式计算，但这个预测过程本身也会带来额外的计算和内存开销。**

传统的 Dense Attention 不需要额外判断稀疏关系，但所有 Q-K 对都需要完整计算，因此计算和内存访问成本会随序列长度快速增长。

Dynamic Sparsity 可以根据输入动态判断哪些 Q-K 对值得保留，但通常需要额外的 sparsity predictor。

于是问题就变成：

```text
为了减少后面的计算
↓
先增加一个预测阶段
↓
预测阶段自己也需要计算和访存
↓
当执行端越来越低比特、越来越轻量时
↓
预测器本身反而可能变成新的瓶颈
```

PADE 的核心想法就是：

> **不再把“预测”和“正式执行”完全分开，而是在同一个 bit-serial 计算过程中边算、边判断、边剪枝。**

# PADE 要解决的三个核心挑战

我目前把 PADE 的方法理解成三个对应关系：

```text
挑战 1：bit-wise 判断可能误剪
→ BUI-GF

挑战 2：PE 负载不均衡、等待内存
→ BS-OOE

挑战 3：稀疏判断与 tiling 冲突
→ ISTA
```

这三个问题分别对应：

```text
剪得准不准
硬件忙不忙
数据搬得多不多
```

---

# 挑战一：只看高位时，可能误剪重要 token

PADE 使用 bit-serial 的方式逐步计算 Q-K 点积。

这意味着系统可能先只看高位，而不是一开始就把完整 8 bit 都算完。

这样做的好处是：

```text
如果很早就能确定某个 token 不重要
↓
后面的低位就不需要继续计算
```

但问题也很明显：

> **只看高位得到的 partial score，不一定等于最终真实 score。**

## 一个具体例子

假设：

\[
Q=[2,1]
\]

\[
K=[3,2]
\]

那么完整点积是：

\[
Q\cdot K
=
2	imes3+1	imes2
=
8
\]

如果把 K 按二进制位贡献拆开：

\[
3=2+1
\]

\[
2=2+0
\]

因此：

\[
K=[3,2]
=
[2,2]+[1,0]
\]

如果当前只计算到较高位部分：

\[
Q\cdot[2,2]
=
2	imes2+1	imes2
=
6
\]

所以此时：

```text
partial score = 6
```

但真实最终结果其实是：

```text
final score = 8
```

如果当前剪枝阈值为：

```text
Threshold = 7
```

直接根据 partial score 判断：

```text
6 < 7
→ 剪掉
```

就会产生误剪。

因为最终结果：

```text
8 > 7
```

这个 token 实际上应该被保留。

# BUI-GF：用不确定区间做保守剪枝

为了解决这个问题，PADE 不会只看当前 partial score 就立即做决定。

它还会考虑：

> **当前还没有计算的低位，最多可能让最终结果发生多大变化。**

所以可以把当前真实 score 看成一个区间。

例如：

```text
当前 partial score = 6

考虑剩余低位以后：
最终可能范围 = [6, 8]
```

而阈值是：

```text
Threshold = 7
```

因为：

```text
6 < 7 < 8
```

说明这个 token 仍然存在超过阈值的可能。

因此：

```text
不能剪
→ 继续计算下一 bit plane
```

PADE 的关键判断可以概括为：

> **只有当“最大可能值”仍然低于阈值时，才进行剪枝。**

例如：

```text
当前 partial score = 2
最终可能范围 = [2,4]
Threshold = 7
```

由于：

```text
4 < 7
```

即使剩余 bit 全部朝最有利的方向变化，也不可能超过阈值。

这时才可以安全地：

```text
直接剪掉
```

所以我目前对 BUI-GF 的理解是：

> **它并不是让 partial score 更准确，而是给 partial score 加上一个“安全边界”，避免因为只看高位而过早误删重要 token。**

---

# 挑战二：bit-level 计算会导致 PE 负载不均

不同 Key 的 bit pattern 不一样。

某些 bit plane 中：

```text
1 很多
```

就意味着需要处理的计算更多。

另一些 bit plane：

```text
1 很少
```

就会更快结束。

于是不同 PE 之间就可能出现：

```text
PE0：还在算
PE1：已经结束
PE2：还在算
PE3：已经空闲
```

从而产生负载不均衡。

另外，bit-serial 计算还需要逐步从 DRAM 请求后续 bit plane。

于是可能出现：

```text
算一下
↓
等内存
↓
空闲
↓
再算
```

这会降低 PE 利用率。

# BS-OOE：减少工作量不均和等待时间

PADE 使用 BS-OOE 解决这个问题。

## BS：Bidirectional Sparsity

我目前的理解是：

如果一个 bit plane 中：

```text
1 很多
```

那么换个角度看：

```text
0 就很少
```

因此可以通过等价的计算变换，选择更少的一侧参与计算。

目的不是改变数学结果，而是：

> **让实际需要处理的 bit 数尽量少。**

这样不同 PE 的工作量可以更加均衡。

## OOE：Out-of-Order Execution

如果某个 PE 正在等当前 Key 的下一个 bit plane：

传统方式：

```text
算 K0
↓
等待 K0 下一 bit
↓
PE 空闲
↓
继续 K0
```

PADE 则允许：

```text
算 K0
↓
K0 等内存
↓
先算 K1 / K2 / K3
↓
K0 数据到达
↓
再回来继续 K0
```

所以 OOE 的本质是：

> **不要让计算单元因为等数据而空闲。**

这让我意识到：

> **硬件加速不仅取决于“少算多少”，也取决于“计算单元有没有一直处于工作状态”。**

---

# 挑战三：稀疏判断与 tiling 会发生冲突

为了减少 DRAM 和 SRAM 之间的数据搬运，矩阵计算通常会进行 tiling。

也就是把一个大矩阵拆成多个小块：

```text
大矩阵
↓
Tile 1
Tile 2
Tile 3
...
```

每次只处理一个 tile。

这样可以减少片外内存访问。

但稀疏判断通常需要参考一整行 Attention score 的相对分布，例如：

```text
这一行谁最大？
阈值应该是多少？
当前 token 相对其他 token 是否足够重要？
```

问题就出现了：

> **如果当前只能看到一个 tile，又怎么知道整行的最大值和完整分布？**

所以：

```text
row-wise sparsity decision
```

和：

```text
tile-wise execution
```

之间存在冲突。

# ISTA：在 tile 内继续做安全稀疏判断

PADE 的 ISTA 尝试把稀疏判断安全地下沉到 tile 内。

我目前的理解是：

如果某个 token 在当前局部 tile 中已经非常弱，那么随着后续元素加入，softmax 分母只会继续增大，它的相对权重不会突然变得更强。

因此可以在 tile 内进行安全判断：

```text
当前 tile 内已经很弱
↓
整体加入更多 token 后只会更弱
↓
可以提前剪枝
```

这样：

```text
稀疏
+
tiling
```

就可以兼容起来。

于是系统只需要继续保留重要 token，并进一步减少不必要的 V 读取和数据搬运。

---

# Scoreboard、Scheduler 和 Data Layout 是怎么把算法落到硬件上的？

今天我还开始理解一个很重要的问题：

> **提出一个算法思想并不等于它能直接在芯片上高效运行。**

PADE 还需要很多硬件机制把这些思想真正实现。

## Scoreboard：记录中间结果

bit-serial 不是一次把一个 Key 全部算完。

例如：

```text
K0 算到第 2 个 bit plane
→ 得到 partial score
↓
暂时去算 K1 / K2
↓
之后再回来继续 K0
```

这时就需要一个地方记住：

```text
K0 当前 partial score 是多少
K0 已经算到哪个 bit plane
```

所以我把 Scoreboard 理解成：

> **中间结果记忆本。**

它让“暂停当前 Key → 先算其他 Key → 再回来继续”成为可能。

## Scheduler：决定谁先算、谁后算

当不同 Key 处在不同状态：

```text
K0：等待数据
K1：可以继续
K2：已经剪掉
K3：下一 bit 已到达
```

系统需要决定：

> 下一步到底让 PE 算谁？

Scheduler 就负责这个问题。

所以可以记成：

> **Scheduler = 决定执行顺序。**

它的目标是尽量减少：

```text
PE 空闲
DRAM 等待
数据重复读取
```

## Data Layout：数据应该怎么摆

PADE 经常需要：

```text
所有 Key 的 MSB
```

而不是一次读取某个 Key 的完整 8 bit。

所以 K 的存储方式应该尽量适合：

```text
MSB 区：
K0.b7 K1.b7 K2.b7 K3.b7 ...

下一位：
K0.b6 K1.b6 K2.b6 K3.b6 ...

...
```

这样 bit-serial 处理时就可以更连续地读取同一个 bit plane。

因此我目前把这三个硬件机制总结成：

```text
Scoreboard
→ 记住算到哪里

Scheduler
→ 决定下一步算谁

Data Layout
→ 决定数据怎么摆，才能取数据更高效
```

---

# 今天最重要的理解

## 我原本以为

我之前更容易把 Sparse Attention 理解成：

```text
先找到不重要的 token
↓
把它们剪掉
↓
计算量下降
↓
加速
```

也就是说，我更多关注的是：

> **到底剪掉了多少计算。**

## 现在我的理解是

读完 PADE 的这一部分后，我开始意识到：

> **“剪得多”并不等于“硬件一定跑得快”。**

一个真实的稀疏 Attention 加速器至少还要同时回答：

```text
1. 剪得准不准？
2. PE 会不会大量空闲？
3. DRAM 数据是不是仍然搬了很多？
4. 中间结果怎么保存？
5. 下一步计算怎么调度？
6. 数据在内存里应该怎么摆？
```

所以 PADE 真正优化的是一整条链：

```text
安全剪枝
↓
减少计算
↓
平衡 PE 工作量
↓
隐藏内存等待
↓
减少数据搬运
↓
优化真实硬件执行效率
```

## 为什么我的理解发生了变化

之前我主要从算法角度理解稀疏：

```text
稀疏率 ↑
→ FLOPs ↓
```

但 PADE 让我继续看到：

```text
FLOPs ↓
≠
真实 latency 一定 ↓
```

真正的性能还受到：

```text
预测器开销
PE 利用率
DRAM latency
内存访问模式
调度方式
数据布局
```

影响。

这让我开始把 Attention 加速理解成一个：

> **算法 + 微架构 + 内存系统共同决定的系统问题。**

---

# 今天一个最值得记住的关系

我觉得今天最值得记住的是：

```text
BUI-GF
→ 解决“剪得准不准”

BS-OOE
→ 解决“PE 忙不忙”

ISTA
→ 解决“数据搬得多不多”
```

然后：

```text
Scoreboard
Scheduler
Data Layout
```

负责把这些思想真正实现到芯片上。

所以：

> **算法告诉硬件“应该做什么”，微架构告诉芯片“怎么高效地做”。**

---

# 还没有完全弄懂的问题

1. BUI 的上下界具体如何从未计算 bit plane 推导出来？
2. BUI-GF 中的剪枝阈值具体如何更新？
3. Bidirectional Sparsity 的等价变换为什么可以保证结果不变？
4. Scoreboard 在乱序执行时如何避免不同 Key 的 partial score 混淆？
5. Scheduler 如何判断当前应该优先处理哪个 bit plane？
6. ISTA 在跨 tile 更新 softmax 时，如何保证最终结果与完整计算一致？
7. K 的 bit-plane-first data layout 在 DRAM / SRAM 中具体是怎样映射的？

# 下一步最小行动

下一步我准备只重点弄懂一个问题：

> **BUI-GF 中“partial score → uncertainty interval → threshold → pruning decision”这一整条数学判断过程。**

希望能够自己手算一个完整例子：

```text
Q / K
↓
逐 bit 计算 partial score
↓
构造上下界
↓
更新 threshold
↓
判断继续计算还是提前剪枝
```

如果能真正把这条链手算清楚，我对 PADE 的核心算法部分应该就会更加扎实。
