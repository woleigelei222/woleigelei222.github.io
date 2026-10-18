---
title: "从数据搬运到近存储筛选：HillInfer 学习记录"
slug: "hillinfer-smartssd-learning-2026-10-17"
kind: "paper"
date: "2026-10-17"
summary: "复盘 HillInfer 的分层 KV 管理、CPU 与 SmartSSD 的负载平衡，以及基于 QK 点积的轻量重要性评估。"
tags: ["边缘计算", "HillInfer", "SmartSSD", "KV Cache", "FPGA", "近存储计算", "学习日志"]
featured: false
published: true
example: false
---

# 从数据搬运到近存储筛选：HillInfer 学习记录

## 今天学了什么？

今天继续总结之前阅读的 **HillInfer**，并准备在完成这轮论文回顾后进入新的阅读主题。

在前一篇 DUAL-BLADE 中，我关注的是**如何让 SSD 中的 KV 更快到达 GPU**。HillInfer 则提出了一个很有启发性的问题：**如果为了判断哪些历史 KV 不需要搬运，却必须先把大量历史 Key 取出来评分，那么数据筛选本身也可能十分昂贵。**

因此，HillInfer 把轻量级 token 重要性评估放到 CPU 与带 FPGA 的 SmartSSD 附近，尽量先完成筛选，再将被选中的原始精度 KV 交给 GPU 执行 Attention。

## 一、为什么使用 SmartSSD？

一般 SSD 主要负责存储和传输，而 SmartSSD 集成 FPGA，可以在存储附近执行部分计算。HillInfer 没有要求 FPGA 承担完整 Attention，而是让它计算历史 Key 与当前 Query 的相关性分数。

完整数据流可以概括为：

```text
Prefill 在 GPU 上生成 KV Cache
           ↓
历史 KV 分层保存到 CPU DRAM / SmartSSD
           ↓
Decode 时 GPU 生成 Query
           ↓
CPU 与 SmartSSD 对各自的历史 Key 并行评分
           ↓
SmartSSD 将分数和 token 位置传回 CPU
           ↓
CPU 汇总并选择重要 token
           ↓
仅将所选原始精度 KV 送到 GPU
           ↓
GPU 执行真正的 Attention
```

这里最需要区分的是：**本轮没有被选中的 token 不一定被永久删除；它仍可能存在于较低层缓存中。筛选是否参与当前 Attention，与缓存的降级/晋升，是两个不同动作。**

## 二、HKM：分层重要性评估与双向缓存池

### 2.1 HIE：评分块流式传输

HKM（Hierarchical KV Cache Manager）会协调 CPU 与 SmartSSD 对本地 KV 进行重要性计算。

原始记录中提到一个直观场景：如果 SmartSSD 需要处理 700 个历史 token，可以将它们分成多个评分块：

```text
FPGA：评分块 1 → 发送分数块 1 → 评分块 2 → 发送分数块 2 → ...
CPU ：               汇总块 1 → 汇总块 2 → ...
```

这使 FPGA 评分、评分结果传输与 CPU 汇总之间可以形成流水线。它让我联想到 DUAL-BLADE 的 Overlap-Cross：两者都利用**不同阶段之间的重叠**隐藏等待，但前者偏向评分处理，后者偏向 SSD I/O 与 DMA。

### 2.2 BKP：Hot Pool 与 Cold Pool

HillInfer 将历史 KV 分成：

- **Hot Pool（CPU DRAM）**：倾向于保存近期 token 及历史上经常被选中的重要 token。
- **Cold Pool（SmartSSD）**：保存更多较冷的历史 KV，并可以在存储附近继续评分。

这两个缓存池不是永久固定的。某个 Cold Pool 中的 token 再次变得重要时，可以晋升到 Hot Pool；与此同时需要从 Hot Pool 中选择相对低价值的数据降级，以满足 DRAM 的容量约束。

因此，“Hot/Cold”不仅是存储位置，也包含基于访问规律的动态管理策略。论文考虑了**近期局部性与历史高命中**两类信号，而不只是最近一次访问的时间。

## 三、APP：为什么不能让 CPU 和 SmartSSD 平均分担 KV？

尽管 CPU 和 SmartSSD 可以同时开始工作，两者的处理吞吐、传输带宽和设备负载并不相同。

如果各分配 50% 的 KV，可能 CPU 早早完成，而 SmartSSD 仍在评分或传输，从而拖慢整体时间。

论文从评分时间与选中 KV 的传输时间出发，建立估计：

\[
T_{\mathrm{CPU}}=\frac{M_c}{f_c}+\frac{\alpha M_c}{B_c}
\]

\[
T_{\mathrm{SSD}}=\frac{M_s}{f_s}+\frac{\alpha M_s}{B_s}
\]

其中 \(M_c,M_s\) 分别是分配给两侧的数据量；\(f_c,f_s\) 是有效评分吞吐；\(B_c,B_s\) 是传输带宽；\(\alpha\) 表示需要保留的 KV 比例。

核心思想是让两侧的估计完成时间尽量接近，同时必须满足 DRAM 容量约束。

**我的理解是：并行不只是“同时开始”，还应当让各方尽量在相近时刻完成，否则总体时间仍由慢的一方决定。**

## 四、CEC：为什么评分时不必完整计算 Softmax？

HillInfer 将存储侧任务控制在 FPGA 能够承担的范围内：它只需要为历史 token 排名，并不在 SmartSSD 中执行最终 Attention。

对于同一行分数，只要 \(s_i>s_j\)，就有：

\[
\frac{e^{s_i}}{\sum_k e^{s_k}}>
\frac{e^{s_j}}{\sum_k e^{s_k}}
\]

因此完整 Softmax **不会改变同一行精确分数的排序**。如果只需要选择得分最高的 token，可以用 \(QK^\top\) 点积排序，跳过评分阶段的指数与归一化操作。

这里纠正我原先一个容易引起误解的说法：**QK 点积不是 Softmax 的“分子”**。准确地说，Softmax 分子是 \(e^{s_i}\)，而 QK 经缩放后产生的是输入分数 \(s_i\)。排序不变是因为指数函数单调、同一行分母相同。

为了减少 FPGA 的算力成本，评分阶段可以采用低精度近似；最终送往 GPU 的仍是被选中的原始精度 KV。但低精度评分可能改变接近分数的排序，**不能据此宣称筛选完全无损**。

## 五、今天最重要的理解变化

**我之前主要关注：** 如何让 SSD 到 GPU 的数据传输路径更短、更快。

**现在进一步理解：** 还可以改变计算发生的位置。对于较轻量的筛选任务，先在数据附近计算、减少不必要的原始 KV 搬运，可能比一味追求更高的 SSD 吞吐更有效。

**为什么产生这种变化：** 对比 DUAL-BLADE 与 HillInfer 后，我发现二者分别对应“优化搬运”和“减少需要搬运的数据”，都是系统级思路，却优化不同的环节。

## 六、目前的疑问与下一步

1. 低精度 QK 评分在什么情况下容易导致重要 token 排名变化？
2. Hot Pool 与 Cold Pool 双向迁移的成本何时会抵消缓存收益？
3. CPU 和 SmartSSD 的工作量比例如何随真实负载变化进行动态修正？
4. 近存储评分能否与其他数据布局或 I/O 重叠技术结合？需要先验证访问粒度是否兼容。

今天的记录以论文理解为主，**未记录个人复现代码、性能实验或硬件测试结果**。下一次计划开始阅读新论文，继续沿 KV Cache 的重要性选择方向深入。

> **今日结论：** 不要只问“数据怎样搬得更快”，还应该问“哪些数据根本不必搬，以及计算是不是可以靠近数据发生”。
