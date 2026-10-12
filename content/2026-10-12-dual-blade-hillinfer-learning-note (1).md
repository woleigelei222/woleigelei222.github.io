---
title: "从 I/O 重叠到近存储重要性评估：DUAL-BLADE 与 HillInfer 学习记录"
slug: "dual-blade-hillinfer-learning-2026-10-12"
kind: "paper"
date: "2026-10-12"
summary: "完成 DUAL-BLADE 的自适应流水线部分后，开始阅读 HillInfer。我逐渐认识到边缘 LLM 的 KV Cache 优化不仅可以让数据搬得更快，还可以把轻量级重要性评估放在数据附近，减少需要搬运的数据。"
tags: ["边缘计算", "LLM", "KV Cache", "DUAL-BLADE", "HillInfer", "SmartSSD", "自适应流水线", "近存储计算"]
featured: false
published: true
example: false
---

# 今天的学习主线

今天先把上一篇 **DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference** 的第三个设计梳理完，再开始学习 **HillInfer: Efficient Long-Context LLM Inference on the Edge with Hierarchical KV Eviction using SmartSSD**。

两篇论文恰好形成了一条连续的思路：当 KV Cache 必须放到 SSD 时，能否把数据传输安排得更高效？如果搬运成本仍然很高，能否让一部分判断直接在 SSD 附近完成，从而少搬一些数据？

这也是我今天最明显的理解变化：**优化数据传输的方式很重要，但减少不必要的传输同样重要。**

# 一、DUAL-BLADE：Overlap-Intra 与 Overlap-Cross

DUAL-BLADE 的第三个设计是 **Adaptive Pipeline Parallelism（自适应流水线并行）**。它不是继续改变 KV Cache 存在哪儿，而是思考 SSD I/O 与 GPU DMA 应该怎样配合，才能减少等待。

## 1. Overlap-Intra：同类 I/O 并行

Overlap-Intra 更接近我一开始理解的“两个线程一起干活”：两个复制线程在同一种硬件路径上并发执行操作，例如同时发起 NVMe 读取。

```text
线程 T1：从 SSD 读取 K
线程 T2：从 SSD 读取 V
             ↓
两个线程可能同时使用 SSD 带宽
```

如果 SSD 还有空闲带宽，同时发起请求有机会提高吞吐量。但如果单条读取流已经接近带宽上限，第二条线程加入后未必会更快，还可能增加资源竞争。

## 2. Overlap-Cross：让不同传输阶段重叠

Overlap-Cross 则是错开线程的启动时间，使不同线程尽可能处于不同硬件执行阶段。

```text
线程 T1：从 SSD 读取 K
         ↓
         把 K 从 Pinned Memory 搬到 GPU

线程 T2：稍晚启动
         ↓
         当 T1 正在进行 GPU DMA 时
         从 SSD 读取 V
```

这样，SSD 读取下一份数据与 GPU DMA 搬运上一份数据便有机会同时进行。我现在理解，“并行”并不一定是让所有线程同时抢同一个设备；**让不同设备分别工作，也是一种有价值的并行。**

## 3. 为什么要自适应选择？

这两种方式各有适用条件，不能简单认定其中一种永远更好。

- SSD 带宽还没有得到充分利用时，Overlap-Intra 可能更合适。
- SSD 已经接近带宽上限时，让另一条线程继续争抢 SSD 可能效果有限，而 Overlap-Cross 更有机会隐藏传输等待。

论文采用的是运行时比较，而不是训练一个模型去预测策略：

```text
Decode 第 1 轮：Warm-up，不用于策略比较
        ↓
Decode 第 2 轮：测试 Overlap-Intra 的吞吐量
        ↓
Decode 第 3 轮：测试 Overlap-Cross 的吞吐量
        ↓
Decode 第 4 轮及之后：采用测得吞吐量更好的方式
```

今天我从这里得到的启发是：**设备看起来很忙，并不代表整个系统的执行路径已经高效。** 当资源被同类请求挤满时，忙碌可能只是争抢严重；合理的错峰和重叠反而能缩短等待。

关于效果，我目前依据论文中的消融实验来理解作者设计的有效性，**没有进行自己的代码实现或性能测量**。

# 二、转向 HillInfer：既然搬数据慢，能否在数据附近计算？

接着我开始阅读 HillInfer。与 DUAL-BLADE 主要优化 KV Cache 的传输路径不同，HillInfer 提供了另一种思路：**把轻量级的 token 重要性评估下沉到 SmartSSD 附近，减少为了“判断重要性”而搬运的大量历史 KV。**

我目前把它的整体执行过程理解为：

```text
GPU 生成当前 Query
        ↓
将 Query 发给 CPU 和 SmartSSD
        ↓
CPU 评估 DRAM 中的历史 Key
SmartSSD 评估存储侧的历史 Key
        ↓
SmartSSD 将重要性分数传给 CPU
        ↓
CPU 汇总分数，选择重要 token
        ↓
只预取选中的历史 K/V 到 GPU
        ↓
GPU 执行真正的 Attention
```

这里有一个我需要特别区分的地方：**CPU 汇总分数并选出重要 token，是本轮 Attention 的筛选决策；未被选中的 KV 并不意味着一定从 SSD 或 DRAM 中永久删除。**

而且，SmartSSD 负责的是轻量级评分，不是代替 GPU 完成整个 Attention。这是我理解这篇论文时非常重要的边界。

# 三、HKM：DRAM 和 SmartSSD 中的 KV 如何管理？

HillInfer 引入了 **HKM（Hierarchical KV Cache Manager）**。我把其中两个缓存池理解为：

```text
CPU DRAM → Hot Pool（热缓存池）
SmartSSD → Cold Pool（冷缓存池）
```

问题在于，token 的重要性会随着当前 Query 改变。某个旧 token 这一轮不重要，下一轮却可能重新受到关注。如果每次都根据单轮评分把它在 DRAM 和 SSD 之间搬来搬去，就容易出现重复的数据迁移。

## 为什么只看 LRU 可能不够？

LRU 主要看最近一次缓存访问时间，而不是 token 在历史 Decode 中被判断为重要的频率。

例如，一个早期 token 即使很久没有被实际访问，也可能在过去多轮的重要性评估中频繁得到较高排名。只按“最近访问时间”淘汰它，可能忽视这种长期重要性。

因此，我目前理解 HKM 还关注两类信息：

1. **时间局部性**：最近生成的 token 往往会在接下来的若干 Decode 步中继续被关注。
2. **历史命中频率**：有些旧 token 虽然不新，却反复被判断为重要，值得优先留在较快的存储层。

Hot Pool 与 Cold Pool 的分配并非永远不变；随着重要性和访问情况变化，KV 可以在两层间晋升或降级。

这一设计让我开始理解：**缓存淘汰不仅是“谁最旧”，还要考虑“谁可能继续有用”。**

# 四、APP：为什么不能简单把任务五五分？

HillInfer 的 **APP（Adaptive Prefetch-based Pipeline）** 主要想解决异构设备间的执行不均衡。

如果 CPU 的计算能力和 SmartSSD 内 FPGA 的计算能力差别很大，即使把 KV 评分任务按数据量各分一半，也未必能让双方同时结束。

```text
CPU：较早完成
      ↓
等待 SmartSSD
      ↓
GPU 的下一阶段可能继续等待
```

因此，我目前对 APP 的核心理解是：**划分工作量时需要考虑设备的处理能力和数据传输成本，让两条路径的整体耗时尽可能接近，再与 GPU 当前层的计算形成重叠。**

今天主要弄清楚了这一设计目标，具体如何由计算吞吐量、带宽和内存上限推导分配比例，仍需要继续用公式练习。

# 五、CEC：让 SmartSSD 的 FPGA 承担得起重要性评估

最后，我开始理解 **CEC（CSD-based Evaluation Configuration）**。

普通 Attention 的公式是：

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt d}\right)V
\]

但 SmartSSD 的任务并不是输出最终的 Attention 结果，而是根据当前 Query 找出更重要的历史 token。所以对同一个 Query 来说，可以先比较原始 \(Q\cdot K\) 分数，而不必为了排名完成整套 Softmax 计算。

为什么可以这样做？因为指数函数严格递增，Softmax 中所有分数又共用同一个正分母，因而在精确计算下不会改变同一行 score 的大小顺序：

\[
s_i>s_j\ \Longrightarrow\
\frac{e^{s_i}}{\sum_k e^{s_k}}>
\frac{e^{s_j}}{\sum_k e^{s_k}}
\]

所以只需要选出排名靠前的 token 时，可以先用 \(QK^T\) 排序。

**这里必须强调：跳过 Softmax 仅用于重要性评估阶段，并不表示 GPU 最终执行的 Attention 也不需要 Softmax。**

我目前对 CEC 的认识，主要集中在这个“只做必要计算”的思路上。至于 FPGA 上更具体的流式执行、低精度评分如何影响排名，还需要继续研究。

# 六、今天最重要的理解变化

## 我原本以为

之前读到 KV Cache 卸载时，我更多会把性能优化理解成：

```text
减少 SSD / DRAM / GPU 之间的搬运时间
→ 提升推理速度
```

对于并行，我也更容易理解成增加线程、让更多操作同时发生。

## 现在我的理解是

通过 DUAL-BLADE 的第三个设计，我逐渐认识到：**并发的请求如果都争抢同一个已经饱和的资源，未必能让系统更快；把不同硬件的执行阶段重叠起来可能更有效。**

而 HillInfer 又让我进一步认识到：**除了让 KV Cache 搬得更快，还可以只把少量 Query 发到数据附近，在那里完成轻量级重要性评估，从源头减少不必要的 KV 传输。**

## 为什么发生变化

因为这两篇论文分别回答了两个不同层次的问题：

```text
DUAL-BLADE：数据确实需要搬时，怎样安排 I/O 更高效？

HillInfer：有没有一部分计算可以靠近数据完成，
           让某些 KV 根本不必被搬过来？
```

我现在开始把边缘 LLM 推理优化看作 **计算、存储、数据传输和异构设备调度之间的协同问题**，而不是只盯着某一种硬件的利用率。

# 七、还需要继续弄懂的问题

1. APP 如何综合 CPU/SmartSSD 的处理吞吐量、传输带宽和 DRAM 容量，推导最合适的工作量比例？
2. FPGA 只使用原始 Q·K 分数排序时，具体有哪些适用前提？低精度评分会不会改变分数接近的 token 的排名？
3. HKM 在执行 Hot Pool 与 Cold Pool 的双向迁移时，如何权衡近期重要性变化与历史命中频率，以避免频繁来回搬运？
4. SmartSSD 的评分阶段如何实现流式处理？与一次性加载完整 Key 矩阵相比，具体减少了哪些资源开销？

# 八、下一步最小行动

下一次学习准备聚焦一个问题：**APP 为什么要让 CPU 和 SmartSSD 的总处理时间尽量接近，以及它的工作量分配公式是怎样推导出来的。**

希望能通过一组简单数字，把下面这条逻辑手算清楚：

```text
CPU / SmartSSD 各有多少 KV
        ↓
各自的计算时间 + 传输时间
        ↓
比较两条路径的完成时间
        ↓
调整数据分配比例
        ↓
尽量减少等待
```

---

*说明：本文记录的是当天的论文阅读与个人理解整理，不包含自行实现的代码、实验或性能测量。*
