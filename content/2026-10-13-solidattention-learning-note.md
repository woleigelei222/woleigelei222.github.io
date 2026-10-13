---
title: "从 HillInfer 到 SolidAttention：让稀疏 Attention 真正适配 SSD"
slug: "hillinfer-solidattention-ssd-co-design-2026-10-13"
kind: "paper"
date: "2026-10-13"
summary: "总结 HillInfer 的低精度重要性评估，并开始精读 SolidAttention，理解 SSD 访问与稀疏 Attention 的矛盾、KV 交错布局以及推测式预取。"
tags: ["边缘计算", "LLM", "KV Cache", "HillInfer", "SolidAttention", "SSD", "稀疏Attention", "预取"]
featured: false
published: true
example: false
---

# 从 HillInfer 到 SolidAttention：让稀疏 Attention 真正适配 SSD

## 今天读了什么？

今天先完成了上一篇论文 **HillInfer** 中 CEC 机制的总结，接着开始阅读 **SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs**。

这两篇论文讨论的都是资源受限设备上的 KV Cache，但侧重点不同：HillInfer 让我认识到，可以在数据附近完成轻量级重要性评估，尽量避免把全部历史 KV 搬回主机之后再判断；SolidAttention 则让我开始关注，即使已经决定只读取重要的 KV，**它们在 SSD 上怎样排列、何时读取，也会直接影响推理速度**。

## 一、HillInfer 收尾：低精度评分不代表最终 Attention 使用低精度 KV

HillInfer 的 CEC 设计中，SmartSSD 保存历史 KV Cache，可以在重要性评估阶段使用低精度近似运算，让 FPGA 更快地判断哪些 token 值得读取。CPU 汇总这些重要性分数后，选择需要的 token；最终传给 GPU、参与 Attention 的仍是被选中 token 的原始存储精度 KV，例如 FP16。

```text
GPU 生成当前 Query
         ↓
CPU / SmartSSD 评估历史 token 的重要性
         ↓
CPU 汇总分数，选出需要的 token
         ↓
只读取被选中的原始精度 KV
         ↓
GPU 执行真正的 Attention
```

我以前容易把“低精度评分”和“低精度 Attention”理解为同一件事。现在更清楚了：**前者是在降低筛选成本，后者才涉及实际 Attention 使用的数据精度**。

不过，低精度评分仍可能改变分数非常接近的 token 的排序，因此不能把它理解成数学上完全无误差的选择方法。

对我而言，HillInfer 最重要的启发是：不一定需要把完整 Attention 放进 SSD，**只要将能够减少数据传输的轻量级判断放在数据附近，也可能获得收益**。

## 二、SolidAttention 的研究动机：KV 读得少，为什么不一定读得快？

在长上下文推理中，历史 KV Cache 会不断增长。当 KV Cache 超出可用 DRAM 后，可以考虑将完整 KV 保存在 SSD 中，在 Decode 时按需加载。

但这里出现一个矛盾：

- **稀疏 Attention** 可能只需要历史序列中少量、分散的 token；
- **SSD** 更容易在较大传输单位、较规律的访问方式下发挥吞吐性能。

例如当前 Query 只需要：

```text
t2、t19、t57、t103……
```

虽然读取的 token 少了，但如果为这些 token 分别发起小型随机 I/O，请求调度、队列等待和带宽利用不足等开销仍可能很大。

```text
KV Cache 规模持续增长
          ↓
DRAM 无法完整容纳
          ↓
KV Cache 保存在 SSD
          ↓
稀疏 Attention 只选出少量重要 KV
          ↓
重要 KV 分散，产生许多小型 I/O
          ↓
读取量减少，但实际读取时延未必同比下降
```

这使我意识到：**需要读取多少数据**和**硬件如何读取这些数据**，是两个不同的问题。

## 三、论文提出的三个挑战

SolidAttention 的三个主要挑战是：

1. **准确率损失（Accuracy Loss）**：盲目扩大 Block 中的 token 数，可能让代表向量遗漏重要信息。
2. **预取不确定性（Prefetching Indeterminacy）**：下一层的 Query 尚未生成时，无法准确知道它需要哪些 KV Block。
3. **数据一致性（Data Inconsistency）**：SSD 异步预取与 GPU 计算共用有限缓存区时，可能出现数据被提前覆盖的问题。

今天主要学习了前两个挑战。第三个问题及其对应的 SSD-aware Scheduler，还需要下一次继续精读。

## 四、挑战一：为什么扩大 Block Size 可能损失准确率？

SSD 喜欢较大的 I/O 请求，因此直觉上可以尝试把一个 Attention Block 从 32 个 token 扩大到 64 个、128 个。

但是在块级稀疏 Attention 中，系统通常会用每个 Block 的代表向量先做重要性判断。可以用一个简化的表达式理解：

\[
\operatorname{Score}(B_i)=Q\cdot R_i
\]

其中，\(Q\) 是当前 Query，\(R_i\) 是第 \(i\) 个 Block 的代表向量。

当一个 Block 包含越来越多不同语义的 token，一个代表向量就需要概括更多特征，可能使真正重要的 token 所在 Block 没被选中。在固定上下文读取预算下，这种风险还可能进一步增加。

所以，**不能为了减少 SSD I/O，就无限增大每个注意力 Block 中 token 的数量**。

### KV Consolidator：改变排列方式，而不是扩大选择单位

SolidAttention 提出了 K/V 交错布局（Interleaved KV Layout）。

传统布局中，K 和 V 可能分开放：

```text
K1 K2 K3 ...
V1 V2 V3 ...
```

交错布局则按 token 组织：

```text
K1 V1 K2 V2 K3 V3 ...
```

例如，需要读取相邻两个 token 的 K/V：

- 分离布局可能需要分别读取 `K1 K2` 和 `V1 V2`；
- 交错布局有机会通过一次更大的连续传输获取 `K1 V1 K2 V2`。

这样并没有减少总 KV 字节数，也没有改变每个 Attention Block 包含的 token 数，主要改变的是**存储顺序和 I/O 传输单位**。在适合合并访问的情况下，I/O 请求次数可以降低，但不能由此直接推导出所有情形下延迟都减半。

今天我最想区分的两个概念是：

| 概念 | 主要影响 |
|---|---|
| 注意力选择粒度 | 一块代表多少 token，可能影响选块准确率 |
| I/O 传输粒度 | 一次读取多少数据，影响 SSD 带宽利用率 |

**SolidAttention 试图增大后者，而不是盲目增大前者。**

此外，如果先独立计算 K 和 V，再重新复制排列成交错形式，就可能引入 GPU 内存搬运开销。因此论文还通过预先安排 K/V 投影权重、直接产生交错数据，并在 Attention Kernel 中使用固定 stride 访问 K 与 V，尽量减少额外重排。

这又提醒了我：**一种优化不能只看 SSD 侧节约了多少时间，还要考虑它是否在 GPU 侧增加了额外开销**。

## 五、挑战二：为什么需要推测式预取？

如果按传统顺序执行，GPU 可能要等当前 Query 生成、选完 Block 后，才让 SSD 开始读取。

```text
GPU 生成 Query
      ↓
根据 Query 计算 Block 重要性
      ↓
选择 Block 2、5、8
      ↓
SSD 开始读取
      ↓
GPU 等待数据到达
      ↓
执行 Attention
```

SSD 读取期间，GPU 可能处于等待状态。因此一个自然的想法是：**在真正需要这些 KV 之前，就提前把它们读出来**。

问题在于，动态稀疏 Attention 的选择结果取决于当前 Query。下一层的 Query 又依赖上一层的计算结果，在它尚未生成时，系统没办法提前准确知道下一层究竟会选择哪些 KV Block。

### 利用历史选择结果做推测

作者观察到连续 Decode 轮次的 Block 选择结果通常存在一定相似性，因此采用上一轮的选择记录来预取下一轮可能需要的 KV。

下面只是为了理解原理的示例：

```text
上一轮选择：Block 2、5、8、10
本轮先预取：Block 2、5、8、10
本轮实际需要：Block 2、5、8、12
```

预取正确的 `2、5、8` 可以继续使用；`12` 需要补读；错误预取的 `10` 则需要处理。

这让我理解到：**推测式预取不是准确预测未来，而是尝试把多数 SSD I/O 提前进行，允许少量预测错误后再纠正。**

不过，真实收益取决于历史结果的稳定性、SSD 传输速度和预测错误后的补读成本，不能简单假设每轮都会预测正确。

## 六、今天最重要的理解变化

### 我原本以为

在学习稀疏 Attention 时，我比较关注“减少了多少 token 的计算或读取”。我原本认为，只要要读取的 KV 变少，SSD 访问就会明显变快。如果 SSD 更擅长读取大块数据，增加 Block Size 似乎就足够了。

### 现在我的理解

我开始把边缘 LLM 中的 KV 访问问题分成三个层面：

1. **读哪些数据？** 由重要性选择决定，关系到模型准确率。
2. **怎么读取？** 由 KV 排列方式与 I/O 粒度决定，关系到 SSD 的实际效率。
3. **何时读取？** 由预取与执行调度决定，关系到 GPU 是否等待。

“读得少”“读得整齐”和“读得早”是不同的优化目标，需要相互配合。

### 为什么我的理解发生变化

HillInfer 让我认识到，可以将轻量级的 token 重要性评估放到 SmartSSD 附近，减少不必要的搬运。SolidAttention 则进一步让我看到：**即使已经选出重要 KV，如果它们的布局不适合 SSD、加载时机又太晚，GPU 仍可能被 I/O 阻塞**。

我的关注点正在从算法本身的计算量，逐渐扩展到**算法与存储系统的协同设计**。

## 七、尚未完全弄懂的问题

1. SolidAttention 的 Block 代表向量具体如何构造？Block Size 对选择召回率的影响如何衡量？
2. 合并 K/V 投影权重后，交错布局和固定 stride 如何在实际 GPU Kernel 中实现？
3. 预取预测错误时，补读和覆盖操作会带来多少额外开销？
4. SSD 异步加载下一层 KV 时，怎样避免覆盖 GPU 尚未使用的数据？
5. SSD-aware Scheduler 如何使用 DAG、关键路径和同步点复用来安排任务？

这些问题目前还属于后续学习方向，今天没有开展个人代码复现或实验验证。

## 八、下一步最小行动

下次重点阅读 **SolidAttention 的数据一致性问题以及 SSD-aware Scheduler**，搞清楚共享缓冲区、计算与 SSD I/O 并发之间的依赖关系。

待这篇论文读完后，再对 **MTDS、DUAL-BLADE、HillInfer、SolidAttention** 做一次整体总结，梳理它们分别从缓存管理、数据迁移、近存储计算和数据布局等角度解决了什么问题。

## 今日总结

今天最大的收获是：

> **减少计算量、减少数据传输量与降低真实推理时延，并不是完全等价的三件事。**

从 HillInfer 到 SolidAttention，我逐渐从“怎样减少不必要的数据搬运”，进一步转向“怎样让必须搬运的数据以更适合硬件的方式传输”。

这也是我目前对边缘 LLM 推理中算法与存储协同优化的一次更深入理解。

---

*说明：本文记录个人论文学习过程。示例 Block 编号为教学示意，不代表论文实验数据；未记录个人实验或复现结果，因此不将作者的实验结论写成个人成果。*
