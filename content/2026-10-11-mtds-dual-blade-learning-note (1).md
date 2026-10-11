---
title: "从缓存淘汰到 NVMe-direct：我对边缘 LLM 数据迁移的进一步理解"
slug: "mtds-to-dual-blade-kv-cache-io-2026-10-11"
kind: "paper"
date: "2026-10-11"
summary: "收尾 MTDS 的滑动窗口利用率和分层淘汰策略，并开始阅读 DUAL-BLADE，理解 Page Cache 抖动、双路径 KV 驻留以及绕过 Page Cache 的 NVMe-direct 数据传输思路。"
tags: ["边缘计算", "LLM", "KV Cache", "MTDS", "DUAL-BLADE", "Page Cache", "NVMe", "存储优化"]
featured: false
published: true
example: false
---

# 今天读了什么？

今天先为之前阅读的 **《Multi-tier dynamic storage of KV cache for LLM inference under resource-constrained conditions》（MTDS）** 做收尾，重点梳理 DRAM 容量阈值、滑动窗口利用率和缓存淘汰的联系。之后开始阅读新论文 **《DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference》**，理解其面向边缘设备内存约束的前两个设计：**双路径 KV 驻留**与 **NVMe-direct 传输路径**。

两篇论文关注的是同一条数据链路的不同层面：

```text
MTDS：是否加载旧 KV？谁先卸载？DRAM 满了淘汰谁？
                 ↓
DUAL-BLADE：确定要访问 SSD 后，如何减少缓存抖动和传输开销？
```

## 一、MTDS 收尾：按 DRAM 压力分层淘汰

我把论文中的存储容量控制整理成四种情况。这里：

- `V` 表示当前 DRAM 中用于 KV Cache 的总占用量；
- `V0`、`V1`、`V2` 是容量阈值，用来区分存储压力；
- `U1`、`U2` 是利用率阈值，用来选择低价值缓存，其中 `U2 > U1`。

| 当前 DRAM 占用量 | 应对策略 | 我目前的理解 |
| --- | --- | --- |
| `V < V0` | 不淘汰 | 空间尚充足 |
| `V0 ≤ V ≤ V1` | 淘汰利用率低于 `U1` 的缓存 | 开始轻度淘汰 |
| `V1 < V ≤ V2` | 淘汰利用率低于 `U2` 的缓存 | 存储压力增大，淘汰更严格 |
| `V > V2` | 暂停新 KV Cache 向 DRAM 卸载 | 避免继续增加内存压力 |

**我现在理解了两类阈值的分工：**`V0/V1/V2` 判断“当前空间有多紧张”，`U1/U2` 判断“在该压力下哪些缓存应该被淘汰”。阈值不是每个缓存自身固定拥有的属性，而是系统管理策略的一部分。

### 滑动窗口利用率：不能只看旧历史

作者将历史使用信息与近期请求的前缀匹配频率结合，用于估计 KV Cache 的后续复用价值。我记录的教学例子是：

- 历史使用分数：`h = 0.10`
- 最近窗口：10 个请求，其中 4 个匹配
- 融合系数：`ξ = 0.25`

因此：

\[
U_{\mathrm{final}}=\xi\frac{m}{N}+(1-\xi)h
=0.25\times\frac{4}{10}+0.75\times0.10=0.175
\]

`0.175` 不是普遍通用的“高价值分数线”；**是否淘汰，还需要与当前 DRAM 压力下的 `U1` 或 `U2` 比较**。这个例子帮助我理解：近期请求的变化可能使原来使用率不高的缓存重新变得值得保留。

### MTDS 三个组件串成一条完整路径

```text
新请求到达
  ↓
Tokenizer 将请求变为 token 序列
  ↓
匹配历史 KV Cache 前缀
  ↓
Selective KV Cache Loader：比较加载和重算成本
  ↓
GPU 执行 Full Load / Partial Load / No Load
  ↓
生成新 KV Cache
  ↓
Offloading Scheduler：判断写回 DRAM 的优先级
  ↓
Adaptive Eviction：DRAM 压力大时决定哪些 KV Cache 转移到 SSD
  ↓
等待后续请求再次复用
```

这让我开始产生一个新的研究视角：**算子内部的访存优化与跨设备 KV Cache 管理是否可以协同考虑？** 这是我目前提出的思考，还不是已经验证的研究结论。

## 二、进入 DUAL-BLADE：为什么 Page Cache 会抖动？

新论文讨论的一个问题是 **Page Cache Thrashing（页缓存抖动）**。

我用一个简化场景理解它：假设 Decode 要反复遍历 **8 GB** 的 KV Cache 工作集，而系统只有 **4 GB** 可供 Page Cache 使用。

```text
第一轮：读取前半部分 KV → 进入 Page Cache
接着读取后半部分 KV → 因空间不足，挤走前半部分
第二轮：又需要前半部分 KV → 再次从 SSD 读取
                       ↓
反复加载与淘汰，同一批数据难以真正被缓存
```

所以问题不只是“Page Cache 比 KV Cache 小”，而是 **Decode 会反复访问一个大于可用缓存容量的工作集**。在这样的访问模式下，缓存可能不断替换即将再次需要的数据，导致缓存命中率下降和 SSD I/O 增多。

### DUAL-BLADE 的第一项设计：双路径 KV 驻留

作者没有把所有 K/V 都交给 Page Cache，也没有完全放弃 Page Cache，而是将数据分到两条访问路径：

| 分组 | 数据访问路径 | 目的 |
| --- | --- | --- |
| Group 1 | Page-cache path | 在可用 DRAM 范围内利用缓存命中优势 |
| Group 2 | NVMe-direct path | 绕过 Page Cache，避免溢出数据反复挤占缓存 |

其中，论文将某层的单个 K 或 V 张量称为 **KPU（KV Placement Unit）**。系统先估算可用 Page Cache 预算 `Bpc`，再用可调预算 `X` 决定多少层的 K/V 进入 Group 1，剩余层进入 Group 2。

我记录的教学例子是：

```text
Transformer：8 层
每层 K+V：1 GB
允许使用 Page Cache 的预算 X：2 GB
                 ↓
前 2 层 K/V → Group 1
后 6 层 K/V → Group 2
```

这个数字例子只用于理解分组，不是我运行的实验，也不代表论文在所有情况下都固定选前两层。分组后还需要执行 **Plan → Bind → Materialize**：先规划归属，再绑定到文件区域或 LBA 区间，最后在首次访问时落实相应存储对象。

## 三、DUAL-BLADE 的第二项设计：缩短 SSD 访问路径

读到这里，我开始意识到：即使已经决定某些 KV Cache 应该走 SSD，**SSD 的实际访问路径也会影响整体推理速度**。

传统 Page Cache 路径中的一种写入过程可以简化为：

```text
GPU
 ↓ DMA
Host Pinned Memory
 ↓ memcpy（额外的主机内存复制）
Page Cache
 ↓ 写回
NVMe SSD
```

DUAL-BLADE 的 NVMe-direct 路径则简化为：

```text
GPU
 ↓ DMA
Host Pinned Memory
 ↓ NVMe I/O
NVMe SSD
```

因此，作者试图消除中间经过 Page Cache 的额外数据复制和常规文件系统处理开销。

这里我特别记住一个容易混淆的地方：**NVMe-direct 不等于 SSD 与 GPU 显存直接通信**。这篇论文仍然使用主机的 Pinned Memory 作为传输缓冲区，只是 SSD 访问绕过了 Page Cache 和部分常规存储软件路径。

### 为什么连续 LBA 值得关注？

在已开始接触的第二项设计中，Tensor 会对应到 SSD 上的逻辑块地址（LBA）。让相关数据落在连续的 LBA 区间，能够使原本逻辑上顺序访问的 KV 数据尽量以顺序 I/O 的形式提交给 NVMe，减少不必要的处理和重排。

我目前只理解了 **“缩短访问路径 + 连续地址组织”** 的作用；更细的 Tensor-ID 映射、MDTS 分块以及 NVMe 指令提交流程，仍需要下次继续学习，不能把它们算作今天已掌握的内容。

## 四、今天最重要的理解变化

### 我原本以为

我之前主要觉得：边缘端 KV Cache 的问题，是 **DRAM 容量不够，就把部分数据放进 SSD**；只要把哪些数据保留、哪些数据淘汰的策略设计好，就完成了主要优化。

### 现在我的理解是

读完 MTDS 的收尾部分并开始接触 DUAL-BLADE 后，我发现还要继续追问：

1. 数据被选中复用时，**加载是否真的比重算快**？
2. 缓存已经要转移到 SSD 时，**Page Cache 是否会因工作集过大而反复抖动**？
3. 即使必须访问 SSD，**能否通过更短的数据路径减少复制与软件开销**？

因此，一个完整的边缘 LLM 推理系统需要同时考虑：

```text
缓存复用价值
   +
DRAM 容量与淘汰策略
   +
Page Cache 工作集行为
   +
GPU / DRAM / SSD 的数据传输成本
```

### 为什么发生变化

**MTDS 让我关注“缓存该如何调度与淘汰”，DUAL-BLADE 则让我开始关注“被调度的数据究竟经过什么路径、以什么代价搬运”。** 这两种视角互补，也让我意识到不能只根据缓存命中率或存储容量推断最终性能。

## 五、今天仍然没有完全弄懂的问题

- Page Cache 抖动发生时，为什么命中率可能突然下降，而不是缓慢下降？
- `Bpc` 与可调参数 `X` 在真实设备上具体如何确定？
- 按层分到 Group 1 / Group 2 后，第一次访问时具体如何完成 Bind 和 Materialize？
- Tensor 到连续 LBA 地址的映射如何实现？
- NVMe-direct 后续还会受到哪些命令大小和 I/O 队列限制？
- 数据传输路径的优化，如何与 MTDS 的复用和淘汰策略形成协同？

这些是后续计划理解的问题，并非已验证的结果。

## 六、下一步最小行动

下一次只聚焦 **DUAL-BLADE 的 NVMe-direct + Sequential LBA**，用一个小型 Tensor 的教学例子弄清：

```text
Tensor ID
  ↓
查找到 SSD 的起始 LBA 和块数
  ↓
按 MDTS 限制拆分 I/O 请求
  ↓
提交 NVMe 读写命令
  ↓
经 Pinned Memory 与 GPU 交换数据
```

等把这条数据流彻底弄懂，再继续看论文的第三项设计 **Adaptive Pipeline Parallelism**。

---

**阅读说明：** 本文是个人学习记录，MTDS 的淘汰决策、DUAL-BLADE 的双路径与 NVMe-direct 思路均为对论文的阶段性理解。文中的 8 GB/4 GB 和 8 层/2 GB 场景均为理解机制的教学例子，未进行本人代码实现或实测。论文实验结果与个人想法未混写。

**论文链接：**

- MTDS：<https://doi.org/10.1007/s40747-025-02200-4>
- DUAL-BLADE：<https://arxiv.org/abs/2604.26557>
