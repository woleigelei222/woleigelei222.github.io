---
title: "从 KV Cache 复用到多层动态存储：我对 MTDS 边缘推理优化的理解"
slug: "mtds-kv-cache-edge-inference-2026-10-10"
kind: "paper"
date: "2026-10-10"
summary: "阅读 MTDS 后，我开始理解边缘端 LLM 推理中的 KV Cache 优化并不只是提高缓存命中率，而是需要同时比较复用与重算成本、协调多 GPU 的 PCIe 带宽，并根据未来复用价值动态管理 DRAM 中的缓存。"
tags: ["边缘计算", "LLM", "KV Cache", "MTDS", "多层存储", "PCIe", "缓存复用", "边缘AI"]
featured: false
published: true
example: false
---

# 今天读了什么？

今天继续阅读一篇聚焦 **KV Cache 管理与边缘端存储优化** 的论文：

**《Multi-tier dynamic storage of KV cache for LLM inference under resource-constrained conditions》**

相比前面阅读的 PADE，这篇论文关注的重点已经从“Attention 怎么算得更快”逐渐转向：

> **已经生成的中间数据应该怎么存、怎么搬、什么时候复用、什么时候直接重新计算。**

这让我开始从“计算优化”进一步理解到“存储与数据传输优化”。

# 先重新理解 Q、K、V 和 KV Cache

今天我首先把 Transformer 里 Query、Key、Value 的关系重新梳理了一遍。

我目前的理解是：

```text
Query
→ 当前 token 想找什么

Key
→ 当前 token 能提供什么匹配特征

Value
→ 如果这个 token 被关注，真正传递什么信息
```

每个 token 都会生成自己的 Q、K、V。

在 Attention 中：

```text
Query
↓
与多个 Key 做匹配
↓
得到 attention score
↓
经过 softmax 得到权重
↓
权重再作用到对应的 Value
↓
得到新的上下文表示
```

真正需要反复使用的是历史 token 的 K 和 V。

如果每生成一个新 token，都重新计算所有历史 token 的 K/V，会产生大量重复计算。

因此系统通常会把历史 K/V 保存下来：

> **这就是 KV Cache。**

# Prefill 和 Decode 对 KV Cache 的使用方式不同

LLM 推理可以粗略分成两个阶段。

## Prefill

Prefill 会处理整段输入 prompt：

```text
用户输入整段 prompt
↓
计算所有输入 token 的 Q/K/V
↓
建立上下文表示
↓
生成并保存 KV Cache
```

所以 Prefill 更偏向：

> **一次性处理整段输入，并建立初始 KV Cache。**

## Decode

Decode 阶段每次只生成一个新 token。

新 token 会使用自己的 Query 去和历史 Key 做 Attention，并读取历史 Value：

```text
生成新 token
↓
读取历史 KV Cache
↓
完成 Attention
↓
生成下一个 token
↓
把新的 K/V 继续追加进缓存
```

我现在对两个阶段的理解可以概括成：

```text
Prefill
→ 主要负责“建立 KV Cache”

Decode
→ 主要负责“不断读取并追加 KV Cache”
```

# 一个反直觉的问题：缓存存在，不代表一定值得复用

一开始我会自然认为：

> 如果 CPU DRAM 里已经有对应的 KV Cache，那当然应该直接拿来用，而不是重新计算。

但论文让我意识到：

> **缓存已经存在，不代表把它搬回 GPU 一定比 GPU 重新计算更快。**

原因是 KV Cache 不在 GPU 上时，需要经过 PCIe 从 DRAM 搬回 GPU。

所以真正应该比较的是：

```text
缓存复用成本
vs.
重新计算成本
```

我目前把这个判断写成：

```text
T_saved
= 使用缓存后能够节省的重新计算时间

V_KV
= 需要传输的 KV Cache 大小

BW
= 当前可用 PCIe 带宽

T_load ≈ V_KV / BW
```

如果：

\[
T_{saved} > T_{load}
\]

说明：

```text
省下来的计算时间
>
搬运缓存的时间
```

这时复用缓存才有实际收益。

反过来，如果：

\[
T_{saved} \le T_{load}
\]

即使缓存命中了，也可能还不如直接让 GPU 重新计算。

因此今天一个很重要的理解是：

> **缓存命中率高，并不等于系统一定更快。**

# Selective KV Cache Loader：先收集信息，再做决策，最后执行

为了解决“到底该复用还是该重算”的问题，论文把 Selective KV Cache Loader 分成三个模块。

## 1. Request Metadata Module

这个模块负责收集请求相关信息，例如：

```text
请求总 token 数
可复用 token 数
KV Cache 加载时间
重新计算时间
```

我把它理解成：

> **负责看清楚当前请求是什么情况。**

## 2. Policy Analysis Module

这个模块根据前面收集到的信息判断：

> 到底应该 Full Load、Partial Load，还是 No Load？

三种策略可以概括成：

| 策略 | 适用情况 | GPU 下一步 |
| --- | --- | --- |
| Policy #1 Full Load | 整个请求前缀匹配，而且加载比重算快 | 完整加载缓存 K/V，跳过相应 Prefill |
| Policy #2 Partial Load | 只有部分前缀匹配，而且这部分值得加载 | 加载匹配部分，其余 token 做增量 Prefill |
| Policy #3 No Load | 没有匹配，或者加载还不如重算 | 不加载旧 K/V，执行完整 Prefill |

我把 Policy Analysis Module 理解成：

> **负责决定“该怎么做”。**

## 3. Policy Execution Module

有了决策之后，还需要真正执行。

例如：

```text
Full Load
→ 调用加载引擎
→ 把 KV Cache 搬回 GPU

Partial Load
→ 加载匹配部分
→ 未匹配部分继续计算

No Load
→ 不加载
→ GPU 完整 Prefill
```

所以这个模块负责：

> **把策略真正落到推理流程中。**

整个逻辑就是：

```text
Request Metadata
→ 收集情况

Policy Analysis
→ 做决定

Policy Execution
→ 真正执行
```

# 第二个问题：多个 GPU 同时卸载 KV Cache 会抢 PCIe

真实边缘节点可能有多个 GPU。

每个 GPU 完成 Prefill 以后，都可能产生新的 KV Cache，并希望把它从：

```text
GPU VRAM
↓
PCIe
↓
CPU DRAM
```

保存下来。

问题是：

> **多个 GPU 共用有限的 PCIe 带宽。**

如果大家同时传，就可能发生：

```text
GPU0 ─┐
GPU1 ─┼→ PCIe → DRAM
GPU2 ─┘
```

也就是带宽竞争。

所以论文提出了 KV Cache Offloading Scheduler。

# KV Cache Offloading Scheduler：谁更值得先保存？

我目前把整个流程理解成：

```text
多个 GPU 完成 Prefill
↓
每个 GPU 产生 KV Cache
↓
先不要一起传
↓
请求进入队列
↓
做前缀匹配
↓
估计每个 KV Cache 的未来复用价值
↓
按优先级排序
↓
高价值 KV Cache 先进入 DRAM
↓
低价值缓存延后
↓
减少 PCIe 争用
```

## Trie：负责找“谁和谁有公共前缀”

例如：

```text
请求 A：
请介绍边缘计算

请求 B：
请介绍边缘计算中的任务卸载

请求 C：
请介绍边缘计算中的数据缓存
```

那么：

```text
“请介绍边缘计算”
```

是 B 和 C 的公共前缀。

这意味着 A 对应的 KV Cache：

> **未来更可能被其他请求复用。**

所以它应该拥有更高的保存优先级。

我目前把 Trie 理解成：

> **快速找出请求之间的公共前缀和复用关系。**

## TimSort：负责根据价值排序

Trie 找出关系以后，系统会进一步产生优先级。

例如：

```text
KV_A：未来复用价值高
KV_B：一般
KV_C：低
```

然后 TimSort 按优先级排序：

```text
KV_A
↓
KV_B
↓
KV_C
```

让高价值缓存先获得 PCIe 带宽。

所以：

```text
Trie
→ 找关系

TimSort
→ 排顺序
```

# 为什么“DRAM 已经有完整副本”反而要降低优先级？

这是今天另一个让我觉得很有意思的地方。

假设 GPU 刚刚生成：

```text
KV_A
```

但 DRAM 里面其实已经有：

```text
完全相同的 KV_A
```

那么再执行一次：

```text
GPU
↓
PCIe
↓
DRAM
```

就只是重复传输。

所以：

> **已经完整存在于 DRAM 的缓存，没有必要再次抢占 PCIe 带宽。**

这种情况应该降低优先级，甚至取消本次卸载。

于是 Scheduler 实际在判断两件事：

```text
如果很多未来请求可能复用
→ 提高优先级

如果 DRAM 已经有完整副本
→ 降低优先级
```

# 为什么不能只用普通 LRU？

今天我还产生了一个问题：

> 为什么不直接用 LRU，把最久没访问的 KV Cache 淘汰掉？

后来我理解到：

```text
“很久没有访问”
≠
“未来马上不会用”
```

传统 LRU 主要依赖历史访问情况。

但边缘设备中：

```text
请求量可能不稳定
DRAM 容量有限
未来请求模式可能快速变化
```

因此：

> **单一固定的 LRU 阈值可能不够灵活。**

如果阈值太保守：

```text
缓存留得太多
→ DRAM 压力大
→ 甚至可能溢出
```

如果阈值太激进：

```text
缓存删得太快
→ 后面马上又需要
→ 命中率下降
```

所以论文进一步使用动态、分层的 eviction 思路，而不是只依赖一个固定 LRU 阈值。

# 今天最重要的理解

## 我原本以为

我以前比较容易把 KV Cache 理解成：

```text
有缓存
→ 直接复用
→ 一定比重新计算快
```

也会自然认为：

```text
缓存命中率越高
→ 系统性能越好
```

## 现在我的理解是

现在我开始理解：

> **KV Cache 优化本质上不是“缓存越多越好”，而是要判断每一次缓存复用、传输和保存是否真的值得。**

真正需要同时考虑：

```text
重新计算时间
缓存加载时间
PCIe 带宽
GPU 数量
未来复用概率
DRAM 容量
缓存淘汰策略
```

所以一个好的边缘 KV Cache 系统需要同时回答：

```text
这个缓存要不要读回来？
这个缓存值不值得写进去？
多个 GPU 谁先传？
DRAM 满了以后谁应该被淘汰？
```

## 为什么我的理解发生了变化

之前我主要从“计算”角度看 LLM 推理优化。

例如前面的论文更关注：

```text
少算多少 Attention
怎么剪枝
怎么提高 PE 利用率
怎么减少计算
```

而 MTDS 让我开始看到：

> **数据什么时候移动、移动到哪里、值不值得移动，本身也可能比计算更重要。**

这让我开始把边缘 LLM 推理理解成：

```text
计算
+
存储
+
数据传输
+
调度
```

共同决定最终性能的系统问题。

# 今天最值得记住的一句话

> **缓存命中不等于缓存值得复用；真正应该比较的是“复用所节省的计算时间”和“把缓存搬回来所付出的传输时间”。**

# 我还没有完全弄懂的问题

1. Selective KV Cache Loader 中的加载时间和重算时间，在真实系统中具体如何预测？
2. 当多个请求同时到达时，Scheduler 的 priority 最终具体如何计算？
3. Trie 前缀匹配与 TimSort 排序的具体复杂度在大规模请求下会不会成为新的开销？
4. Adaptive Eviction 中未来请求信息和历史 LRU 信息具体如何融合？
5. DRAM、SSD 与 GPU VRAM 三层之间，KV Cache 在真实系统中的数据格式是否完全一致？
6. 当 PCIe 带宽动态变化时，T_load 的预测误差会不会导致策略选择错误？

# 下一步最小行动

下一步我准备重点理解：

> **Adaptive Eviction：为什么 MTDS 不使用固定 LRU，而是把未来请求匹配信息和历史访问情况结合起来决定谁被淘汰。**

希望把下面这条链真正弄清楚：

```text
DRAM 容量压力
↓
历史使用率
+
未来请求匹配情况
↓
最终 utilization score
↓
分层 eviction threshold
↓
保留 / 淘汰 KV Cache
```

如果把这一部分也理解清楚，我就能把 MTDS 的三个核心问题真正串起来：

```text
读不读？
↓
谁先写？
↓
满了以后删谁？
```
