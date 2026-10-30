---
title: "10 月 29 日学习记录：从 KV 槽位复用到自索引检索，再到 DuoAttention 门控训练"
slug: "2026-10-29-thinkv-self-indexing-duoattention-review"
kind: "paper"
date: "2026-10-29"
summary: "回顾 ThinKV 的 Continuous Thinking、Self-Indexing KVCache 的 Sign Code 与 LUT-GEMV，以及 DuoAttention 的隐藏状态蒸馏、L1 正则和 Head 重排。"
tags: ["论文精读", "KV Cache", "ThinKV", "Self-Indexing KVCache", "DuoAttention", "稀疏注意力"]
featured: false
published: true
example: false
---

# 10 月 29 日学习记录：三篇 KV Cache 论文的机制串联

> 这篇记录根据当天原始笔记整理。内容以**论文机制回顾与个人理解**为主；为了讲清公式而加入的数值均标明为教学示例，不是我实际完成的实验。

## 一、今天学习了什么

今天先回顾 **ThinKV** 怎样避免淘汰 KV 后的大规模显存搬运，再整理 **Self-Indexing KVCache** 如何利用压缩索引检索历史 token，最后重点回顾 **DuoAttention** 如何通过隐藏状态蒸馏与 L1 正则自动区分 Retrieval Head 和 Streaming Head。

我逐渐形成了一条主线：**KV Cache 优化不仅是“删掉多少数据”，还需要考虑留下哪些数据、如何访问它们，以及 GPU 上怎样真正高效执行。**

---

## 二、ThinKV：Continuous Thinking 为什么可以减少显存搬运

### 2.1 原先的疑问

假设物理 KV Cache 中有 6 个槽位：

```text
物理槽位：   0    1    2    3    4    5
当前块：    A    B    C    D    E    F
```

TBE 决定淘汰不连续的 B、D、F 后，可能只剩：

```text
物理槽位：   0    1    2    3    4    5
剩余块：    A   空    C   空    E   空
```

一种直观办法是把 A、C、E 搬到连续区域。但这会产生额外的显存读写，并可能与 Attention 争夺 HBM 带宽。

### 2.2 CT 的处理思路

**Continuous Thinking（CT）** 的重点不是频繁搬动剩余 KV，而是通过逻辑索引、块元数据及访问掩码，让 Attention 正确找到仍然有效的 KV；释放的物理槽位可以留给之后的新块使用。

例如：

```text
逻辑有效块顺序： A → C → E
物理地址：        0   2   4
可复用的槽位：    1、3、5
```

注意，**物理位置不连续不等于 Attention 的逻辑位置可以混乱**。系统仍要维护正确的 K/V 配对和逻辑顺序。对于使用 RoPE 的模型，还必须正确处理与位置有关的表示。

**今天的理解变化：** 以前容易以为“淘汰 KV 后必须整理得连续才能使用”，现在理解到，**只要索引与掩码维护正确，就可以避免许多不必要的物理搬运**。

---

## 三、Self-Indexing KVCache：把 Key 的压缩表示用作检索索引

### 3.1 研究问题

长上下文 Decode 中，新 Query 往往需要评估大量历史 Key。作者希望同时减少：

1. KV Cache 本身的存储开销；
2. 找出重要历史 token 时需要读取和计算的 Key 数据量。

它的核心思想是：**Key 的低成本编码不仅用于压缩，还直接充当检索索引。**

### 3.2 Sign-Based VQ：每 4 维得到一个 Code

假设某个 Key 的四维子向量为：

\[
k=[2,-3,0.5,-1]
\]

使用“非负数编码为 1，负数编码为 0”的教学约定：

\[
[+,-,+,-]\longrightarrow1010_2=10
\]

所以这个子向量属于 **Code 10**。4 个符号位共有：

\[
2^4=16
\]

种可能，因此**每个四维分组**最多包含 16 种 Code。若 Key 维度为 128，则有：

\[
G=128/4=32
\]

个分组；每个历史 token 对应 **32 个分组 Code**，而不是整个 128 维 Key 只有一个 Code。

这里的 Code 是 **Key 的符号类别编号**，不是 Tokenizer 产生的词表 ID。

### 3.3 Centroid 来自 Key，而不是 Query

假设两个 Key 子向量具有相同 Code：

\[
k_A=[2,-3,1,-2],\qquad k_B=[4,-1,2,-3]
\]

它们都属于 Code 10。其簇中心为：

\[
c_{10}=\frac{k_A+k_B}{2}=[3,-2,1.5,-2.5]
\]

虽然 A、B **共用一个检索用 Centroid**，但它们并没有共用同一条真实 Key；实际存储的量化 K/V 仍分别对应 A、B。

### 3.4 Entropy-Aware Normalization 的作用

如果某个通道的 Key 取值总是正数，符号编码就容易全部变成 1，区分能力很差。

作者采用按通道均值中心化：

\[
K'_{i,d}=K_{i,d}-\mu_d,\qquad
\mu_d=\frac1L\sum_{i=1}^{L}K_{i,d}
\]

例如 \([8,9,10,11,12]\) 的均值为 10，中心化后是 \([-2,-1,0,1,2]\)。符号分布因此可能更均衡。

**需要澄清：** 均值中心化不是保证每个通道正负各占一半，而是缓解部分通道符号过度偏斜的问题。

### 3.5 LUT-GEMV：为什么“查 Code”能代替大量浮点点积

对当前 Query 的每个四维分组 \(q^{(g)}\)，先计算它与该组所有有效 Centroid 的相关性：

\[
LUT_g[c]=q^{(g)}\cdot c_{g,c}
\]

历史第 \(i\) 个 token 已经记录了每组 Code，因此其近似检索分数可以写成：

\[
\boxed{\widehat s_i=\sum_{g=1}^{G}LUT_g[Code_{i,g}]}
\]

这意味着：**当前 Q 先和少量 Centroid 做浮点点积，随后扫描大量历史 token 时只需读取紧凑 Code、查表、累加。**

#### 一个手算例子（教学数据）

假设只有两组，当前 LUT 是：

| Code | 第一组 LUT | 第二组 LUT |
|---|---:|---:|
| 3 | -2 | 3 |
| 5 | 1 | 5 |
| 10 | 6 | 0 |
| 12 | 4 | 2 |

历史 token A 的 Code 为 \([10,5]\)，B 的 Code 为 \([12,3]\)。

\[
\widehat s_A=LUT_1[10]+LUT_2[5]=6+5=11
\]

\[
\widehat s_B=LUT_1[12]+LUT_2[3]=4+3=7
\]

因此本次近似检索中 A 排在 B 前面。**LUT 数值是教学假设，不是论文实验输出。**

### 3.6 我对 Centroid 更新机制的进一步理解

- **新 Query 到来**：Query 改变，需要根据当前 Query 重新生成 LUT；不代表历史 Code 必须重算。
- **历史 Key 不变**：从这些 Key 统计出的 Centroid 可以复用。
- **新 Key 加入**：新 token 需要有自己的编码与量化 KV；如果采用动态更新的 Codebook，那么受影响的簇统计可能需要维护。

重要边界：**不能直接断言论文在每一个 Decode Step 都增量重算所有 Centroid。** 实际动态 Codebook 更新方式应以代码实现为准；若中心化均值同时更新，还可能影响已有 Code。

### 3.7 最终 Attention 仍用选中 token 自己的 K/V

Sign Code 与 Centroid 负责的是**近似检索**，不是完全替代真实的 Key 和 Value。选出 Top-k 后，再读取对应 token 的低比特 K/V，反量化并进行最终 Sparse Attention。

我用一个 Value 的 2-bit 量化例子回顾：

\[
V=[2,4,6,8],\qquad B=2
\]

\[
q_s=\frac{8-2}{2^2-1}=2,\qquad z_p=2
\]

\[
Q(V)=\operatorname{round}\left(\frac{V-z_p}{q_s}\right)=[0,1,2,3]
\]

二进制对应 \([00,01,10,11]\)。反量化公式：

\[
D(V)=q_sQ(V)+z_p
\]

在这组恰好落在量化格点上的教学数据中，可以准确恢复原值；现实量化通常存在误差。Key 的量化还涉及符号和幅值分离，不能简单认为 K/V 的处理完全相同。

---

## 四、DuoAttention：通过学习门控参数识别 Retrieval Head

### 4.1 为什么要分两类 Head？

DuoAttention 观察到，并不是每个 Attention Head 都需要完整历史：

| 类型 | 推理时的历史访问 | KV 保留方式 |
|---|---|---|
| Retrieval Head | Full Attention，能够访问完整历史 | 保存完整历史 KV |
| Streaming Head | 主要使用 Sink + Recent | 只保存开头 Sink 和最近窗口 |

**与 Self-Indexing KVCache 的区别：** DuoAttention 的 Retrieval Head 并不表示“动态挑选 Top-k token”，而是**这个 Head 继续使用完整历史 Attention**。

### 4.2 离线识别：给每个 Head 训练一个门控

训练阶段的混合输出：

\[
\boxed{A_{mixed}^{(\ell,h)}=\alpha_{\ell,h}A_{full}^{(\ell,h)}+(1-\alpha_{\ell,h})A_{stream}^{(\ell,h)}}
\]

其中 \(\alpha\) 越大表示越依赖 Full 路径。原模型的主要参数冻结，只优化门控。

作者使用 Passkey 合成任务，让模型必须从长文本中召回远距离信息，以便识别真正依赖完整历史的 Heads。

### 4.3 蒸馏损失：比较最终隐藏状态

\[
\boxed{\mathcal L_{distill}=\frac1N\sum_{n=1}^N\sum_{t=T-l+1}^T\|H_{full}^{(n,t)}-H_{mixed}^{(n,t)}\|_2^2}
\]

这里 \(H\) 是模型最终隐藏状态，**不等于某个 Head 的 Attention 输出**。使用末尾答案 token 的隐藏状态，是为了让训练重点落在远距离信息召回时的模型行为。

假设只有一个样本、一个目标位置，两个隐藏状态分别为：

\[
H_{full}=[9,12],\qquad H_{mixed}=[9,8]
\]

则：

\[
\mathcal L_{distill}=\|[9,12]-[9,8]\|_2^2=0^2+4^2=16
\]

这是教学例子，不是论文测量结果。

### 4.4 L1 正则：为什么不让全部 \(\alpha=1\)？

如果只有蒸馏目标，模型可能倾向于令所有 Head 使用 Full Attention，虽然误差小，却几乎不省 KV。

因此加入：

\[
\mathcal L_{reg}=\sum_{\ell,h}|\alpha_{\ell,h}|
\]

\[
\boxed{\mathcal L=\mathcal L_{distill}+\lambda\mathcal L_{reg}},\qquad \lambda=0.05
\]

**蒸馏项**负责约束性能退化，**正则项**负责对 Full 的依赖施加成本。

### 4.5 从初始化到更新一次 \(\alpha\)（教学手算）

使用单个门控的简化隐藏状态模型：

\[
H_{full}=[9,12],\quad H_{mixed}=[9,4+8\alpha]
\]

得到：

\[
\mathcal L(\alpha)=64(1-\alpha)^2+0.05|\alpha|
\]

假设初始化 \(\alpha_0=0.5\)，学习率 \(\eta=0.001\)。

**① 蒸馏项：**

\[
\mathcal L_{distill}=64(1-0.5)^2=16
\]

**② 正则项：**

\[
\lambda\mathcal L_{reg}=0.05\times0.5=0.025
\]

**③ 总损失：**

\[
\mathcal L=16+0.025=16.025
\]

**④ 分别求梯度：**

\[
\frac{d\mathcal L_{distill}}{d\alpha}=-128(1-\alpha)=-64
\]

\[
\lambda\frac{d|\alpha|}{d\alpha}=0.05\quad(\alpha>0)
\]

总梯度：

\[
\frac{d\mathcal L}{d\alpha}=-64+0.05=-63.95
\]

**⑤ 更新：**

\[
\alpha_1=\alpha_0-\eta\frac{d\mathcal L}{d\alpha}=0.5-0.001(-63.95)=\boxed{0.56395}
\]

这个 Head 的 \(\alpha\) 增大，说明在该示例下，增大 Full 路径权重带来的蒸馏改善超过了正则惩罚。真实训练是多个门控共同优化，上述过程只说明梯度更新机制。

### 4.6 部署：二值化和 Head Reordering

训练完成后，以指定阈值 \(\tau\) 分类：

\[
Head_{\ell,h}=\begin{cases}
Retrieval,&\alpha_{\ell,h}>\tau\\
Streaming,&\alpha_{\ell,h}\leq\tau
\end{cases}
\]

部署时**不再对同一个 Head 同时执行 Full 和 Streaming 两条路径**，而是只使用选定的那一种。

同时将 Head 从交错排列如 \([R,S,R,S]\) 重排为 \([R,R,S,S]\)，以减少不规则访问。应当同步调整 Q/K/V 对应通道和输出投影 \(W_O\) 的输入通道，保证计算语义不变；对于 GQA，还要维持共享 KV 关系。

---

## 五、三篇论文连起来看

| 论文 | 重要性判断依据 | 主要优化层次 | 系统关键点 |
|---|---|---|---|
| ThinKV | Thought 类型和历史片段价值 | Thought Segment | CT 索引与槽位复用 |
| Self-Indexing KVCache | Q 与 Sign-Code Centroid 的近似相关性 | Token 检索 + 量化 | LUT-GEMV、稀疏计算 |
| DuoAttention | 门控训练后各 Head 对完整历史的依赖 | Head 分类 | 双缓存、Head Reordering |

我最重要的收获是：**不同论文选择不同的“优化粒度”：块、token、Head。找到重要部分只是第一步，如何让缓存布局和 GPU 执行配合，才决定实际收益。**

## 六、还想继续弄清楚的问题

1. Self-Indexing KVCache 的在线 Codebook 维护在实际实现中如何处理新增 Key 与中心化统计？
2. DuoAttention 的 Retrieval Head 分类在不同任务和推理阶段是否需要动态调整？
3. ThinKV CT 对逻辑位置和物理 KV 槽位的映射，在 Triton Kernel 中如何实现？

这些是后续阅读与复核方向，不表示今天已完成对应实验。

## 七、下一步计划

继续阅读 Head 级 KV 压缩的另一种路线 **RazorAttention**，重点观察其 **Echo Head / Induction Head 的免训练识别方法**，并与 DuoAttention 的 \(\alpha\) 优化路线做比较。

## 八、记录边界

- **已完成：** 阅读与回顾上述论文机制，整理公式和理解差异。
- **教学示例：** 本文所有人工设定的 Q、K、Value、LUT 和隐藏状态数字。
- **未记录：** 当天没有提供可核实的代码文件、GitHub 提交、独立复现实验或实验日志，因此不将其写成已完成成果。
