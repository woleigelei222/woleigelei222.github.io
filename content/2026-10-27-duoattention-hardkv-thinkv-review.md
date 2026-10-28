---
title: "从 Head 部署到 Thought 级压缩：DuoAttention、HARD-KV 与 ThinKV 学习回顾"
slug: "2026-10-27-duoattention-hardkv-thinkv-review"
kind: "paper"
date: "2026-10-27"
summary: "梳理 DuoAttention 的门控二值化、Head 重排和消融实验，并回顾 HARD-KV 的三级缓存与 Logits Calibration，以及 ThinKV 的 KDE 思考分类、混合精度和 K-means 渐进淘汰。"
tags: ["论文精读", "LLM 推理", "KV Cache", "DuoAttention", "HARD-KV", "ThinKV", "学习笔记"]
featured: false
published: true
example: false
---

# 从 Head 部署到 Thought 级压缩：DuoAttention、HARD-KV 与 ThinKV 学习回顾

> **学习日期：**2026 年 10 月 27 日  
> **学习方式：**延续 DuoAttention 精读，并回顾之前学习的 HARD-KV 与 ThinKV。  
> **记录说明：**本文整理的是当日实际学习与讨论的内容。公式解释和少量数字示例属于理解性梳理，并非本人完成的代码实验；涉及仍需核实的论文实现细节会单独标注。

## 一、今天的学习主线

今天的学习分为两个部分：

1. **继续阅读 DuoAttention**：理解为什么训练阶段的连续门控 $\alpha$ 必须在部署前二值化，以及为什么需要重排 Head；接着回看消融实验所验证的设计取舍。
2. **回顾前期论文**：从 HARD-KV 的动态 Head-wise KV 管理，回顾到 ThinKV 的 Thought-wise 混合量化与渐进淘汰，重点巩固 Logits Calibration、KDE 和 K-means 的含义。

在这次整理中，我逐渐把注意力从“算法选中了哪些 KV”转移到“选择策略怎样落实为实际的缓存结构和 GPU 执行”。

---

## 二、DuoAttention：训练门控为什么要在部署前二值化？

### 2.1 连续门控 $\alpha$ 是训练工具，不是最终执行策略

DuoAttention 在识别 Retrieval Head 和 Streaming Head 时，采用两条 Attention 路径的输出混合：

$$
A_{\mathrm{mixed}}
=\alpha A_{\mathrm{full}}+(1-\alpha)A_{\mathrm{stream}}.
$$

例如，某个 KV Head 学到 $\alpha=0.7$，训练时便可以把该 Head 的输出表示为：

$$
A_{\mathrm{mixed}}=0.7A_{\mathrm{full}}+0.3A_{\mathrm{stream}}.
$$

这个连续参数有利于通过梯度优化找到哪些 Head 真正依赖完整历史。但是，如果在实际 Decode 时仍同时运行 Full 与 Streaming 两条路径，Full 路径仍需要保存和读取完整历史 KV，难以获得预期的缓存收益。

因此部署前需要根据阈值 $\tau$ 做一次离散选择：

$$
\operatorname{Type}(h)=
\begin{cases}
\mathrm{Retrieval/Full}, & \alpha_h>\tau,\\
\mathrm{Streaming}, & \alpha_h\le\tau.
\end{cases}
$$

**我今天重新理解到：**门控学习回答的是“哪些 Head 值得保留完整历史”，二值化回答的是“正式部署时每个 Head 实际执行哪条路径”。

### 2.2 阈值 $\tau$ 与 Retrieval Head 比例

论文可以根据预设的目标 Retrieval KV Head 比例，对训练后的 $\alpha$ 排序，再选取对应阈值。它不是要求所有模型都使用固定的 $\tau=0.5$。

一个边界问题是：若多个 Head 的门控值相同，仅靠严格的数值阈值未必能精确选出目标数量，具体实现需要处理并列情况。这个问题不改变整体方法，但提醒我：**算法描述中的比例约束，落地时还需要明确排序和边界规则。**

### 2.3 Retrieval Head 比例不等于整个 KV Cache 的占用比例

假设 Retrieval KV Head 占比为 $r$，完整序列长度为 $L$，每个 Streaming Head 保留 $S$ 个 Sink token 与 $W$ 个 Recent token。忽略 Head 之间的维度差异和缓存管理元数据时，有：

$$
\boxed{
\frac{M_{\mathrm{duo}}}{M_{\mathrm{full}}}
\approx r+(1-r)\frac{S+W}{L}
}
$$

其中：

- $r$：采用完整缓存的 KV Head 比例；
- $S$：Sink 保留量；
- $W$：Recent Window 长度；
- $L$：完整上下文长度。

例如设 $r=0.25$、$L=10000$、$S=64$、$W=256$，则：

$$
\frac{M_{\mathrm{duo}}}{M_{\mathrm{full}}}
\approx0.25+0.75\times\frac{320}{10000}=0.274.
$$

即这一**教学配置下**约为原来的 $27.4\%$，而不是恰好 $25\%$。原因是 Streaming Head 仍保留固定窗口；当 $L$ 越来越大时，该比例才趋近 $r$。

### 2.4 为什么需要 Head 重排？

训练得到的 Head 分类可能交错排列，例如：

```text
原始顺序：R, S, R, S, S, R
重排之后：R, R, R, S, S, S
```

若保持交错布局，GPU 可能需要额外的索引收集与重组来分别运行两种 Attention。将同类型 Head 排成连续的通道区间，更容易进行规则切片、独立缓存管理和专用 Kernel 调度。

这里我原来的表述需要修正：**重排的目的不是“连续切片可能引入更多操作”，恰恰相反，是为了减少不规则 Gather/Scatter 的潜在成本。**

另一个容易忽略的问题是计算等价性。如果只交换 Head 输出而不调整输出投影，模型结果通常会变化。因此需要同步调整相关的 Q/K/V 投影通道及输出投影对应通道；对 GQA 还要保持 Query Head 组和共享 KV Head 的对应关系。

**这一部分的收获：**模型结构不变，并不意味着物理数据布局不重要。算法设计必须考虑最终的 GPU 访问模式。

---

## 三、DuoAttention 消融实验带来的理解

我今天回顾了几个关键消融所对应的问题。

### 3.1 为什么不只用 Attention 分数识别重要 Head？

注意力权重只说明一个 Head **关注了哪些位置**，却无法直接给出“限制它访问历史信息后，模型最终表现受多大影响”的答案。

因为 Attention 输出还涉及 Value：

$$
O_h=\sum_i a_{h,i}V_{h,i},
$$

后续还要经过输出投影和 Transformer 层。因此，单看 Attention 权重大小，不能完整衡量某个 Head 对最终输出的影响。

DuoAttention 使用 Full/Mixed 模型的最终隐藏状态偏差作为门控优化的代理目标，侧重**输出敏感性**而不只是局部统计分数。

### 3.2 为什么使用合成 Passkey 数据？

普通语言模型训练目标中，很多位置可由局部上下文预测，未必能充分检验长期依赖。合成 Passkey 则要求模型在长文本末尾回忆随机埋入的历史内容，能够提供更明确的远距离检索监督。

我的理解是：**训练任务如何设计，会影响最后识别出什么样的“重要 Head”。**

### 3.3 Sink 与 Recent 的窗口究竟要多大？

Streaming Head 一方面需要部分初始 Attention Sink，另一方面需要近期上下文；只保留一侧可能改变其行为。

今天进一步想到一个问题：识别 Head 时使用的 $S/W$ 与正式部署时使用的 $S/W$，是否必须相同？

两种窗口的目的有所差别：

- 识别阶段的窗口影响“一个 Head 在何种约束下被判定可以 Streaming”；
- 部署阶段的窗口决定实际缓存开销，以及最终质量。

**待研究问题：**能否联合优化 Retrieval Head 比例和不同 Head 的 Streaming 窗口，而不是先固定统一窗口再完成二元分类？这只是我在阅读后提出的思考，并非已经完成的实验。

---

## 四、回顾第一篇：HARD-KV

### 4.1 研究问题：动态 Head-wise 选择与 GPU 规则布局的冲突

HARD-KV 研究的是：允许不同 Attention Head 按需求动态保留不同数量的 KV，是否能在保证重要信息的同时避免不规则索引拖慢 GPU 推理？

同一 Head 的选择集合还可能随 Decode 更新。算法上更灵活，系统上却带来动态 Block Table、索引和 Attention Kernel 的额外成本。

### 4.2 创新一：Cascade Cache

论文将 KV 的生命周期分成三个**逻辑层级**：

| 层级 | 管理对象 | 主要目的 |
|---|---|---|
| Dense Cache | 最新的 KV | 暂时保留完整近期历史，避免过早淘汰 |
| Sparse Cache | 已完成重要性筛选的历史 KV | 允许保留不规则的重要 token |
| Condensed Cache | 更早且经过整理的 KV | 减少长期执行的不规则访问开销 |

它不是 GPU、CPU、SSD 三个物理层级。这个区分也是今天复习时特别需要记住的。

### 4.3 创新二：Logits Calibration

不同选择方法会对原始 Attention logits $z$ 施加评分变换 $g$。例如 SnapKV 的聚合或 R-KV 的相似性评分，会改变分数的形状与尺度。

因此，即使同样设置 Top-p 阈值 $p$，直接把变换后分数输入 Softmax，也可能产生完全不同的保留数量。

温度 Softmax 的基础形式为：

$$
P_i(T)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.
$$

其中较低的 $T$ 通常使分布更尖锐，较高的 $T$ 使分布更平缓。

论文针对不同评分变换讨论两种温度应用顺序：

**方式一：满足相应排序保持条件的变换**

$$
\hat S_T=\operatorname{Softmax}(g(z/T)).
$$

**方式二：更一般的评分变换**

$$
\hat S_T=\operatorname{Softmax}\!\left(\frac{g(z)}{T}\right).
$$

这里需要区分：**Softmax 能将一组有限实数评分转换为概率分布，但“得到合法概率分布”不等于“恢复原始 Attention 的概率质量含义”。** Logits Calibration 要尽量保留原算法的 token 排名，同时通过寻找合适的温度，让相应 Top-k 的累计概率质量接近参考分布。

例如，原始 logits 为 $[3,2,1,0]$，若某方法把它们放大成 $[9,6,3,0]$，两者 Top-k 排名没有变化，但使用同样的 Top-p 可能选择不同数量；对后一组采用 $T=3$ 可以在这个简单例子中恢复原始分布。此例仅用于说明，不是 SnapKV 或 R-KV 的真实变换公式。

### 4.4 尚未进一步展开的部分

HARD-KV 还提出 **Index Regularization**，以应对动态 Head-wise 索引和 GPU 执行规则之间的冲突。今天主要回顾了它的动机，没有对具体 Block Table、Rewrite 与 Sparse Loading 机制进行新的深入推导，留待后续复习。

---

## 五、回顾第二篇：ThinKV

### 5.1 为什么要从 Token 走向 Thought？

ThinKV 关注长链式推理中不同思考片段的重要性差异。

作者将思考过程对应到 **Reasoning、Execution、Transition** 三类模式，并尝试利用 Attention 稀疏度来识别它们，从而采用不同的量化和淘汰策略。

这与单纯按 token 的 Attention 排名进行淘汰不同：**Thought 片段的结构和推理过程中的角色，也被纳入缓存管理。**

### 5.2 Thought Decomposition：离线 KDE 与在线分类

作者通过离线校准数据分析 Attention 稀疏度分布，使用 Kernel Density Estimation（KDE）寻找多个统计峰值之间的低谷，提取分类阈值。

KDE 的常见形式为：

$$
\boxed{\hat f(x)=\frac{1}{nh}\sum_{i=1}^{n}K\!\left(\frac{x-x_i}{h}\right)}
$$

其中 $x_i$ 是稀疏度样本，$K$ 是核函数，$h$ 是带宽。

例如，若稀疏度样本主要集中在 $0.15$、$0.48$、$0.85$ 附近，可尝试寻找群体间的两个密度低谷作为分类阈值。这些数字是教学假设，**不等于论文实际阈值**。

在线 Decode 阶段则测量选定层的稀疏度并与校准阈值比较，而不是每个新 token 都完整重做 KDE。

### 5.3 TBQ：根据 Thought 类型分配 KV 精度

我原始笔记中将其误写成了 **TBD**，正确名称是 **TBQ（Think Before You Quantize）**。

这项机制根据不同 Thought 类型的重要性采用不同精度。论文主要实验配置为 **R4E4T2**：Reasoning 和 Execution 使用 4-bit，Transition 使用 2-bit。这里的 2-bit 对应 ternary 量化，不应直接等同于普通的四级 INT2。

其意义是：不将所有 token 视为相同重要，也不单纯追求每个数的位宽越低越好。

### 5.4 TBE：渐进淘汰历史 Thought

TBE（Think Before You Evict）决定哪些历史 Thought 的 KV 需要进一步减少。它以 Thought 片段为管理单位，采用逐级降低保留量的策略，而不是一旦判为低重要性就完全删除。

例如论文使用的保留档位为：

$$
128\rightarrow64\rightarrow32\rightarrow16\rightarrow8\rightarrow4.
$$

其中 4 是所讨论配置中的最低保留量，以减少彻底丢失推理轨迹信息的风险。

### 5.5 为什么使用 post-RoPE Key 做 K-means？

当 Thought 需要从当前保留量缩减到目标数量时，TBE 对该片段中的 **post-RoPE Key** 进行 K-means 聚类，以提取能够代表不同 Key 区域的中心。

K-means 的基本流程是：

1. 初始化 $K$ 个聚类中心；
2. 按距离把每个 Key 分到某个簇；
3. 对每个簇求均值，更新 Centroid；
4. 重复以上步骤，直到簇分配和中心基本稳定。

它的目标不是判断某一次 Query 关注了哪些 token，而是尽量用有限数量的代表覆盖片段中的 Key 表示。

**这里保留一项实现细节上的谨慎表述：**今天讨论了“选择距离 Centroid 最近的原始 KV”作为一种容易理解的代表点策略，但不能仅凭这个教学规则断言论文一定使用最近邻原始点作为最终缓存。论文附录表述为簇由 Centroid Key 代表，并保留对应 Value；Centroid 与 Value 的具体对应实现还需要结合代码进一步核查。因此本文不把“必然保留最近原始 KV”写成已经证实的论文事实。

### 5.6 尚未回顾的下一项机制

今天复习主要到 TBE 的聚类淘汰。**Continuous Thinking（CT）** 的物理槽位复用、Eviction Mask 和减少 Gather/Compaction 的系统实现，是下次需要继续回顾的部分；今天没有记录新的代码或性能测试。

---

## 六、把三篇论文放在一起比较

| 论文 | 优化粒度 | 重要性依据 | 主要系统问题 |
|---|---|---|---|
| DuoAttention | KV Head | Full 改为 Streaming 后的最终输出敏感性 | Head 分类后二值化、重排及双缓存部署 |
| HARD-KV | Head / Token | 动态评分与校准后的 Top-p 预算 | 不规则动态索引和 GPU 规则执行之间的冲突 |
| ThinKV | Thought Segment / Token | Attention 稀疏模式、Thought 类型及 Key 聚类代表性 | 动态淘汰、混合量化和缓存槽位管理 |

三篇论文从不同角度说明：**缓存压缩不仅是“删多少”，也包括“根据什么选择”和“如何在 GPU 上真正高效执行”。**

---

## 七、今天最重要的理解变化

### 我原本倾向于这样理解

- 学到连续门控 $\alpha$ 后，以为直接根据数值大小切换 Head 就足够了。
- 认为 Retrieval Head 的比例可以直接等同于整体缓存的剩余比例。
- 更关注聚类产生的 Centroid，容易忽略其与实际 KV 保留和 Value 对应关系之间的区别。

### 现在的理解

- 连续 $\alpha$ 是离线优化工具，部署时必须明确 Full/Streaming 的离散执行策略，并同步调整相应通道布局。
- Streaming Head 仍有 Sink + Recent 的固定缓存成本，真实保留比例取决于窗口大小和上下文长度。
- Logits Calibration 不是单纯做一次 Softmax，而是校准不同评分方法的概率质量尺度。
- KDE 用于发现稀疏度统计模式间的边界，K-means 则用于在单个 Thought 片段内寻找 Key 表示的代表；两者解决的问题不同。

### 为什么会发生这些变化？

当我把论文公式放回实际的训练、Decode 和 GPU 内存执行流程中，才发现一个数学上看似简单的选择决策，可能受到阈值边界、缓存窗口、索引布局和代表值实现方式等因素影响。因此，读论文时需要同时核对算法目标与系统实现。

---

## 八、仍然值得进一步弄清楚的问题

1. DuoAttention 能否针对不同 KV Head 自适应设置 Sink/Recent 窗口，而不仅进行 Full/Streaming 二分类？
2. HARD-KV 的 Index Regularization 在什么 Decode 长度或稀疏度下，能摊销 Rewrite 的额外成本？
3. ThinKV 聚类后的 Centroid Key 与对应 Value，在实际代码中如何精确保存和映射？
4. 如果结合 Thought 类型、Head 输出敏感性和历史 Attention 统计，是否能提高 KV 预算分配的稳定性？

这些是**阅读后的待验证问题**，不是已经完成的研究结论。

## 九、下一步最小行动

**继续回顾 ThinKV 的 Continuous Thinking 系统机制**：重点弄清楚被淘汰 KV 的物理槽位怎样复用、Eviction Mask 怎样参与寻址，以及它为什么能减少 Gather/Compaction 开销。

> **学习状态说明：**当天内容为论文精读与概念回顾；原始记录没有提供代码运行、实验结果或 Git commit，因此本文没有补造任何实践数据。
