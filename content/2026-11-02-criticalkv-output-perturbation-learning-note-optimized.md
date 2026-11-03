---
title: "从 Attention 权重到输出扰动：我对 CriticalKV 两阶段缓存淘汰的理解"
slug: "2026-11-02-criticalkv-output-perturbation-learning-note"
kind: "paper"
date: "2026-11-02"
summary: "精读 CriticalKV 后，我进一步理解了为什么注意力权重不足以独立判断 KV 的重要性，并通过输出扰动、Softmax 重归一化、L1 误差上界和两阶段 Top-K 手算梳理了关键 KV 选择过程。"
tags: ["论文精读", "CriticalKV", "KV Cache", "Attention", "输出扰动", "L1 范数", "Top-K", "LLM 推理"]
featured: false
published: true
example: false
---

# 今天读了什么？

今天开始学习 **CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective**（ICML 2026）。

在上一篇 Ada-KV 中，我主要学习的是：**当 KV Cache 总预算有限时，应该怎样在不同 Head 之间分配缓存名额？** Ada-KV 通过 Attention 权重建立淘汰误差上界，再用跨 Head 的预算分配降低这个上界。

但 CriticalKV 进一步追问了一个我之前没有认真区分的问题：

> **Attention 权重高，就一定意味着这个 KV 被删除后会造成更大的输出误差吗？**

这篇文章的切入点是 **Attention 输出扰动（Output Perturbation）**。作者认为，Key 决定 Query 的匹配权重，而 KV 的最终作用还取决于 Value 和输出投影矩阵。因此，不能只根据 Attention 权重判断缓存条目的重要性。

我的理解逐渐从“保留更多高权重 KV”转向“在固定预算下，尽量让压缩后的 Attention 输出接近完整输出”。

> **阅读范围说明**：本篇记录我今天学习的 Definition 3.1、Theorem 3.2、Theorem 3.3、Assumption 3.4、Theorem 3.5 与 Algorithm 1 的主要机制。文中的小数组与手算是教学模拟，并非论文实际实验数据；今天没有记录代码复现或真实性能测试。

# 一、先重新理解：为什么不能只看 Attention 权重？

单个 Attention Head 的输出写作：

\[
A=\operatorname{Softmax}\left(\frac{qK^\top}{\sqrt{d_h}}\right),
\qquad o=AVW^O
\]

其中，\(A_i\) 表示当前 Query 对第 \(i\) 个 Key 的注意力权重，\(V_i\) 是对应 Value，\(W^O\) 是该 Head 的输出投影矩阵。

为了方便分析，定义投影后的 Value：

\[
U=VW^O,\qquad U_i=V_iW^O
\]

于是：

\[
\boxed{o=\sum_{i=1}^{n}A_iU_i}
\]

这让我意识到，**Attention 权重描述的是“关注多少”，投影后的 Value 决定的是“关注后向输出传递什么”**。这两部分需要合在一起看。

## 1.1 一个让我改变看法的手算

假设有三个历史 KV，当前 Query 的注意力权重为：

\[
A=[0.60,\ 0.25,\ 0.15]
\]

把投影后 Value 简化为标量：

\[
U=[1,\ 2,\ 12]
\]

原始输出为：

\[
\begin{aligned}
o&=0.60\times1+0.25\times2+0.15\times12\\
 &=0.60+0.50+1.80\\
 &=\boxed{2.90}
\end{aligned}
\]

| KV | Attention \(A_i\) | 投影后 Value \(U_i\) | 原始加权项 \(A_iU_i\) |
|---|---:|---:|---:|
| KV1 | 0.60 | 1 | 0.60 |
| KV2 | 0.25 | 2 | 0.50 |
| KV3 | 0.15 | 12 | **1.80** |

KV3 的权重最小，但对应的原始加权项最大。它提醒我：**高 Attention 并不自动等于高输出贡献**。不过，真正判断删除某个 KV 的误差，还必须考虑剩余权重的重新归一化，不能仅凭表中最后一列下结论。

## 1.2 同样保留两个 KV，结果为什么不同？

设预算 \(b=2\)。

**方案 A：传统 Attention Top-2**，保留 KV1、KV2。两者原始权重之和是：

\[
S_A=0.60+0.25=0.85
\]

删除 KV3 后，剩余权重重新归一化：

\[
\hat o_A=\frac{0.60\times1+0.25\times2}{0.85}
=\frac{1.10}{0.85}\approx\boxed{1.2941}
\]

实际输出扰动：

\[
L_A=|2.90-1.2941|\approx\boxed{1.6059}
\]

**方案 B：保留 KV1、KV3**，原始权重之和：

\[
S_B=0.60+0.15=0.75
\]

压缩后输出：

\[
\hat o_B=\frac{0.60\times1+0.15\times12}{0.75}
=\frac{2.40}{0.75}=\boxed{3.20}
\]

相应扰动：

\[
L_B=|2.90-3.20|=\boxed{0.30}
\]

| 选择策略 | 保留位置 | 压缩后输出 | 实际 L1 扰动 |
|---|---|---:|---:|
| Attention Top-2 | KV1、KV2 | 1.2941 | 1.6059 |
| 教学替代选择 | KV1、KV3 | 3.2000 | **0.3000** |

通过这个例子，我确认：**简单的 Attention Top-K 不一定使实际输出误差最小**。这里的“保留 KV1、KV3”只是演示问题的替代选择，并不意味着任意 KV 分布下 CriticalKV 必然选中这组。

# 二、Definition 3.1：到底怎样定义“关键 KV”？

过去我比较容易把“关键 KV”直接理解为“Attention Score 排名靠前的 KV”。

这篇论文给出了一个更明确的定义：从 \(n\) 个历史 KV 中选出 \(b\) 个，使压缩前后 Attention 输出的 L1 距离尽可能小。

\[
\boxed{\min_{\hat K,\hat V}\ L=\|o-\hat o\|_1}
\]

其中：

- \(o\)：完整历史 KV 参与计算时的 Attention 输出。
- \(\hat o\)：压缩后只使用选中 KV 得到的输出。
- \(b\)：允许保留的 KV 条目数量。
- \(L\)：输出扰动，而不是任务准确率或模型训练损失。

这个定义只告诉我们**要优化什么**，并没有立即告诉我们怎样高效找到最优子集。直接枚举所有保留组合的代价很高，所以作者随后引入可计算的理论上界。

# 三、Theorem 3.2：删除 KV 为什么必须重新归一化？

作者用二值变量表示保留决策：

\[
N_i=\begin{cases}
1,&\text{保留第 }i\text{ 个 KV}\\
0,&\text{淘汰第 }i\text{ 个 KV}
\end{cases}
\qquad \sum_i N_i=b
\]

在 Softmax 中，被删除位置相当于将其 logit 设为 \(-\infty\)。作者证明，压缩后的 Attention 权重可以直接用原始权重写成：

\[
\boxed{A_i'=\frac{N_iA_i}{\sum_jN_jA_j}}
\]

记：

\[
S=\sum_jN_jA_j
\]

那么 \(S\) 是保留 KV 的**原始 Attention 权重总和**，而不是最后重新归一化后的权重之和（后者始终是 1）。

例如：

\[
A=[0.6,0.3,0.1],\quad N=[1,1,0]
\]

则：

\[
S=0.6+0.3=0.9
\]

\[
A'=\left[\frac{0.6}{0.9},\frac{0.3}{0.9},0\right]
=\left[\frac23,\frac13,0\right]
\]

**我现在的理解：删除一个 KV 不仅使它自己的贡献消失，还会改变其他所有保留 KV 的归一化权重。** 因此不能简单地用原始输出减去被删除位置的 \(A_iU_i\) 来计算新输出。

# 四、Theorem 3.3：输出扰动上界怎样一步步推导？

这部分是今天花时间最多的公式。我把完整推导保留下来，方便以后回看。

定义：

\[
U_i=V_iW^O,\qquad u_i=\|U_i\|_1
\]

以及：

\[
S=\sum_iN_iA_i,\qquad C=\sum_iA_iu_i
\]

这里的 \(C\) 是**所有原始 Attention 权重乘投影后 Value 范数的和**，与上一篇 Ada-KV 使用的统一范数界常数不能直接混为一谈。

## 步骤 1：写出压缩前后输出

\[
o=\sum_iA_iU_i
\]

\[
\hat o=\sum_i\frac{N_iA_i}{S}U_i
\]

两者之差：

\[
o-\hat o=\sum_i\left(A_i-\frac{N_iA_i}{S}\right)U_i
\]

## 步骤 2：对 L1 范数使用三角不等式

\[
\begin{aligned}
L&=\left\|\sum_i\left(A_i-\frac{N_iA_i}{S}\right)U_i\right\|_1\\
&\leq\sum_i\left|A_i-\frac{N_iA_i}{S}\right|\|U_i\|_1\\
&=\sum_i\left|A_i-\frac{N_iA_i}{S}\right|u_i
\end{aligned}
\]

这里的意义是：把难以直接分析的**向量输出差异**，转成各 KV 对应的**标量项之和**。但因为用了三角不等式，所得是上界，不一定等于实际误差。

## 步骤 3：分开讨论删除与保留

对被删除的 KV，\(N_i=0\)：

\[
\left|A_i-\frac{0\cdot A_i}{S}\right|u_i=A_iu_i
\]

对保留的 KV，\(N_i=1\)，且 \(0<S\leq1\)：

\[
\left|A_i-\frac{A_i}{S}\right|u_i
=A_i\left(\frac1S-1\right)u_i
\]

因此：

\[
\theta=\sum_{N_i=0}A_iu_i
+\left(\frac1S-1\right)\sum_{N_i=1}A_iu_i
\]

## 步骤 4：合并为论文的上界公式

因为：

\[
\sum_{N_i=0}A_iu_i=C-\sum_{N_i=1}A_iu_i
\]

所以：

\[
\begin{aligned}
\theta
&=C-\sum_{N_i=1}A_iu_i
+\left(\frac1S-1\right)\sum_{N_i=1}A_iu_i\\
&=C-\left(2-\frac1S\right)\sum_{N_i=1}A_iu_i
\end{aligned}
\]

最终得到：

\[
\boxed{L\leq\theta=C-\left(2-\frac1S\right)\sum_iN_iA_iu_i}
\]

**这就是我今天最想真正弄明白的理论式。** 它同时包含保留的 Attention 质量 \(S\)，以及保留条目的 \(A_i\|V_iW^O\|_1\)，而不仅是 Attention 权重本身。

# 五、把理论上界代入具体数字

为保持和上面的推导一致，再使用一组用于核对上界的正标量数据：

\[
A=[0.6,0.3,0.1],\quad U=[10,20,30],\quad N=[1,1,0]
\]

由于数值为正：

\[
u=[10,20,30]
\]

完整输出：

\[
o=0.6\times10+0.3\times20+0.1\times30=15
\]

压缩输出：

\[
\hat o=\frac{0.6\times10+0.3\times20}{0.9}
=\frac{12}{0.9}=13.3333
\]

真实输出扰动：

\[
L=|15-13.3333|\approx\boxed{1.6667}
\]

接下来分别计算三个中间量：

\[
C=0.6\times10+0.3\times20+0.1\times30=\boxed{15}
\]

\[
S=0.6+0.3=\boxed{0.9}
\]

\[
\sum_iN_iA_iu_i=0.6\times10+0.3\times20=\boxed{12}
\]

代入 Theorem 3.3：

\[
\begin{aligned}
\theta&=15-\left(2-\frac1{0.9}\right)\times12\\
&=15-0.8889\times12\\
&\approx\boxed{4.3333}
\end{aligned}
\]

于是：

\[
\boxed{L\approx1.6667\leq\theta\approx4.3333}
\]

这个结果让我明确区分两件事：

- **实际输出扰动 \(L\)**：必须根据压缩前后的输出计算。
- **理论上界 \(\theta\)**：用于设计可优化的选择准则，但通常比实际误差更宽松。

# 六、为什么还需要“先 Attention、再 Value”的两阶段选择？

起初我想：既然上界中已经出现了 \(A_iu_i\)，那直接对所有 KV 按照这个乘积排序不就可以了吗？

后来我发现，公式前面还有一项：

\[
2-\frac1S
\]

当保留的 Attention 质量太低时，这个系数可能为负：

\[
S=0.4\quad\Rightarrow\quad 2-\frac1{0.4}=-0.5
\]

为了给第二阶段建立可证明的优化条件，论文引入 **Assumption 3.4**，要求第一阶段选中条目的原始 Attention 权重累计达到：

\[
\boxed{\sigma=\sum_iN_i'A_i>0.5}
\]

这里需要纠正一个容易写反的判断：**不是 \(S<0.5\) 时就有利于第二阶段，而是 \(\sigma>0.5\) 时，\(2-1/\sigma\) 才为正。**

所以第一阶段的意义不是“只相信 Attention”，而是先确保足够的注意力质量。第二阶段才在剩余 KV 中利用投影后 Value 的信息进一步选择。

这也是一种理论条件驱动的折中，而不是“Value 越大必然越重要”的简单规则。

# 七、Algorithm 1：从 6 个 KV 里选择 4 个的完整手算

论文的两阶段预算为：

\[
b'=\alpha b,\qquad b''=b-b'
\]

主实验采用 \(\alpha=0.5\)。注意：**这个 \(\alpha\) 是第一阶段的预算比例，不是 Ada-KV 的预算平滑系数，也不是 DuoAttention 的可训练门控参数。**

为了手算，设有 6 个 KV，且：

\[
A=[0.40,0.25,0.15,0.10,0.06,0.04]
\]

\[
u=[1,2,8,3,15,4]
\]

总缓存预算 \(b=4\)，因此：

\[
b'=0.5\times4=2,\qquad b''=4-2=2
\]

| KV | 原始 Attention \(A_i\) | \(u_i=\|V_iW^O\|_1\) | \(A_iu_i\) |
|---|---:|---:|---:|
| KV1 | 0.40 | 1 | 0.40 |
| KV2 | 0.25 | 2 | 0.50 |
| KV3 | 0.15 | 8 | 1.20 |
| KV4 | 0.10 | 3 | 0.30 |
| KV5 | 0.06 | 15 | 0.90 |
| KV6 | 0.04 | 4 | 0.16 |

## Stage 1：先按 Attention Top-2

根据 \(A_i\) 选出 KV1、KV2：

\[
\boxed{\mathcal K_1=\{KV1,KV2\}}
\]

第一阶段的累计注意力质量：

\[
\sigma=0.40+0.25=\boxed{0.65}>0.5
\]

因此满足本次手算所需的 Assumption 3.4。

## Stage 2：在剩余 KV 中做联合评分

论文的工程评分包含微小平滑项：

\[
\boxed{s_i^{(2)}=(A_i+10^{-4})u_i}
\]

Stage 1 已选中的 KV 不再参与第二阶段。对剩余四个位置：

\[
\begin{aligned}
s_3^{(2)}&=(0.15+0.0001)\times8=\boxed{1.2008}\\
s_4^{(2)}&=(0.10+0.0001)\times3=\boxed{0.3003}\\
s_5^{(2)}&=(0.06+0.0001)\times15=\boxed{0.9015}\\
s_6^{(2)}&=(0.04+0.0001)\times4=\boxed{0.1604}
\end{aligned}
\]

最大的两个为 KV3、KV5：

\[
\boxed{\mathcal K_2=\{KV3,KV5\}}
\]

最终：

\[
\boxed{\mathcal K=\mathcal K_1\cup\mathcal K_2=\{KV1,KV2,KV3,KV5\}}
\]

被淘汰的则是 KV4、KV6。

如果只做 Attention Top-4，得到的是 KV1、KV2、KV3、KV4。CriticalKV 在这个示例里选择了权重更小、但投影后 Value 范数更大的 KV5。

**注意**：这组数据展示的是两阶段**如何执行**，不是证明它在此处一定取得最小的实际输出误差。要验证实际误差，需要保留投影后 Value 的完整向量以及计算对应的压缩输出。

# 八、Theorem 3.5：第二阶段为什么可以用 \(A_i u_i\) 排名？

当第一阶段已经确定，且：

\[
\sigma>0.5
\]

论文为第二阶段推导的可优化上界具有形式：

\[
\boxed{\hat\theta=C'-\left(2-\frac1\sigma\right)\sum_iN_i''A_iu_i}
\]

其中 \(C'\) 和 \(\sigma\) 在第二阶段选择时固定，而 \(N_i''\) 是第二阶段的选择变量。

因为 \(\sigma>0.5\)，所以：

\[
2-\frac1\sigma>0
\]

因此：

\[
\min\hat\theta\quad\Longleftrightarrow\quad\max\sum_iN_i''A_iu_i
\]

在第二阶段名额固定为 \(b''\) 时，从剩余 KV 中挑选 \(A_iu_i\) 最大的 \(b''\) 个位置即可最小化这一进一步推导的上界。

这解释了两个阶段的具体分工：

```text
输入：原始 Attention、KV、输出投影、缓存预算
                  |
                  v
Stage 1：按 Attention Top-k 保留一部分 KV
                  |
        争取使 sigma > 0.5
                  |
                  v
移除 Stage 1 已选中条目
                  |
                  v
Stage 2：对剩余 KV 计算
        (Attention + 1e-4) × 投影后 Value 的 L1 范数
                  |
                  v
选出剩余名额，合并两阶段选择结果
```

这里我尤其要记住：**第二阶段优化的是推导出的扰动上界，并非保证实际 \(L\) 在每种情形下都全局最优。** 此外，如果第一阶段没有满足 \(\sigma>0.5\)，相应理论条件就不自动成立，不能将论文经验结果理解成对所有模型的数学保证。

# 九、它和 Ada-KV、SnapKV、HeadKV 怎样结合？

CriticalKV 并不需要彻底替换已有的 KV 管理系统。

我的理解是：

| 方法 | 主要回答的问题 | 与 CriticalKV 的关系 |
|---|---|---|
| SnapKV | 如何借助 Observation Window 估计旧 KV 的重要性？ | 保留观察窗口式评分流程，改进最后的选择规则 |
| Ada-KV | 不同 Head 该分到多少缓存预算？ | 先分预算，再在每个 Head 内用 CriticalKV 选择条目 |
| HeadKV | 如何根据 Head 特征确定缓存分配？ | 可以沿用预算机制并替换条目选择方式 |
| CriticalKV | 哪些 KV 更能保持 Attention 输出？ | 提供两阶段的关键 KV 选择模块 |

因此，这几篇论文可以串成一条研究路线：

```text
DuoAttention / RazorAttention
  → 判断哪些 Head 更依赖历史信息

Ada-KV
  → 在不同 Head 之间分配缓存名额

CriticalKV
  → 在给定名额下，进一步筛选具体 KV
```

今天阅读的重点是方法原理和数学推导。至于论文在真实模型上减少多少精度损失、引入多少额外运行成本，我尚未把实验表格逐项核对，暂不在这篇学习记录中写成个人已验证结论。

# 十、今天最重要的理解变化

## 我原本以为

我之前比较容易把 KV Cache 淘汰理解为：

```text
算出每个历史 Token 的 Attention 权重
             ↓
按权重从大到小排序
             ↓
保留 Top-K 就能最大程度保证输出
```

而且在 Ada-KV 中学习到“保留更多原始 Attention 质量可以降低理论上界”之后，我一度觉得只要优化 Head 预算，再按 Attention 排序就够了。

## 现在我的理解是

完整 Attention 输出并不是只由 \(A\) 决定的：

\[
\boxed{o=AVW^O}
\]

所以至少要分清三个层次：

1. **Key 与 Query 的关系**：决定 Attention 权重。
2. **Value 与输出投影**：决定各位置传递到输出空间的信息。
3. **Softmax 重新归一化**：删除任意 KV 后，会改变剩余位置的权重分配。

因此，关键 KV 不应简单理解为“权重最大的 KV”，而应联系**压缩前后的输出扰动**来判断。

## 为什么我的理解发生了变化

最直接的原因就是第一个手算：权重只有 0.15 的 KV3，因为对应的投影后 Value 是 12，在保留它后反而能够显著降低构造示例的输出误差。

更进一步，Theorem 3.3 让我看到，作者不是只提出一个启发式乘积分数，而是从输出差异、掩码归一化、三角不等式逐步构造了上界，再通过 Assumption 3.4 和 Theorem 3.5 解释两阶段贪心策略。

> **今天最值得记住的一句话：Attention 权重告诉我“看哪里”，但判断 KV 是否值得保留，还需要考虑这个位置实际向输出传递了什么，以及删除它之后其他位置的权重怎样改变。**

# 十一、我还没有完全弄懂的问题

1. 第一阶段固定 \(\alpha=0.5\) 只是实验中的通用选择。不同模型或 Head 的 Attention 分布差异很大时，能否自适应决定第一阶段预算，并可靠保证 \(\sigma>0.5\)？
2. 计算 \(\|V_iW^O\|_1\) 会带来额外的计算和访存开销。实际系统如何复用投影结果、降低这一开销？
3. 第二阶段优化的是上界 \(\hat\theta\)，它与真实 L1 输出扰动之间的差距在什么条件下较小？
4. 当多个 Query Heads 共享同一组 KV（GQA）时，如何汇总不同 Query Heads 的重要性，避免互相矛盾的淘汰决策？
5. 如果未来 Query 和当前 Observation Window 的关注模式变化明显，离线或 Prefill 后一次性选择的 KV 是否仍然可靠？

# 十二、下一步最小行动

下一次准备集中阅读 **CriticalKV 的实验与消融**，重点核对以下关系：

```text
两阶段关键 KV 选择
        ↓
理论输出扰动上界是否下降？
        ↓
实际 Attention 输出误差是否同步下降？
        ↓
RULER / LongBench 任务分数能否改善？
        ↓
额外计算是否抵消 Decode 收益？
```

我尤其想结合 Ada-KV 的实验一起比较：**同样的缓存预算下，究竟是“分配名额”带来的收益更大，还是“选对具体 KV”带来的收益更大？** 这可能是后续深入研究 KV Cache 压缩时值得持续追问的问题。

---

*整理说明：本文根据当天的论文学习原始记录编辑，公式推导和教学案例经 AI 辅助整理；文中没有将教学手算、计划中的验证或论文结果表述为已经完成的个人实验。*
