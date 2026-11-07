---
title: "从 Lookahead 机制到 FreqKV：DCT、IDCT 与 KV Cache 压缩手算"
slug: "freqkv-dct-idct-2026-11-07"
kind: paper
date: "2026-11-07"
summary: "整理上一阶段 Lookahead KV 重要性预测的训练思路，重点手算 FreqKV 如何沿 Token 序列做 DCT、截断低频系数、使用不同长度的 IDCT 构造压缩 KV，并推导幅值修正因子。"
tags: ["FreqKV", "KV Cache", "DCT", "IDCT", "大模型推理"]
featured: false
published: true
example: false
---

# 从 Lookahead 机制到 FreqKV：DCT、IDCT 与 KV Cache 压缩手算

> **学习日期：**2026 年 11 月 7 日  
> **阅读论文：** *FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension*  
> **记录说明：**本文是个人论文阅读与数学推导笔记。数值是用于理解机制的教学示例，不是论文实验结果，也不代表已完成代码复现。

## 一、收尾上一篇：Lookahead 重要性预测如何训练

今天首先回顾了上一篇文章末尾的训练流程。我的理解是，该方法尝试让可学习的 Lookahead Tokens 在真实回答尚未生成时，提前预测 Prompt 中哪些 KV 更重要。

整个过程可以概括为：

1. **准备训练数据：**输入 Prompt $X$，以及事先得到的参考回答 $Y$。
2. **GT Forward Pass：**将 $[X;Y]$ 输入模型，以真实回答中的 Query 对 Prompt KV 的关注情况构造目标重要性分布：

   $$
   p=\hat{s}^{\mathrm{GT}}.
   $$

3. **Lookahead Forward Pass：**将 $[X;P]$ 输入模型，使用 Lookahead Tokens $P$ 预测 Prompt KV 的重要性分布：

   $$
   q=\hat{s}^{\mathrm{LKV}}.
   $$

4. **计算分布差异：**通过 KL 散度约束预测结果接近 GT：

   $$
   D_{\mathrm{KL}}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}.
   $$

5. **反向传播：**按照我的阅读记录，训练中更新 Lookahead Embeddings 与 LoRA 参数，使预测分布逐渐接近参考分布。

这一机制的关键不只是“预测哪些 Token 重要”，还在于利用真实回答提供的监督信息训练预测器。与完全免训练的 KV 压缩方案相比，它需要额外训练。按照我当前的理解，其关注重点主要是 Prefill 后 Prompt KV 的筛选；Decode 过程中新增 KV 的持续管理仍值得进一步研究。**以上是对上一篇方法的阶段性归纳，并非已复现实验得出的结论。**

## 二、FreqKV 为什么选择频域压缩？

接下来开始阅读 *FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension*。

传统的 Token 选择式压缩可以想成：原来缓存中有 100 个 Token，只允许保留相当于 20 个 Token 的预算，那么根据重要性选出 20 个，舍弃其他 80 个。

FreqKV 提供了另一种思路：**不把压缩简单理解为删除某些 Token，而是沿 Token 序列方向把 K/V 变换到频域，保留一定数量的频率分量，再构造长度更短的压缩状态。**

两种方式的直观区别如下：

| 维度 | Token 选择式方法 | 频域压缩思路 |
|---|---|---|
| 压缩对象 | 原始 Token 对应的 KV 条目 | 整段 KV 序列的频率表示 |
| 核心操作 | 挑选、保留或丢弃条目 | DCT、频率截断、逆变换 |
| 被保留的状态 | 通常还能对应原始 Token | 不再与原始 Token 一一对应 |
| 需要关注的风险 | 被删除 Token 的信息不可用 | 高频细节丢失与重建误差 |

我的阅读记录中还提到：该方法考虑了 Prefill 之后 KV 仍会随着 Decode 持续增长的问题，采用缓存达到设定容量后再次压缩、再继续接收新增 KV 的思路。**具体压缩触发条件、缓存组织方式和算法细节仍需对照论文原文核实。**

### 2.1 为什么关注低频？

按照当天摘录的论文实验描述，作者从 CNN 文档中取样，通过少量同义词替换制造轻微文本扰动，再比较 K/V 频率分量在扰动前后的相似性。笔记中记录的设置是 **100 篇文档、约 1% 的词替换**；这一具体实验配置还需回查原文。

当前得到的理解是：**较低频的表示对这类轻微扰动可能更稳定；高频分量往往更敏感于局部变化。**这为优先保留低频提供了经验动机，但不意味着高频一定不重要。例如，精确数字、符号和局部关键事实可能依赖细节，不能直接等同于低频信息。

## 三、DCT 到底沿哪个维度进行？

这部分是今天首先需要纠正的概念：**DCT 不是对 Q 排名，不是对 Attention Head 排名，也不是沿单个 Token 的 Key 特征维度做 PCA；在这个例子中，它沿 Token 序列长度 $N$ 进行。**

设某个 KV Head 的 Key 缓存为：

$$
K=\begin{bmatrix}
1&3&5\\
2&4&6\\
3&5&7\\
4&6&8
\end{bmatrix}\in\mathbb{R}^{4\times3}.
$$

- **行：**4 个 Token 位置，$N=4$。
- **列：**3 个 Key 通道，$d_k=3$。
- **执行方式：**分别对每一列的 4 个数做 DCT，例如第一列 $x=[1,2,3,4]$；Value 的各通道也类似处理。

需要注意，完整的 DCT 并未减少元素个数；真正降低缓存规模的是后续**频率截断与较短状态的构造**。

## 四、先推导 DCT-II 和归一化系数

对长度为 $N$ 的一维序列 $x_0,\ldots,x_{N-1}$，使用正交 DCT-II：

$$
z_t=\alpha_t\sum_{n=0}^{N-1}x_n
\cos\left(\frac{\pi t(2n+1)}{2N}\right),
\qquad t=0,\ldots,N-1,
$$

其中：

$$
\boxed{\alpha_t=
\begin{cases}
\sqrt{1/N},&t=0,\\
\sqrt{2/N},&1\le t<N.
\end{cases}}
$$

### 4.1 为什么 $t=0$ 和 $t>0$ 的系数不同？

令未归一化的基函数为：

$$
b_t(n)=\cos\left(\frac{\pi t(2n+1)}{2N}\right).
$$

当 $t=0$ 时，所有 $b_0(n)=1$，所以平方长度为：

$$
\sum_{n=0}^{N-1}b_0(n)^2=N.
$$

当 $1\le t<N$ 时，使用恒等式 $\cos^2\theta=(1+\cos2\theta)/2$，可得：

$$
\begin{aligned}
\sum_{n=0}^{N-1}b_t(n)^2
&=\frac N2+\frac12\sum_{n=0}^{N-1}
\cos\left(\frac{\pi t(2n+1)}N\right)\\
&=\frac N2.
\end{aligned}
$$

最后一个等号成立，是因为对于整数 $1\le t<N$，这些余弦项的总和为零。于是，令各基向量归一化后的平方长度为 1，便得到：

$$
\alpha_0=\frac1{\sqrt N},\qquad
\alpha_{t>0}=\frac1{\sqrt{N/2}}=\sqrt{\frac2N}.
$$

对于 $N=4$：

$$
\alpha_0=\frac12,\qquad
\alpha_1=\alpha_2=\alpha_3=\frac1{\sqrt2}.
$$

**我的理解：**归一化系数不是由 Key 的实际数值学习出来的，而是由序列长度和余弦基向量的长度决定的。

## 五、手算 Key：4 个 Token 如何变成 2 个频率系数？

取矩阵第一列：

$$
x=[1,2,3,4],\qquad N=4.
$$

### 5.1 计算直流系数 $z_0$

$$
\begin{aligned}
z_0&=\frac12(1+2+3+4)\\
&=\boxed{5}.
\end{aligned}
$$

### 5.2 计算第一个非零频率系数 $z_1$

$$
\begin{aligned}
z_1=\frac1{\sqrt2}\bigg[&
1\cos\frac\pi8+2\cos\frac{3\pi}8\\
&+3\cos\frac{5\pi}8+4\cos\frac{7\pi}8\bigg]
\approx\boxed{-2.230442}.
\end{aligned}
$$

其余两个系数同理：

$$
z_2=0,\qquad z_3\approx-0.158513.
$$

| 频率系数 | 约值 | 直观含义 |
|---|---:|---|
| $z_0$ | 5.000000 | 直流分量，反映序列整体水平 |
| $z_1$ | -2.230442 | 较慢的变化趋势 |
| $z_2$ | 0 | 在本例中恰好为零 |
| $z_3$ | -0.158513 | 更快的变化成分 |

完整频域表示为：

$$
Z^K=[5,-2.230442,0,-0.158513].
$$

如果只保留 $L=2$ 个低频系数：

$$
\boxed{Z^K_{\mathrm{keep}}=[5,-2.230442].}
$$

此时保存的是**两个频率系数**，不是“Token 0 和 Token 1”。例如 $z_0$ 由全部四个位置共同贡献，不能说截断 $z_2,z_3$ 就是删掉了最后两个 Token。

## 六、两个系数为什么还能算出四个值？

这里需要严格区分两种运算。

### 6.1 长度仍为 4 的 IDCT：近似恢复原位置

如果仅把缺失系数视为 0，即：

$$
\widehat Z=[5,-2.230442,0,0],
$$

然后执行**4 点 IDCT**，得到：

$$
\begin{aligned}
\hat x_n
&=\sum_{t=0}^{3}\alpha_t\widehat z_t
\cos\left(\frac{\pi t(2n+1)}8\right)\\
&=2.5-1.577161\cos\left(\frac{\pi(2n+1)}8\right).
\end{aligned}
$$

代入 $n=0,1,2,3$：

| 位置 | 原始值 | 4 点 IDCT 重建值（约） |
|---|---:|---:|
| 0 | 1 | 1.042893 |
| 1 | 2 | 1.896447 |
| 2 | 3 | 3.103553 |
| 3 | 4 | 3.957107 |

因此：

$$
\boxed{\hat x\approx[1.043,1.896,3.104,3.957].}
$$

**为什么两个系数会产生四个值？**因为这两个系数是两个已知余弦基函数的权重，公式仍可以在四个位置 $n=0,1,2,3$ 上分别求值。得到的是**四个近似值**，而不是恢复了丢弃的全部高频信息。

### 6.2 长度变为 2 的 IDCT：构造两个压缩状态

FreqKV 在当前笔记涉及的公式中使用的是另一种操作：保留 $L=2$ 个系数后，执行**2 点 IDCT**，而不是先补零恢复到四个位置。

对于正交 2 点 IDCT：

$$
\operatorname{IDCT}_{2}([z_0,z_1])
=\left[\frac{z_0+z_1}{\sqrt2},\frac{z_0-z_1}{\sqrt2}\right].
$$

代入 Key 的两个保留系数：

$$
\begin{aligned}
K'
&=\operatorname{IDCT}_2([5,-2.230442])\\
&\approx\boxed{[1.958373,5.112695]}.
\end{aligned}
$$

这两个新数是对整段历史序列的**压缩表示**，不能当成两个被选中的原始 Token。

> **易混淆处：**4 点 IDCT 和 2 点 IDCT 是两种不同演示，**不是 Key 用 4 点、Value 用 2 点**。在对称的频域压缩示例中，Key、Value 都要经过各自的 DCT、截断、较短长度的 IDCT 和幅值修正。

## 七、Value 为什么也要做 DCT？

仍然使用便于手算的示例，令：

$$
V=[2,4,6,8]=2K.
$$

DCT 是线性的，所以：

$$
\begin{aligned}
Z^V
&=\operatorname{DCT}(2K)
=2\operatorname{DCT}(K)\\
&\approx[10,-4.460885,0,-0.317025].
\end{aligned}
$$

只保留前两个系数：

$$
Z^V_{\mathrm{keep}}=[10,-4.460885].
$$

直接执行 2 点 IDCT：

$$
\begin{aligned}
V'&=\operatorname{IDCT}_2([10,-4.460885])\\
&\approx\boxed{[3.916746,10.225390]}.
\end{aligned}
$$

在这个示例中 $V=2K$，因此变换后的 Value 仍为对应 Key 数值的两倍。真实模型中的 K/V 不必满足这种倍数关系，这只是方便计算的设定。

## 八、为什么还要乘 $\sqrt{L/N}$？

当前笔记对应的压缩重构公式为：

$$
\boxed{
\widetilde K=\sqrt{\frac LN}\operatorname{IDCT}_L(Z^K_{\mathrm{keep}}),\qquad
\widetilde V=\sqrt{\frac LN}\operatorname{IDCT}_L(Z^V_{\mathrm{keep}}).
}
$$

这里 $N=4,L=2$，所以：

$$
\sqrt{\frac LN}=\sqrt{\frac24}=\frac1{\sqrt2}\approx0.707107.
$$

### 8.1 不做幅值修正的问题

原始 Key 均值：

$$
\mu_K=\frac{1+2+3+4}{4}=2.5.
$$

直接做 2 点 IDCT 后：

$$
\mu_{K'}=\frac{1.958373+5.112695}{2}\approx3.535534.
$$

均值被放大了。这是因为正交 DCT/IDCT 的**变换长度发生变化，归一化尺度也发生了变化**。

### 8.2 从直流系数推导修正因子

正交 DCT 的直流系数满足：

$$
z_0=\frac1{\sqrt N}\sum_{n=0}^{N-1}x_n
=\sqrt N\,\mu.
$$

若改用长度 $L$ 的 IDCT，但沿用同一个 $z_0$，重构序列的均值为：

$$
\mu'=\frac{z_0}{\sqrt L}.
$$

设需要乘一个系数 $c$ 使均值回到原始水平：

$$
c\frac{z_0}{\sqrt L}=\frac{z_0}{\sqrt N}
\quad\Longrightarrow\quad
\boxed{c=\sqrt{\frac LN}}.
$$

因此该修正项至少有一个明确的数学作用：**抵消变换长度改变造成的直流/均值尺度偏差**。

### 8.3 Key 和 Value 的最终压缩数值

$$
\begin{aligned}
\widetilde K
&=\frac1{\sqrt2}[1.958373,5.112695]\\
&\approx\boxed{[1.384779,3.615221]},\\[6pt]
\widetilde V
&=\frac1{\sqrt2}[3.916746,10.225390]\\
&\approx\boxed{[2.769558,7.230442]}.
\end{aligned}
$$

检查均值：

$$
\frac{1.384779+3.615221}{2}=2.5,
\qquad
\frac{2.769558+7.230442}{2}=5.
$$

| 对象 | 原始序列 | 压缩后两个状态 | 原始均值 / 压缩后均值 |
|---|---|---|---|
| Key | $[1,2,3,4]$ | $[1.384779,3.615221]$ | $2.5/2.5$ |
| Value | $[2,4,6,8]$ | $[2.769558,7.230442]$ | $5/5$ |

**要注意：均值保持不代表信息无损，也不代表 Attention 输出必然相同。**高频被截断后，序列细节已经发生变化；这种误差对不同任务的影响仍然需要实验检验。

## 九、今天的核心理解：从“删除 Token”到“压缩历史表示”

今天最重要的理解变化有三点：

1. **原先容易把压缩后的两个数理解成“保留下来的两个 Token”。**现在知道，频率系数由多个原始位置共同贡献，长度为 2 的重建状态不与某两个原始 Token 一一对应。
2. **原先容易把“两个系数重建四个数”和“生成两个压缩状态”混成同一个过程。**现在可以区分：前者是补零后做 4 点 IDCT 的近似恢复；后者是对保留系数做 2 点 IDCT，输出数量确实减少到 2。
3. **原先只记住了幅值修正公式。**现在理解 $\sqrt{L/N}$ 可以从直流分量 $z_0=\sqrt N\mu$ 出发推导，作用是让不同变换长度下的平均水平保持一致，而不是把丢失的信息补回来。

从边缘推理的角度看，这类方法值得关注的原因是 KV Cache 通常随上下文变长而增长；如果能在预算限制内维护更短的 K/V 表示，就有机会减少缓存占用。但**实际收益还取决于变换成本、压缩频率、数据搬运、注意力计算方式及精度损失**，目前还不能仅凭四个数的手算判断运行性能。

## 十、还需要继续验证的问题

- **Attention 计算：**新 Query 到来以后，如何利用 $\widetilde K,\widetilde V$ 计算注意力？相比原始四组 K/V，Softmax 权重和输出的误差有多大？
- **Decode 阶段：**缓存什么时候触发再次压缩？多次压缩后误差如何累积？
- **压缩策略：**是否总是保留低频，还是还会考虑层、Head、位置或任务差异？需要回查论文实现。
- **复现实验：**目前只完成了公式推导和小规模手算，尚未进行代码复现，也没有报告真实的延迟、显存占用或准确率结果。

**下一步最小行动：**固定一个 Query，分别使用原始 $K,V$ 与压缩后 $\widetilde K,\widetilde V$ 手算一次 Attention，比较输出差异，再对照 FreqKV 论文的实际 Attention 流程确认该教学示例的适用范围。

---

*整理说明：本文根据本人 2026-11-07 的学习记录编写，使用 AI 辅助优化行文、组织推导和复核示例计算。对论文的机制概括仍需结合原文核对，不将教学手算视为真实实验结果。*
