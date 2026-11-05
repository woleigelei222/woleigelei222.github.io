---
title: "LookaheadKV 学习笔记：从未来重要性 GT 到 Lookahead Tokens、LoRA 与 KL 训练"
slug: "2026-11-05-lookaheadkv-gt-lora-kl-learning-note"
kind: "paper"
date: "2026-11-05"
summary: "沿着 DefensiveKV 的历史风险估计问题，学习 LookaheadKV 如何用可学习软 Token 和选择性 LoRA 在不生成草稿的情况下预测未来 KV 重要性，并逐步手算 GT、Attention、低秩投影与 KL 散度。"
tags: ["论文精读", "LookaheadKV", "KV Cache", "Attention", "LoRA", "KL散度", "大模型推理"]
featured: false
published: true
example: false
---

# LookaheadKV：不生成草稿，如何预测未来真正需要的 KV？

> **论文**：*LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation*  
> **发表**：ICLR 2026（正式会议论文）  
> **作者团队**：Jinwoo Ahn、Ingyu Seong、Akhil Kedia、Junhan Kim、Hyemi Jang、Kangwook Lee、Yongkweon Jeon；Samsung Research  
> **今日阅读重点**：研究动机、GT Importance、Learnable Lookahead Tokens、Selective Lookahead LoRA、KL 散度训练  
> **记录说明**：本文由本人原始学习记录整理，保留已学习的机制、例子和理解变化；补齐的手算均注明为**教学假设**，不代表本人进行了论文复现实验，也不代表原论文报告了这些数值。

## 01 / 研究动机：只看过去的 Query，为什么仍然可能误删 KV？

今天继续沿着前几篇 KV Cache 压缩论文的研究路线阅读。之前研究过的 DefensiveKV 给我留下的一个重要认识是：**某个 KV 即使只被少数历史 Query 高度关注，也不应该因为平均分较低就轻易淘汰。** 它通过历史峰值和先验风险修正，尽量缓解历史观测不充分造成的低估。

但是这仍然存在一个问题：无论取平均值还是最大值，评分都依赖已经观察到的 Query。**未来真实回答会提出怎样的检索需求，在 Prefill 结束前并不知道。** 即使观察了最近的一段 Query，也不能保证这些 Query 恰好反映回答真正会关注的内容。

LookaheadKV 的出发点正是这一矛盾：

> 与其只根据 Prompt 内部的历史 Query 估计重要性，能不能在真正生成答案之前，先预测**未来回答对 Prompt KV 的关注分布**？

### 1.1 / 为什么不用草稿答案直接判断？

已有一类思路是先生成一段近似回答，再从草稿 Query 对原始 Prompt Key 的 Attention 反推重要性。例如，在「阅读长论文，提炼三个创新」的场景中，草稿可能已经提到算法定义、关键公式和实验结果，因而能更准确地指出应当保留的历史 KV。

问题在于，生成草稿本身需要计算：

1. 部分方法需要草稿模型额外处理输入，形成 Prefill 状态；
2. 草稿需要逐 Token 自回归生成，产生连续的 Decode 开销；
3. 生成草稿后，还要计算重要性评分、选出 KV，再开始正式回答。

这些操作发生在用户看到正式回答的**首个 Token 之前**，因而会增加 TTFT（Time to First Token）。不同草稿方法的具体步骤并不完全相同，不能笼统认为每种方法都执行同样的双重 Prefill。

**我的理解变化**：过去我主要比较「KV 打分是否准确」，现在开始把「为了获得这份打分，必须付出多少 Prefill/Decode 时间」也作为评价标准。高质量评分如果依赖漫长的草稿生成，实际部署仍可能不划算。

## 02 / 任务定义：真正的 GT 重要性是怎样算出来的？

在 LookaheadKV 中，GT 指 *Ground Truth Importance Scores*，即以**目标模型真实生成回答时的 Attention 统计**为监督目标的 KV 重要性分数。这里的 GT 不是人工给 KV 贴的标签，也不是看某个词是否直接出现在答案里。

设原始输入 Prompt 为：

$$
X=(x_1,x_2,\ldots,x_{n_{\mathrm{in}}})
$$

目标模型生成的完整回答为：

$$
Y=(y_1,y_2,\ldots,y_{n_{\mathrm{out}}})
$$

对某一层、某一个 Attention Head，完整序列的注意力矩阵记为：

$$
A=\operatorname{Softmax}\left(\frac{QK^{\top}}{\sqrt{d_h}}\right)
$$

在因果注意力掩码下，作者提取的是**回答中 Query 对输入 Prompt 中 Key 的 Attention**，再沿着回答 Token 维度求平均：

$$
\boxed{
s_j^{\mathrm{GT}}
=\frac{1}{n_{\mathrm{out}}}
\sum_{i=n_{\mathrm{in}}+1}^{n_{\mathrm{in}}+n_{\mathrm{out}}}A_{i,j},
\qquad 1\le j\le n_{\mathrm{in}}
}
$$

| 符号 | 具体含义 |
| --- | --- |
| $n_{\mathrm{in}}$ | 输入 Prompt 的 Token 数 |
| $n_{\mathrm{out}}$ | 真实生成回答的 Token 数 |
| $A_{i,j}$ | 第 $i$ 个回答 Query 对第 $j$ 个输入 Key 的 Attention 权重 |
| $s_j^{\mathrm{GT}}$ | 输入第 $j$ 个 KV 在真实回答下的平均 Attention 重要性 |

注意两个层级：**评分对象是 Prompt 的 KV；评分的观察者是未来实际回答的 Query。** 这与仅取 Prompt 末尾 Query 进行评估的方法不同。

### 2.1 / 手算：三个输入 KV、两个真实回答 Query

为了展示式子的每一步，我将输入和回答分别简化成 3 个、2 个 Token：

$$
X=[x_1,x_2,x_3],\qquad Y=[y_1,y_2]
$$

假设某层某个 Head 的完整 Attention 已经过 Softmax，回答部分的权重为下表（**教学构造，不是原论文的实测 Attention**）：

| 回答 Query | $K_{x_1}$ | $K_{x_2}$ | $K_{x_3}$ | $K_{y_1}$ | $K_{y_2}$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| $Q_{y_1}$ | 0.500 | 0.250 | 0.125 | 0.125 | 不可见 |
| $Q_{y_2}$ | 0.400 | 0.200 | 0.100 | 0.100 | 0.200 |

每行对**可关注位置**求和都为 1。因为我们只计算输入 KV 的重要性，因此保留前三列：

$$
A_{Y\rightarrow X}=\begin{bmatrix}
0.50 & 0.25 & 0.125\\
0.40 & 0.20 & 0.100
\end{bmatrix}
$$

对每列求平均：

$$
\begin{aligned}
s_1^{\mathrm{GT}}&=(0.50+0.40)/2=0.45,\\
s_2^{\mathrm{GT}}&=(0.25+0.20)/2=0.225,\\
s_3^{\mathrm{GT}}&=(0.125+0.100)/2=0.1125.
\end{aligned}
$$

所以这个 Head 得到：

$$
\boxed{s^{\mathrm{GT}}=[0.45,\ 0.225,\ 0.1125]}
$$

如果有机会提前知道真实回答，那么以这个 GT 向量做 Top-2，理论上会保留 KV1、KV2。但**真实推理时恰恰不知道回答 $Y$**，无法直接使用 GT 做在线淘汰；GT 的作用是**离线训练监督**。

### 2.2 / 为什么 GT 的分数之和不一定等于 1？

$$
0.45+0.225+0.1125=0.7875\ne1
$$

原因是回答 Query 还能关注回答中已经出现的位置；我们只截取了它对输入 Prompt 的 Attention 子矩阵。这并不违反 Softmax 归一化，因为完整 Attention 行本身仍然和为 1。

论文在计算 KL 损失之前，对这个重要性向量做 L1 归一化：

$$
\widehat s_j^{\mathrm{GT}}
=\frac{s_j^{\mathrm{GT}}}{\lVert s^{\mathrm{GT}}\rVert_1}
$$

于是上面的教学数据变成：

$$
\widehat s^{\mathrm{GT}}
=\left[\frac{0.45}{0.7875},\frac{0.225}{0.7875},\frac{0.1125}{0.7875}\right]
=\boxed{[0.5714,\ 0.2857,\ 0.1429]}
$$

**需要区分两次归一化**：Softmax 是逐个 Query 对其允许关注的 Key 做归一化；L1 归一化是对最后得到的 *Prompt KV 重要性向量* 做归一化，以供 KL 比较。二者不是同一个操作。

## 03 / 第一项机制：Learnable Lookahead Tokens 代替草稿 Query

论文并不需要在线先生成完整回答，而是在输入 Prompt 后面拼接一组**可训练的软 Token**：

$$
\boxed{\widehat X=[X;P]},\qquad
P=[p_1,p_2,\ldots,p_{n_{\mathrm{lookahead}}}]
$$

其 Embeddings 在训练前初始化，再通过后续的损失反向传播优化。论文主要实验采用 **32 个 Lookahead Tokens**；这不意味着「预测未来的 32 个文字 Token」，而是用 32 个可学习向量汇总潜在的未来检索关注模式。

### 3.1 / 简化输入序列

为便于观察，假设只有三个原始输入位置和两个软 Token：

```text
原始 Prompt：     x1   x2   x3
额外软 Token：                 p1   p2
Prefill 序列：   x1   x2   x3   p1   p2
```

经过 Transformer 后，重点提取 **$Q_{p_1},Q_{p_2}$ 对 $K_{x_1},K_{x_2},K_{x_3}$** 的 Attention，而不是让 $p_1,p_2$ 输出一段自然语言草稿。

### 3.2 / 手算：从 Lookahead Query 得到预测评分

假设训练后，某层某 Head 的 Attention 权重为（**教学假设**）：

| Lookahead Query | KV1 | KV2 | KV3 | $p_1$ | $p_2$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| $Q_{p_1}$ | 0.60 | 0.10 | 0.20 | 0.10 | 不可见 |
| $Q_{p_2}$ | 0.20 | 0.50 | 0.15 | 0.05 | 0.10 |

由于因果掩码，$p_1$ 无法关注后面的 $p_2$。只提取 Prompt 的前三列：

$$
A_{P\rightarrow X}=\begin{bmatrix}
0.60&0.10&0.20\\
0.20&0.50&0.15
\end{bmatrix}
$$

论文预测分数的定义是：

$$
\boxed{
\widetilde s_j
=\frac{1}{n_{\mathrm{lookahead}}}
\sum_{i=n_{\mathrm{in}}+1}^{n_{\mathrm{in}}+n_{\mathrm{lookahead}}}
A_{i,j}^{\mathrm{LKV}}
}
$$

逐列求平均：

$$
\begin{aligned}
\widetilde s_1&=(0.60+0.20)/2=0.40,\\
\widetilde s_2&=(0.10+0.50)/2=0.30,\\
\widetilde s_3&=(0.20+0.15)/2=0.175.
\end{aligned}
$$

得到：

$$
\boxed{\widetilde s=[0.40,\ 0.30,\ 0.175]}
$$

在计算 KL 散度前，先求和 $0.40+0.30+0.175=0.875$，再做 L1 归一化：

$$
\widehat s^{\mathrm{LKV}}
=\left[\frac{0.40}{0.875},\frac{0.30}{0.875},\frac{0.175}{0.875}\right]
=\boxed{[0.4571,\ 0.3429,\ 0.2000]}
$$

而在实际淘汰阶段，论文 Algorithm 2 对预测重要性经过 MeanReduce、Pooling 后再做 Top-k。若为了演示省略 Pooling、预算为 2，则本例选择 KV1、KV2。

**我目前的理解**：Lookahead Tokens 的作用不是代替语言模型生成答案，而是代替「用答案 Query 判断哪些历史 KV 重要」这一统计过程。它们计算出的是**重要性分数**，不是直接输出最终回答内容。

### 3.3 / 这些软 Token 最后是否加入正式 KV Cache？

不会把它们当作正式回答的历史 Prompt KV 长期保存。其作用是在 Prefill 阶段帮助选择原始 Prompt 的 K/V；正式 Decode 使用经过筛选的 Prompt Cache，并使用模型原本的生成路径。

这也解释了 LookaheadKV 为什么能省下草稿方法的开销：**32 个软 Token 可以参与一次 Prefill 前向计算，但不需要执行 32 次顺序的自回归采样。** 当然，它们仍然增加了少量计算，并非完全没有成本。

## 04 / 第二项机制：Lookahead LoRA 为什么只对软 Token 激活？

我最初认为：已经给模型添加了可学习 Token，是否只训练这些 Token 的 Embedding 就够了？读到后面才发现，作者还希望增强这些软 Token 形成有效 Query 的能力，同时尽量不改变普通 Prompt Token 的原有处理过程。

论文给出了选择性 LoRA 的 Q/K 投影表达式：

$$
\boxed{
Q_{\mathrm{LKV}}=[X;P]W_q+[0;P]\Delta W_q
}
$$

$$
\boxed{
K_{\mathrm{LKV}}=[X;P]W_k+[0;P]\Delta W_k
}
$$

这里 $[0;P]$ 是**选择性门控的结构表示**：普通输入位置的 LoRA 分支为零，而追加的 Lookahead Tokens 使用新增 LoRA 投影。基础大模型权重 $W_q,W_k$ 保持冻结。

把 Q 投影按普通 Token 与 Lookahead Token 展开：

$$
Q_{\mathrm{LKV}}=
\begin{bmatrix}XW_q\\PW_q\end{bmatrix}
+
\begin{bmatrix}0\\P\Delta W_q\end{bmatrix}
=
\begin{bmatrix}XW_q\\P(W_q+\Delta W_q)\end{bmatrix}
$$

因此：

- **普通 Prompt Token**：$q_x=xW_q$；不执行 LoRA 更新分支。
- **Lookahead Token**：$q_p=p(W_q+\Delta W_q)$；除了基础投影还带有低秩更新。

在真实系统中，不仅 Q/K，论文主实验还对 V/O 投影及 FFN 的 up、down、gate 线性模块加入选择性 LoRA；原论文主实验中秩 $r=8$、LoRA 缩放参数 $\alpha=32$，额外可训练参数占模型比例低于 0.5%。这里的 $\alpha$ 是 LoRA 超参数，**不是 DuoAttention 的 Head 门控系数**。

### 4.1 / 完整手算：二维输入与秩 1 的 LoRA

为看清低秩分解，假设 Lookahead Token 向量为：

$$
p=[1,2],\qquad W=I_2
$$

基础投影：

$$
pW=[1,2]
$$

将 LoRA 更新写成两个低秩矩阵的乘积：

$$
A_{\mathrm{LoRA}}=\begin{bmatrix}1\\2\end{bmatrix},\qquad
B_{\mathrm{LoRA}}=\begin{bmatrix}0.2&-0.1\end{bmatrix}
$$

矩阵维度分别是 $2\times1$ 与 $1\times2$，秩最多为 1。此处为方便计算，**假设 LoRA 缩放因子为 1**：

$$
\Delta W=A_{\mathrm{LoRA}}B_{\mathrm{LoRA}}
=\begin{bmatrix}
0.2&-0.1\\
0.4&-0.2
\end{bmatrix}
$$

计算更新分支：

$$
\begin{aligned}
p\Delta W
&=[1,2]\begin{bmatrix}0.2&-0.1\\0.4&-0.2\end{bmatrix}\\
&=[1\times0.2+2\times0.4,\ 1\times(-0.1)+2\times(-0.2)]\\
&=\boxed{[1,-0.5]}.
\end{aligned}
$$

将基础投影与 LoRA 相加：

$$
\boxed{p(W+\Delta W)=[1,2]+[1,-0.5]=[2,1.5]}
$$

如果完全相同的数值来自普通 Prompt Token $x=[1,2]$，它不会启用 LoRA，因此仍有：

$$
q_x=xW=[1,2]
$$

**同样的输入数值，是否经过 LoRA 分支会改变输出。** 但这不意味着 Lookahead Token 已经知道哪个 KV 最重要；LoRA 改善预测方向必须依靠后续的训练监督。

### 4.2 / LoRA 为什么能改变 Attention 排名？

继续用二维教学例子，只考虑两个 Prompt Key：

$$
K_1=[1,0],\qquad K_2=[0,1],\qquad d_h=2
$$

不加 LoRA 时，$q=[1,2]$，两个 Logits 是：

$$
\left[\frac{qK_1^\top}{\sqrt2},\frac{qK_2^\top}{\sqrt2}\right]
=\left[\frac1{\sqrt2},\frac2{\sqrt2}\right]
$$

对**这两个 Key** 做 Softmax，得到约：

$$
A_{\mathrm{base}}=[0.330,\ 0.670]
$$

加上教学设定的 LoRA 后，$q'=[2,1.5]$：

$$
\left[\frac{q'K_1^\top}{\sqrt2},\frac{q'K_2^\top}{\sqrt2}\right]
=\left[\frac2{\sqrt2},\frac{1.5}{\sqrt2}\right]
$$

Softmax 变为：

$$
A_{\mathrm{LoRA}}=[0.587,\ 0.413]
$$

在这个教学设定下，Lookahead Query 从偏向 KV2 变成偏向 KV1。**这里只计算了两个历史 Key，忽略了真实长序列里的其他可关注位置**，因此这组概率不是论文模型的完整 Attention。

我现在的认识是：**Lookahead Embeddings 决定由哪些特殊向量发起预测；选择性 LoRA 改变这些向量经过模型投影后的表达能力；GT 和 KL 训练则决定它们应该朝哪个方向改变。**

## 05 / 训练监督：KL 散度怎么衡量预测得准不准？

GT 与 Lookahead 的重要性向量都要先做 L1 归一化，使其成为可比较的分布。论文针对每一层、每个 Attention Head 计算：

$$
D_{\mathrm{KL}}(p\|q)
=\sum_{i=1}^{n_{\mathrm{in}}}p_i\ln\frac{p_i}{q_i},\qquad
p=\widehat s^{\mathrm{GT}},\ q=\widehat s^{\mathrm{LKV}}
$$

- $p_i$：根据目标模型真实回答计算的 Prompt KV 重要性分布。
- $q_i$：Lookahead Tokens 预测的重要性分布。
- $\ln$：自然对数。
- $p_i\ln(p_i/q_i)$：位置 $i$ 对总 KL 的贡献，**单项可以为负，总 KL 非负**。

### 5.1 / 完整手算：三个 KV 的 KL 散度

下面换用另一组**独立的教学分布**，专门展示 KL 计算。它不是第 02 节、第 03 节两张 Attention 矩阵的直接计算结果：

$$
p=[0.375,\ 0.40625,\ 0.21875]
$$

$$
q=[0.375,\ 0.3125,\ 0.3125]
$$

两组数都已经归一化。逐项计算：

| KV | 真实 $p_i$ | 预测 $q_i$ | 预测偏差 |
| --- | ---: | ---: | --- |
| KV1 | 0.37500 | 0.37500 | 一致 |
| KV2 | 0.40625 | 0.31250 | 低估 |
| KV3 | 0.21875 | 0.31250 | 高估 |

第一项：

$$
D_1=0.375\ln\frac{0.375}{0.375}=0
$$

第二项：

$$
\begin{aligned}
D_2&=0.40625\ln\frac{0.40625}{0.3125}\\
&=0.40625\ln(1.3)\\
&\approx\boxed{0.10659}
\end{aligned}
$$

第三项：

$$
\begin{aligned}
D_3&=0.21875\ln\frac{0.21875}{0.3125}\\
&=0.21875\ln(0.7)\\
&\approx\boxed{-0.07802}
\end{aligned}
$$

因此，总 KL 散度为：

$$
\boxed{
D_{\mathrm{KL}}(p\|q)
=0+0.10659-0.07802\approx0.02856
}
$$

这里应避免一个误解：第三项为负并不意味着「预测较好」或总损失为负；因为 $q_3>p_3$，比值小于 1，单项对数自然为负。但整体 KL 散度依然非负，且仅当两个分布一致（在适用条件下）才等于 0。

### 5.2 / 预测错了以后，究竟更新谁？

本例的 GT 最重视 KV2，而预测却把 KV2 与 KV3 看得同样重要。所以，为减小损失，需要使预测逐步向「提高 KV2、降低 KV3」的方向调整。训练不会直接人工改动评分表，而是沿计算图反向传播：

```text
GT 回答 Y  --------->  GT 重要性 p（训练监督，停止更新）
Lookahead Tokens P --->  Q/K 与 Attention ---> 预测重要性 q
                                             |
                                         KL(p || q)
                                             |
                                           反向传播
                                             |
                            更新 P 和各选择性 LoRA 低秩参数
                            原始 LLM 权重保持冻结
```

**要特别注意**：LookaheadKV 的 $q$ 是从 Attention 权重平均、L1 归一化得到的，真实梯度会通过这些步骤和 Transformer 传播到 $P$ 与 LoRA 参数。不能把简化的 $q-p$ 直接当成论文 LoRA 权重的梯度。

### 5.3 / 为什么损失还要对所有层和 Head 求平均？

论文在模型的每层每个 Head 上都计算 GT 和预测分布：

$$
\boxed{
\mathcal L_{\mathrm{LKV}}
=\frac{1}{LH}\sum_{\ell=1}^{L}\sum_{h=1}^{H}
D_{\mathrm{KL}}\!\left(
\widehat s_{\ell,h}^{\mathrm{GT}}\,\big\|\,
\widehat s_{\ell,h}^{\mathrm{LKV}}
\right)
}
$$

这里 $L$ 是层数，$H$ 是用于求平均的 Head 数。其目标不是仅让某一层某一 Head 的预测变好，而是让 Lookahead Tokens 适应模型内部多种不同的关注模式。

例如只有两层、每层两个 Head，它们的 KL 值**假设**为 $0.02856,0.05,0.01,0.03144$，则：

$$
\mathcal L_{\mathrm{LKV}}
=\frac{0.02856+0.05+0.01+0.03144}{2\times2}
=\boxed{0.03}
$$

这与论文采用的层/Head 平均损失结构一致，但四个具体数值仍只是教学数据。

## 06 / 把 GT、Lookahead、LoRA 和 KL 连成一条完整训练路径

论文的训练思路可以按以下步骤复述：

1. **准备输入与参考回答**：对训练输入 $X$，使用目标模型预先生成真实回答 $Y$。
2. **GT Forward Pass**：在冻结的原始模型上，提取每层、每 Head 的「回答 Query → Prompt Key」Attention，并计算 GT 分数。
3. **Lookahead Forward Pass**：把软 Token $P$ 拼在 Prompt 后，选择性激活 LoRA，获得预测分数。
4. **分布归一化与损失**：分别将 GT、预测重要性做 L1 归一化，再计算所有层/Head 的平均 KL 散度。
5. **反向传播与更新**：优化 Lookahead Embeddings 与选择性 LoRA 参数；原始模型参数保持冻结。
6. **重复训练**：让未来预测分布在训练数据上更接近真实回答的重要性分布。

实际推理时则更短：

```text
Prompt X
   ↓
拼接已经训练好的 Lookahead Tokens P
   ↓
一次 Prefill 前向 + Lookahead LoRA
   ↓
提取 Lookahead Query → Prompt Key Attention
   ↓
MeanReduce → Pooling → Top-k 选出原始 Prompt KV
   ↓
移除不作为正式上下文保留的 Lookahead KV
   ↓
正式 Decode（不预先生成草稿答案）
```

这也是我认为本文与之前 DuoAttention 的训练思路最明显的联系与区别：DuoAttention 使用隐藏状态蒸馏和正则学习 Head 类型；LookaheadKV 使用 GT 重要性分布的 KL 散度学习**预测未来关注位置**的软 Token 与低秩适配器。两者都不需要训练全部原始 LLM 权重，但监督目标、可训练参数和部署路径并不相同。

## 07 / 今日理解变化与仍需辨析的细节

### 7.1 / 我原本的理解

最初容易把「未来重要性预测」理解成先把答案生成出来，然后再判断回答中哪些词对应历史 KV；或者把 Lookahead Tokens 当成已经生成的未来文字 Token。这样的理解忽略了该方法最重要的工程目标——避免自回归草稿带来的首 Token 延迟。

### 7.2 / 现在的理解

- **GT 不是在线可用的答案**：它是离线借助真实回答计算的训练监督；正式推理不能提前读取未来答案。
- **Lookahead Tokens 不是自然语言草稿**：它们是学习出来的软向量，输出对历史 Prompt KV 的重要性预测，而非回答内容。
- **选择性 LoRA 不是替换全部大模型权重**：只在软 Token 路径叠加低秩更新，原始权重冻结，普通 Prompt Token 的新增 LoRA 分支为零。
- **KL 对齐的对象是重要性分布**：不是最终生成词的概率分布，也不是 DuoAttention 的最终隐藏状态。
- **评分与部署要分开**：训练时计算真实回答 GT；部署时只有 Lookahead 预测分数参与 Top-k，不能偷偷使用 GT。

### 7.3 / 仍然值得继续追问的问题

**问题一：KL 较小，是否一定选对 Top-k？** 不一定。特别是两个 KV 的真实重要性十分接近时，预测分布的小偏差仍可能改变排序。值得进一步比较 KL、Top-k 重叠率和任务准确率之间的关系。

**问题二：不同层/Head 的预测目标会不会冲突？** 同一组 Lookahead Tokens 要服务于许多不同的注意力模式，具体如何利用不同层的表示能力值得继续阅读消融和实现细节。

**问题三：节省草稿生成就一定带来较低 TTFT 吗？** 不一定。仍需把 Lookahead Tokens 的额外前向、LoRA 分支、Attention 评分和 Top-k 的真实开销与草稿式方法比较，不能把「没有草稿」理解成「零开销」。

**问题四：训练依赖模型生成的 $Y$，换模型后能否直接通用？** 这是一个应当通过跨模型实验或代码验证的问题，不能仅凭当前笔记推断可迁移性。

## 08 / 下一次学习的最小行动

本次重点完成了**方法机制与数学手算的理解整理**，未记录独立代码实现、论文复现或实际性能测量。下次优先阅读论文第 6～9 页的实验部分以及附录 A/C：

1. 根据 Algorithm 1/2，画出训练路径与实际 Prefill 淘汰路径的对照图；
2. 核对 Table 3 的真实 TTFT 数值，区分**淘汰阶段额外耗时**与**完整首 Token 延迟**；
3. 阅读不同 Lookahead Token 数量、LoRA 启用位置的消融实验，判断性能与开销的拐点。

> 本文部分定义解释、矩阵例子与公式手算经过 AI 辅助整理和校验；所有构造数值均用于教学，不作为论文原始实验结果或个人已验证成果。

---

**论文定位**：LookaheadKV, ICLR 2026，主要参照原文 Section 2、Section 3.1、Section 3.2 和 Appendix A。  
**原论文**：[OpenReview PDF](https://openreview.net/pdf?id=RVLMGPXt2i)  
**项目代码**：[SamsungLabs/LookaheadKV](https://github.com/SamsungLabs/LookaheadKV)
