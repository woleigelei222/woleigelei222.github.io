---
title: "DuoAttention 精读（一）：从双路径门控到 Passkey 蒸馏——理解 Retrieval Head 的识别"
slug: "2026-10-26-duoattention-gating-passkey-distillation"
kind: "paper"
date: "2026-10-26"
summary: "记录 DuoAttention 的双路径 Attention 门控机制、合成 Passkey 训练数据、蒸馏损失与 L1 正则化，并梳理隐藏状态 H 和 FFN 在输出敏感性判断中的作用。"
tags: ["论文精读", "LLM 推理", "KV Cache", "DuoAttention", "Retrieval Head", "Streaming Attention", "知识蒸馏"]
featured: false
published: true
example: false
---

# DuoAttention 精读（一）：从双路径门控到 Passkey 蒸馏

> **学习日期：**2026 年 10 月 26 日  
> **阅读论文：** *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads*（ICLR 2025）  
> **本次学习范围：** 双 Attention 路径的输出融合、门控参数 $\alpha$、合成 Passkey 数据、蒸馏损失与 L1 正则化，以及隐藏状态 $H$ 与 FFN 的关系。  
> **记录说明：**本文整理的是当天的论文理解和讨论，不代表已复现训练代码或取得实验结果。

## 一、今天研究的核心问题

最近阅读 KV Cache 压缩相关论文时，一个反复出现的问题是：**哪些历史 KV 可以被压缩，而哪些必须保留？**

DuoAttention 给出的思路是区分两类 KV Head：

- **Retrieval Head：**如果限制其对中间历史 token 的访问，模型最终输出可能产生明显偏差，因此需要保留完整历史 KV。
- **Streaming Head：**如果只允许访问开头的 Attention Sink 和最近的 token，模型输出仍然相近，则可使用较小的缓存窗口。

我今天重点理解的不是这两个名称本身，而是**论文如何通过可训练门控以及输出偏差，把 Head 分类变成一个可优化的问题**。

## 二、两条 Attention 路径为什么要分别计算？

DuoAttention 在离线识别阶段，为同一组输入计算两种 Attention：

1. **Full Attention：**使用 Causal Mask，允许当前位置关注此前全部可见的历史 token。
2. **Streaming Attention：**使用 Sink + Recent 的掩码，仅允许访问起始位置及最近窗口。

因为两条路径能访问的 Key 集合不同，Softmax 的归一化分母也可能不同。因此，不能简单地在 Softmax 之前把两个分数矩阵按权重混合，再认为等价于论文的方法。

论文实际采用的是**先分别完成 Softmax 和 Value 加权求和，再融合两条 Attention 的输出向量**：

$$
A_{\mathrm{mixed}}=\alpha A_{\mathrm{full}}+(1-\alpha)A_{\mathrm{stream}},\qquad 0\leq\alpha\leq1.
$$

这里 $\alpha$ 是每层、每个 KV Head 对应的可学习门控值：

| 门控值 | 两条路径的组合 | 直观含义 |
|---|---|---|
| $\alpha=1$ | 100% Full | 完全依赖完整历史路径 |
| $\alpha=0.5$ | 两条路径各占一半 | 训练中的连续过渡状态 |
| $\alpha=0$ | 100% Streaming | 完全使用 Sink + Recent 路径 |

> **注意：**这里的 $\alpha$ 是离线优化得到的门控参数，不是每个请求运行时动态预测的一次性重要性分数。在 GQA 结构中，缓存管理还需要考虑一个 KV Head 被多个 Query Heads 共享的事实。

### 2.1 一个能手算的混合例子

假设已经分别得到两个 Attention **输出向量**：

$$
A_{\mathrm{full}}=[2,4],\qquad A_{\mathrm{stream}}=[1,3].
$$

当 $\alpha=0.8$ 时：

$$
\begin{aligned}
A_{\mathrm{mixed}}
&=0.8[2,4]+0.2[1,3]\\
&=[1.6+0.2,\ 3.2+0.6]\\
&=\boxed{[1.8,3.8]}.
\end{aligned}
$$

这个例子帮助我区分了两个位置：**Softmax 用来计算每条路径各自的注意力权重；$\alpha$ 用来混合两条路径的输出。**

## 三、为什么不能只优化输出一致性？

作者希望混合模型的输出尽量接近完整 Full Attention 模型，这对应蒸馏损失 $L_{\mathrm{distill}}$。

但如果只要求输出一致，那么一开始所有 Head 的门控值都为 1 时，混合模型与完整模型基本一致，蒸馏损失接近 0。这样虽然准确，却完全没有实现 KV Cache 压缩。

因此论文还引入 **L1 正则化**：

$$
L_{\mathrm{reg}}=\sum_i\sum_j|\alpha_{i,j}|.
$$

最后共同优化：

$$
\boxed{L=L_{\mathrm{distill}}+\lambda L_{\mathrm{reg}}}.
$$

论文实验采用 $\lambda=0.05$。两部分的目标有所不同：

- **蒸馏损失**：要求模型尽量保留原来的长距离检索能力，避免关键 Head 被过度压缩。
- **L1 正则化**：鼓励门控值下降，让尽可能多的 Head 使用较低成本的 Streaming 路径。

我用一个直观例子来理解：假设 Head A 改为 Streaming 后，模型隐藏状态几乎不变，那么降低 $\alpha_A$ 的正则化收益可能大于蒸馏损失；而 Head B 如果改为 Streaming 就导致明显的输出偏差，优化器便更倾向于保留较高的 $\alpha_B$。

这不是对两个 Head 分别设置规则，而是**所有门控参数通过同一个总体损失联合优化**。

## 四、为什么作者专门设计合成 Passkey 任务？

我原本对门控训练还有一个疑问：为什么不直接使用普通自然语言文本？

现在的理解是：**文本很长，不代表下一 token 的预测一定需要访问很远的历史。**普通语言建模中，很多 token 可以依靠局部上下文预测。因此，真正负责远距离检索的 Head 可能没有得到充分检验。

为此，DuoAttention 在长篇背景文本的随机位置插入 **10 组随机生成的 Passkey**，每组包含 **32 个词**，然后在文本末尾要求模型回忆这些 Passkey。由于密码内容本身是随机的，仅凭最近上下文很难可靠推断答案。

### 4.1 简化情景

以下是帮助理解的教学示例，不是论文实际输入：

- 在 token 300 附近埋入“密码 A7-K2”；
- 在 token 12000 附近埋入“密码 B9-Q4”；
- 在文本末尾询问此前出现的两个密码。

如果某个 Head 转为 Streaming 后无法再访问对应的中间历史位置，那么混合模型的答案位置表示就可能偏离 Full Attention 模型。这种数据更容易暴露 Head 对**长距离信息检索**的实际作用。

**Passkey 数据的主要用途是识别 Head，而不是重新训练 LLM 学会某个密码。**论文冻结原模型参数，重点优化门控值。

## 五、蒸馏损失：为什么要比较最后隐藏状态 $H$？

作者并不直接将“最终密码答对/答错”作为主要可微损失，而是比较两条模型路径在 **Passkey 答案 token 位置**的最后隐藏状态。

论文公式（1）可以写为：

$$
\boxed{
L_{\mathrm{distill}}
=\frac{1}{N}\sum_{n=1}^{N}
\sum_{t=T-l+1}^{T}
\left\|H_{\mathrm{full}}^{(n)}[t]-H_{\mathrm{mixed}}^{(n)}[t]\right\|_2^2
}.
$$

| 符号 | 理解 |
|---|---|
| $N$ | 样本数量 |
| $T$ | 序列总长度 |
| $l$ | 序列末尾参与监督的 Passkey 答案 token 数 |
| $H_{\mathrm{full}}$ | 完整 Full Attention 模型的最后隐藏状态 |
| $H_{\mathrm{mixed}}$ | 使用门控混合 Attention 后的最后隐藏状态 |
| $\|\cdot\|_2^2$ | 两个向量对应元素之差的平方和 |

我今天理解的关键细节有两个：

**第一，监督重点放在答案位置。**如果对长篇故事中所有普通 token 一视同仁地计算误差，局部语言建模信号可能淹没长距离检索信号。

**第二，这不是对每个 Head 分别计算独立的蒸馏误差。**多个 Head 的影响还会经过输出投影及后续 Transformer 层耦合，最后由模型级隐藏状态的差异共同体现。

### 5.1 一次蒸馏损失手算

假设只有一个答案 token，教师隐藏状态为：

$$H_{\mathrm{full}}=[2,4].$$

一种混合配置得到：

$$H_{\mathrm{mixed}}=[1,1].$$

那么误差是：

$$L_{\mathrm{distill}}=(2-1)^2+(4-1)^2=1+9=\boxed{10}. $$

另一种配置得到：

$$H_{\mathrm{mixed}}=[1.9,3.9],$$

其误差为：

$$L_{\mathrm{distill}}=(2-1.9)^2+(4-3.9)^2=\boxed{0.02}. $$

第二个配置显然更接近原始模型。**这里的向量和误差是教学假设，不是对论文模型运行得到的实验数值。**

## 六、补充理解：隐藏状态 $H$ 与 FFN 是什么关系？

这部分是我在理解蒸馏损失时进一步追问的问题。

过去我容易把 Attention 输出和最终隐藏状态混为一谈。现在逐渐明确：

- **Attention 输出**负责聚合其他 token 的信息；
- **输出投影、残差连接、FFN 与后续层**会进一步变换这些信息；
- **$H$** 是模型前向传播到最后时得到的隐藏表示，因此不等于某个 Head 的 Attention 权重或输出。

对于常见的 Pre-Norm Transformer，可以用下面的简化结构理解：

$$
U^{(l)}=H^{(l)}+\operatorname{Attention}\!\big(\operatorname{Norm}(H^{(l)})\big),
$$

$$
H^{(l+1)}=U^{(l)}+\operatorname{FFN}\!\big(\operatorname{Norm}(U^{(l)})\big).
$$

这里为了说明数据流，把多头输出投影等细节包含在 Attention 函数内部。真实 Llama 类模型通常采用 RMSNorm 和 SwiGLU FFN。

因此，当改变某个 Head 的 $\alpha$ 时，首先改变的是混合 Attention 输出，而这个变化可能经过后续层被放大、减弱或与其他 Head 的信息相互作用，最终才体现在：

$$H_{\mathrm{full}}-H_{\mathrm{mixed}}.$$

**FFN 不是凭空给定一个输出值**；它根据输入隐藏向量和冻结的模型权重执行线性变换及非线性激活。在教学手算中直接假设 FFN 输出只能用于解释流程，不能当作真实模型复现。

## 七、今天最重要的理解变化

### 7.1 我原本以为

初步理解 DuoAttention 时，容易认为只要观察一个 Head 对远距离 token 的 Attention 权重大小，就能决定它应不应该保留全部 KV。

### 7.2 现在的理解

**Head 的重要性不能只从局部 Attention 权重判断，还要考虑限制该 Head 后，对模型最终表示造成的影响。**DuoAttention 通过 Passkey 数据制造明确的长距离依赖，再利用蒸馏损失与 L1 正则项联合学习门控值。

### 7.3 为什么理解发生了变化

在具体推导双路径混合、Passkey 监督以及隐藏状态 $H$ 的计算路径后，我认识到：模型输出不只是 Attention 权重的结果，还依赖 Value、输出投影、残差、FFN 和后续层。用最终隐藏状态比较两套 Attention 策略，能间接反映这些作用的综合结果。

## 八、目前仍值得继续追问的问题

以下是基于本次学习提出的**待研究问题**，不是论文已经证明的结论：

1. 使用合成 Passkey 训练出的静态 Head 分类，遇到不同任务或输入分布时是否仍然稳定？
2. 如果有些 Head 的重要性随着生成阶段变化，固定 $\alpha$ 分类是否会限制进一步压缩？
3. DuoAttention 的输出敏感性能否与 HARD-KV 的 Head-wise 动态预算、ThinKV 的不同思考类型精度分配结合？这种组合是否值得付出额外训练与索引管理开销？

## 九、下一步学习计划

接下来继续阅读 DuoAttention 的 **Head 二值化与部署流程**，重点理解：如何根据目标比例确定阈值 $\tau$；为何需要重排 Q/K/V 的 Head 通道；以及 Retrieval Head 全量缓存、Streaming Head 固定窗口缓存如何分别作用于 Decode 与 Chunked Prefill。

这一步仍属于计划，尚未作为本次学习成果记录。

---

> **本日总结：**DuoAttention 的关键不只是划分 Retrieval / Streaming Head，而是通过“合成远距离检索任务 → 输出级蒸馏约束 → L1 压缩压力 → 学习门控参数”的链条，把 Head 缓存选择转化为一个可优化的问题。
