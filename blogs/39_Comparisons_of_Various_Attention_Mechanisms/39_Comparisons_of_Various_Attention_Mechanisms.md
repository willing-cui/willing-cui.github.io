> 本文介绍了各类注意力机制的数学表示，按核心机制、多头变体、稀疏/局部机制、视觉机制和跨模态机制分类。

## 1. 核心注意力机制

### 1.1 缩放点积注意力 (SDPA)

这是 Transformer 中最基础的注意力计算单元，其公式为：

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_k}}\\right)V
$$

其中：
- $Q \\in \\mathbb{R}^{N \\times d\_k}$：查询矩阵
- $K \\in \\mathbb{R}^{M \\times d\_k}$：键矩阵
- $V \\in \\mathbb{R}^{M \\times d\_v}$：值矩阵
- $d\_k$：键向量的维度，用于缩放点积结果。

### 1.2 双向注意力 (Bidirectional Attention)

双向注意力是自注意力的一种，其核心特点是序列中的每个位置都可以关注到所有其他位置（包括过去和未来的信息）。它通过一个全1的注意力掩码实现，不限制信息流方向。

其数学表示与 SDPA 相同，但掩码 $M$ 为全1矩阵：

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_k}} + M\\right)V
$$

其中 $M\_{ij} = 0$（无掩码效应），确保每个位置都能关注到所有位置。

### 1.3 单向/因果注意力 (Causal Attention)

因果注意力通过下三角掩码确保每个位置只能关注到它之前的位置，无法"看到"未来信息。其数学表示为：

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_k}} + M\_{\\text{causal}}\\right)V
$$

其中因果掩码 $M\_{\\text{causal}}$ 定义为：

$$
M\_{\\text{causal},ij} = 
\\begin{cases} 
0, & \\text{if } j \\le i \\\\
-\\infty, & \\text{if } j > i 
\\end{cases}
$$

这使得 softmax 后未来位置的权重为0，实现自回归特性。

### 1.4 交叉注意力 (Cross-Attention)

交叉注意力用于处理两个不同序列，其 $Q$ 来自一个序列，而 $K$ 和 $V$ 来自另一个序列。数学表示为：

$$
\\text{CrossAttention}(Q\_A, K\_B, V\_B) = \\text{softmax}\\left(\\frac{Q\_A K\_B^\\top}{\\sqrt{d\_k}}\\right)V\_B
$$

其中：
- $Q\_A \\in \\mathbb{R}^{N\_A \\times d\_k}$：来自序列 A 的查询
- $K\_B, V\_B \\in \\mathbb{R}^{N\_B \\times d\_k}$：来自序列 B 的键和值。

## 2. 多头注意力及其变体

### 2.1 多头注意力 (MHA)

MHA 将输入投影到多个子空间，并行计算多个 SDPA。设头数为 $h$，每个头的维度为 $d\_h = d/h$，则第 $i$ 个头的计算为：

$$
\\begin{aligned}
Q\_i &= X\_q W\_i^Q \\in \\mathbb{R}^{T\_t \\times d\_h} \\\\
K\_i &= X\_k W\_i^K \\in \\mathbb{R}^{T\_s \\times d\_h} \\\\
V\_i &= X\_v W\_i^V \\in \\mathbb{R}^{T\_s \\times d\_h} \\\\
\\text{head}\_i &= \\text{Attention}(Q\_i, K\_i, V\_i) \\in \\mathbb{R}^{T\_t \\times d\_h}
\\end{aligned}
$$

多头注意力的最终输出为：

$$
\\text{MHA}(X\_q, X\_k, X\_v) = \\text{Concat}(\\text{head}\_1, \\ldots, \\text{head}\_h) W^O
$$

其中 $W^O \\in \\mathbb{R}^{d \\times d}$ 为输出投影矩阵。

### 2.2 多查询注意力 (MQA)

MQA 让所有注意力头共享同一组 $K$ 和 $V$ 矩阵，仅 $Q$ 独立。设头数为 $h$，则：

$$
\\begin{aligned}
Q\_i &= X\_q W\_i^Q \\in \\mathbb{R}^{N \\times d/h} \\\\
K &= X\_k W^K \\in \\mathbb{R}^{N \\times d} \\\\
V &= X\_v W^V \\in \\mathbb{R}^{N \\times d} \\\\
\\text{head}\_i &= \\text{Attention}(Q\_i, K, V) \\in \\mathbb{R}^{N \\times d} \\\\
\\text{MQA}(X\_q, X\_k, X\_v) &= \\text{Concat}(\\text{head}\_1, \\ldots, \\text{head}\_h) W^O
\\end{aligned}
$$

这大幅减少了 KV 缓存，但可能牺牲部分模型质量。

### 2.3 分组查询注意力 (GQA)

GQA 是 MHA 和 MQA 的折中方案，将头分为 $g$ 组，组内共享 $K$ 和 $V$。设总头数为 $h$，每组头数为 $h/g$，则第 $s$ 组（$s=1,\\ldots,g$）的计算为：

$$
\\begin{aligned}
Q\_i^{(s)} &= X\_q W\_i^{Q,(s)} \\in \\mathbb{R}^{N \\times d\_h} \\\\
K^{(s)} &= X\_k W^{K,(s)} \\in \\mathbb{R}^{N \\times d} \\\\
V^{(s)} &= X\_v W^{V,(s)} \\in \\mathbb{R}^{N \\times d} \\\\
\\text{head}\_i^{(s)} &= \\text{Attention}(Q\_i^{(s)}, K^{(s)}, V^{(s)}) \\in \\mathbb{R}^{N \\times d}
\\end{aligned}
$$

最终输出为所有头的拼接：

$$
\\text{GQA}(X\_q, X\_k, X\_v) = \\text{Concat}(\\text{head}\_1^{(1)}, \\ldots, \\text{head}\_{h/g}^{(g)}) W^O
$$

当 $g = h$ 时，GQA 退化为 MHA；当 $g = 1$ 时，退化为 MQA。

## 3. 稀疏/局部注意力

稀疏/局部注意力通过限制注意力计算的范围，将标准注意力的 $O(L^2)$ 复杂度降低到 $O(L \\cdot K)$ 或 $O(L \\log L)$，其中 $K$ 是每个位置关注的邻居数量。

其数学表示为：

$$
\\text{SparseAttention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_k}} \\odot M\_{\\text{sparse}}\\right)V
$$

其中 $M\_{\\text{sparse}}$ 是一个稀疏掩码矩阵，$M\_{ij} = 1$ 仅当位置 $j$ 在位置 $i$ 的允许关注范围内，否则为 $-\\infty$。

常见的稀疏模式包括：
- **局部注意力**：每个位置只关注相邻的 $K$ 个位置
- **滑动窗口注意力**：类似局部注意力，但窗口在序列上滑动
- **带状注意力**：注意力限制在对角带状区域内
- **全局+局部混合**：某些特殊位置（如 [CLS]）关注整个序列，其他位置关注局部窗口

## 4. 视觉注意力机制

### 4.1 通道注意力 (Channel Attention)

通道注意力通过为每个通道分配不同的权重，强调对任务最有贡献的通道。以 SE 模块为例：

$$
\\begin{aligned}
z\_c &= \\frac{1}{H \\times W} \\sum\_{i=1}^H \\sum\_{j=1}^W x\_c(i,j) \\\\
s &= \\sigma(W\_2 \\delta(W\_1 z)) \\\\
F\_{\\text{channel}} &= s \\odot x
\\end{aligned}
$$

其中：
- $x \\in \\mathbb{R}^{H \\times W \\times C}$：输入特征
- $z \\in \\mathbb{R}^C$：全局平均池化后的通道描述符
- $W\_1 \\in \\mathbb{R}^{C/r \\times C}$, $W\_2 \\in \\mathbb{R}^{C \\times C/r}$：全连接层权重
- $\\delta$：ReLU 激活函数
- $\\sigma$：sigmoid 激活函数
- $\\odot$：逐通道乘法

### 4.2 空间注意力 (Spatial Attention)

空间注意力通过对特征图中的特定空间位置进行加权，突出对任务最有贡献的区域。其数学表示为：

$$
\\begin{aligned}
z\_{\\text{max}} &= \\max\_{c} x(i,j,c) \\\\
z\_{\\text{avg}} &= \\frac{1}{C} \\sum\_{c=1}^C x(i,j,c) \\\\
z &= \\text{concat}(z\_{\\text{max}}, z\_{\\text{avg}}) \\\\
m &= \\sigma(f^{7 \\times 7}(z)) \\\\
F\_{\\text{spatial}} &= m \\odot x
\\end{aligned}
$$

其中 $f^{7 \\times 7}$ 表示 $7 \\times 7$ 卷积操作。

## 5. 跨模态注意力

跨模态注意力用于融合不同模态的信息，其 $Q$ 来自模态 A，$K$ 和 $V$ 来自模态 B。数学表示为：

$$
\\text{CrossModalAttention}(Q\_A, K\_B, V\_B) = \\text{softmax}\\left(\\frac{Q\_A K\_B^\\top}{\\sqrt{d\_k}}\\right)V\_B
$$

其中：
- $Q\_A \\in \\mathbb{R}^{N\_A \\times d\_k}$：来自模态 A（如文本）的查询
- $K\_B, V\_B \\in \\mathbb{R}^{N\_B \\times d\_k}$：来自模态 B（如图像）的键和值

注意力矩阵 $A \\in \\mathbb{R}^{N\_A \\times N\_B}$ 的每个元素 $A\_{ij}$ 表示模态 A 的第 $i$ 个位置与模态 B 的第 $j$ 个位置的相关性。

## 6. 其他注意力变体

### 6.1 加性注意力 (Additive Attention)

加性注意力通过非线性变换计算相似度，适用于 $Q$ 和 $K$ 维度不同的情况：

$$
\\text{score}(q, k) = v^\\top \\tanh(W\_q q + W\_k k)
$$

其中 $W\_q, W\_k$ 为可学习权重矩阵，$v$ 为可学习向量。

### 6.2 点积注意力 (Dot-Product Attention)

点积注意力是 SDPA 的前身，不加缩放因子：

$$
\\text{Attention}(Q, K, V) = \\text{softmax}(QK^\\top)V
$$

当 $d\_k$ 较大时，点积结果方差增大，可能导致 softmax 梯度消失，因此 Transformer 采用缩放点积注意力。

### 6.3 线性注意力 (Linear Attention)

线性注意力通过核函数近似 softmax，将计算复杂度从 $O(L^2)$ 降低到 $O(L)$：

$$
\\text{LinearAttention}(Q, K, V) = \\phi(Q) \\left(\\phi(K)^\\top V\\right)
$$

其中 $\\phi$ 为适当的特征映射函数。

## 7. 注意力机制详细对比表

下表从数学形式、核心特点、计算复杂度、典型应用等维度对各类注意力机制进行了系统对比。

| 机制类型                  | 数学表示                                                     | 核心特点                             | 计算复杂度                  | 典型应用                          |
| :------------------------ | :----------------------------------------------------------- | :----------------------------------- | :-------------------------- | :-------------------------------- |
| **缩放点积注意力 (SDPA)** | $\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_k}}\\right)V$ | 基础计算单元，缩放点积+softmax       | $O(L^2d)$                   | Transformer 所有注意力层的基础    |
| **双向注意力**            | $\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_k}} + M\\right)V$, $M\_{ij} = 0$ | 全局上下文，无掩码，可关注所有位置   | $O(L^2d)$                   | BERT 等编码器，文本理解任务       |
| **因果注意力**            | $\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_k}} + M\_{\\text{causal}}\\right)V$, $M\_{\\text{causal},ij} = \\begin{cases} 0, & j \\le i \\\\ -\\infty, & j > i \\end{cases}$ | 自回归，下三角掩码，只能关注过去位置 | $O(L^2d)$                   | GPT、LLaMA 等自回归语言模型       |
| **交叉注意力**            | $\\text{CrossAttention}(Q\_A, K\_B, V\_B) = \\text{softmax}\\left(\\frac{Q\_A K\_B^\\top}{\\sqrt{d\_k}}\\right)V\_B$ | 跨序列交互，$Q$ 与 $K/V$ 来源不同    | $O(L\_q L\_k d)$              | 机器翻译、摘要等序列转换任务      |
| **多头注意力 (MHA)**      | $\\text{MHA}(X\_q, X\_k, X\_v) = \\text{Concat}(\\text{head}\_1, \\ldots, \\text{head}\_h) W^O$, $\\text{head}\_i = \\text{Attention}(Q\_i, K\_i, V\_i)$ | 并行多子空间，表达力强               | $O(L^2d)$                   | 原始 Transformer，高质量生成任务  |
| **多查询注意力 (MQA)**    | $\\text{MQA}(X\_q, X\_k, X\_v) = \\text{Concat}(\\text{head}\_1, \\ldots, \\text{head}\_h) W^O$, $\\text{head}\_i = \\text{Attention}(Q\_i, K, V)$ | 共享 $K/V$，推理高效，KV 缓存减少    | $O(L^2d)$                   | ChatGLM2、Gemini 等，资源受限场景 |
| **分组查询注意力 (GQA)**  | $\\text{GQA}(X\_q, X\_k, X\_v) = \\text{Concat}(\\text{head}\_1^{(1)}, \\ldots, \\text{head}\_{h/g}^{(g)}) W^O$, $\\text{head}\_i^{(s)} = \\text{Attention}(Q\_i^{(s)}, K^{(s)}, V^{(s)})$ | 组内共享 $K/V$，平衡效率与质量       | $O(L^2d)$                   | LLaMA-2、Mistral 等主流开源模型   |
| **稀疏/局部注意力**       | $\\text{SparseAttention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_k}} \\odot M\_{\\text{sparse}}\\right)V$ | 降低计算复杂度，处理长序列           | $O(LKd)$ 或 $O(L \\log L d)$ | 长文档理解、基因组分析等          |
| **通道注意力**            | $F\_{\\text{channel}} = s \\odot x$, $s = \\sigma(W\_2 \\delta(W\_1 z))$ | 聚焦特征通道，强调重要通道           | $O(HWC)$                    | 图像分类、目标检测等视觉任务      |
| **空间注意力**            | $F\_{\\text{spatial}} = m \\odot x$, $m = \\sigma(f^{7 \\times 7}(z))$ | 聚焦空间位置，突出重要区域           | $O(HWC)$                    | 图像分类、目标检测等视觉任务      |
| **跨模态注意力**          | $\\text{CrossModalAttention}(Q\_A, K\_B, V\_B) = \\text{softmax}\\left(\\frac{Q\_A K\_B^\\top}{\\sqrt{d\_k}}\\right)V\_B$ | 融合不同模态信息                     | $O(L\_A L\_B d)$              | 视觉问答、图像描述等多模态任务    |
| **加性注意力**            | $\\text{score}(q, k) = v^\\top \\tanh(W\_q q + W\_k k)$           | 非线性相似度计算，鲁棒性强           | $O(Ld^2)$                   | 早期机器翻译模型                  |
| **点积注意力**            | $\\text{Attention}(Q, K, V) = \\text{softmax}(QK^\\top)V$       | 不加缩放，计算简单                   | $O(L^2d)$                   | 早期注意力模型                    |
| **线性注意力**            | $\\text{LinearAttention}(Q, K, V) = \\phi(Q) \\left(\\phi(K)^\\top V\\right)$ | 核函数近似，降低复杂度               | $O(Ld)$                     | 超长序列处理                      |

## 总结与选择

- **SDPA** 是注意力计算的"原子操作"，所有其他机制都基于它构建。
- **双向注意力** 与 **因果注意力** 是信息流方向上的两种基本模式，分别服务于理解（NLU）和生成（LM）任务。
- **MHA、MQA、GQA** 是在多头设计上的不同权衡，分别侧重模型能力、推理效率和两者的平衡。

在实际应用中，选择哪种注意力机制取决于具体任务、序列长度、硬件资源以及对模型质量和推理速度的权衡。