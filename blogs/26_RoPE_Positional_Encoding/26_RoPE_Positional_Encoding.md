RoPE（Rotary Position Embedding，旋转位置编码）是近年来在Transformer架构中广泛使用的一种位置编码方法，由苏剑林等人提出。它的核心思想是通过**旋转矩阵**对token的嵌入向量进行旋转操作，从而将位置信息编码到注意力机制中。RoPE因其良好的**外推性**、**可扩展性**和**理论优雅性**，被广泛应用于LLaMA、ChatGLM、PaLM等主流大模型中。

<span class="image main">
<img class="main img-in-blog" style="max-width: 40%" src="./blogs/26_RoPE_Positional_Encoding/SuperMacro_Rope.webp" alt="Rope" />
<i>Rope, <br> By No machine-readable author provided. <a href="//commons.wikimedia.org/wiki/User:HiveHarbingerCOM" title="User:HiveHarbingerCOM">HiveHarbingerCOM</a> assumed (based on copyright claims). - No machine-readable source provided. Own work assumed (based on copyright claims)., <a href="https://creativecommons.org/licenses/by/3.0" title="Creative Commons Attribution 3.0">CC BY 3.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=3661818">Link</a></i>
</span>


## 一、RoPE 的核心思想

### 1. 基本直觉
在自注意力机制中，我们需要计算查询向量（Query）和键向量（Key）的内积，以衡量它们之间的相关性。RoPE的目标是：**让这个内积结果包含两个token之间的相对位置信息**。

具体来说，对于位置为 $m$ 的token，其查询向量 $q_m$ 和位置为 $n$ 的token的键向量 $k_n$，经过RoPE编码后，它们的内积应仅依赖于相对位置 $m-n$，即：
$$
\langle \text{RoPE}(q_m, m), \text{RoPE}(k_n, n) \rangle = g(q_m, k_n, m-n)
$$
这意味着，模型在计算注意力时，能自动感知到两个token之间的相对距离，而不是绝对位置。

### 2. 数学形式：复数空间中的旋转
RoPE将token的嵌入向量视为复数向量。对于向量的每一维，将其视为一个复数（相邻两维为一组，分别作为实部和虚部），然后通过旋转操作引入位置信息。

假设我们有一组二维向量 $(x, y)$，可以将其表示为复数 $z = x + iy$。如果将其旋转角度 $\theta$，新复数为：
$$
z' = z \cdot e^{i\theta} = (x \cos\theta - y \sin\theta) + i(x \sin\theta + y \cos\theta)
$$
写成矩阵形式：
$$
\begin{bmatrix} x' \\\\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\\\ y' \end{bmatrix}
$$
这个矩阵就是**二维旋转矩阵**。

RoPE将这一思想扩展到高维向量：将向量的每一对相邻维度视为一个二维平面，并对每个平面应用不同频率的旋转。旋转角度由token的位置和该平面的频率共同决定。


## 二、RoPE 的数学推导

### 1. 位置编码的通用形式
假设我们有一个 $d$ 维的查询向量 $q$ 和键向量 $k$，我们希望找到函数 $f(q, m)$ 和 $f(k, n)$，使得：
$$
\langle f(q, m), f(k, n) \rangle = g(q, k, m-n)
$$
RoPE的解决方案是：**对 $q$ 和 $k$ 的每个维度对应用旋转，旋转角度与位置成线性关系**。

### 2. 具体公式
将 $d$ 维向量分为 $d/2$ 组二维向量。对于第 $i$ 组（$i=0,1,\dots,d/2-1$），定义其旋转角频率为：
$$
\theta_i = 10000^{-2i/d}
$$
（注：这里的10000是基础频率，可调整。）

对于位置为 $m$ 的token，其查询向量 $q$ 的第 $i$ 组二维向量 $(q_{2i}, q_{2i+1})$ 经过RoPE编码后变为：
$$
\begin{bmatrix} q_{2i} \cos(m\theta_i) - q_{2i+1} \sin(m\theta_i) \\\\ q_{2i} \sin(m\theta_i) + q_{2i+1} \cos(m\theta_i) \end{bmatrix}
$$
同理，对键向量 $k$ 的第 $i$ 组应用相同的旋转（但使用位置 $n$）。

### 3. 矩阵表示
整个RoPE操作可以写成一个分块对角矩阵 $R_m$（每个块是一个二维旋转矩阵）：
$$
R_m = \text{diag}\left( R_m^{\theta_0}, R_m^{\theta_1}, \dots, R_m^{\theta_{d/2-1}} \right)
$$
其中每个块：
$$
R_m^{\theta_i} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\\\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
$$
则RoPE编码后的查询和键向量为：
$$
q_m' = R_m q, \quad k_n' = R_n k
$$

### 4. 内积的性质
计算内积：
$$
\langle q_m', k_n' \rangle = (R_m q)^T (R_n k) = q^T R_m^T R_n k
$$
由于旋转矩阵是正交矩阵（$R_m^T R_m = I$），且 $R_m^T R_n = R_{n-m}$（旋转矩阵的复合性质），因此：
$$
\langle q_m', k_n' \rangle = q^T R_{n-m} k
$$
这正是我们想要的性质：内积只依赖于相对位置 $n-m$。

### 5. 一个简化的例子

为了便于理解，我们进行以下简化：

1.  **维度简化**：假设我们的词向量只有 **2 个维度**（实际中可能是 768 或 1024 等）。
2.  **词向量**：假设我们有一个单词的词向量是 $[1, 0]$（一个二维向量）。

#### 步骤 1: 将二维向量视为复数

RoPE 将每两个维度视为一个**复数**。
*   我们的向量 $[x, y]$ 可以看作复数 $z = x + iy$。
*   在我们的例子中，$[1, 0]$ 就是复数 $1 + 0i$。

#### 步骤 2: 定义旋转

对于位置为 $m$ 的词，RoPE 会将其对应的复数向量旋转 $m \theta$ 的角度。
*   旋转一个复数 $z$ 角度 $\theta$ 的公式是：$z' = z \cdot e^{i\theta}$。

在二维实数空间中，这个旋转操作可以写成矩阵乘法：

$$
\begin{bmatrix} x' \\\\ y' \end{bmatrix} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\\\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x \\\\ y \end{bmatrix}
$$

这个矩阵就是**旋转矩阵**。

#### 步骤 3: 具体数值演算

假设我们选择一个基础的旋转角度 $\theta = 30^\circ$（即 $\pi/6$ 弧度，实际中 $\theta$ 是预设的频率参数）。

**情况 A: 位置 $m=0$**
*   旋转角度 = $0 \times 30^\circ = 0^\circ$。
*   旋转矩阵是单位矩阵。
*   编码后的向量 = $[1, 0]$（不变）。

**情况 B: 位置 $m=1$**
*   旋转角度 = $1 \times 30^\circ = 30^\circ$。
*   $\cos(30^\circ) \approx 0.866$, $\sin(30^\circ) = 0.5$。
*   计算：
    $$
    \begin{aligned}
    x' &= 1 \times 0.866 - 0 \times 0.5 = 0.866 \\\\
    y' &= 1 \times 0.5 + 0 \times 0.866 = 0.5
    \end{aligned}
    $$
*   编码后的向量 $\approx [0.866, 0.5]$。

**情况 C: 位置 $m=2$**
*   旋转角度 = $2 \times 30^\circ = 60^\circ$。
*   $\cos(60^\circ) = 0.5$, $\sin(60^\circ) \approx 0.866$。
*   计算：
    $$
    \begin{aligned}
    x' &= 1 \times 0.5 - 0 \times 0.866 = 0.5 \\\\
    y' &= 1 \times 0.866 + 0 \times 0.5 = 0.866
    \end{aligned}
    $$
*   编码后的向量 $\approx [0.5, 0.866]$。

**关键点**：位置编码不再是一个简单的“加法”，而是对原始向量进行了一个**旋转**。位置越高，旋转的角度越大。

### 6. RoPE 与三角恒等式的关系

**RoPE与积化和差公式有非常紧密的关系。** 可以说，积化和差公式是RoPE能够实现“相对位置编码”的**数学核心**。

#### 6.1 数学推导的核心

RoPE 的核心目标是证明：
$$
\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m-n)
$$

在复数域中，这个变换的最优解是**旋转**：
$$
f(q, m) = q \cdot e^{im\theta}
$$

#### 6.2 复数视角的推导

计算复数形式的“内积”（实际上是埃尔米特内积）:
$$
f(q, m) \cdot \overline{f(k, n)} = (q e^{im\theta}) \cdot \overline{(k e^{in\theta})} = (q \bar{k}) \cdot e^{i(m-n)\theta}
$$

将 $q$ 和 $k$ 写成它们的实部和虚部（即向量的两个维度）：
*   设 $q = q_1 + i q_2$
*   设 $k = k_1 + i k_2$
*   那么 $q \bar{k} = (q_1 k_1 + q_2 k_2) + i (q_2 k_1 - q_1 k_2)$
*   而 $e^{i(m-n)\theta} = \cos((m-n)\theta) + i \sin((m-n)\theta)$

将这两项相乘，我们得到一个新的复数，它的**实部**（也就是我们最终需要的实数内积）为：
$$
\text{实部} = (q_1 k_1 + q_2 k_2) \cos((m-n)\theta) + (q_2 k_1 - q_1 k_2) \sin((m-n)\theta)
$$

**这正是实数向量 ($q$)与旋转后的 ($k$)做标准点积的结果 -- 详见6.3。**

#### 6.3 矩阵形式的对应

RoPE的二维旋转矩阵形式：
$$
R_m = \begin{bmatrix} \cos m\theta & -\sin m\theta \\\\ \sin m\theta & \cos m\theta \end{bmatrix}
$$

当我们将这个矩阵分别作用于 $\mathbf{q}$ 和 $\mathbf{k}$，然后计算它们的点积（内积）：
$$
\begin{aligned}
&\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle \\\\
&= (\mathbf{q}^T R_m^T) (R_n \mathbf{k}) \\\\
&= \mathbf{q}^T (R_m^T R_n) \mathbf{k}
\end{aligned}
$$

由于旋转矩阵是正交矩阵（$R^T = R^{-1}$），并且旋转矩阵的乘法满足 $R_m^T R_n = R_{n-m}$（相当于旋转角度相减）。
$$
R_{n-m}=\begin{bmatrix} \cos ((n-m)\theta) & -\sin ((n-m)\theta) \\\\ \sin ((n-m)\theta) & \cos ((n-m)\theta) \end{bmatrix}
$$

$$
R_{n-m}\begin{bmatrix} k_1 \\\\ k_2 \end{bmatrix}=\begin{bmatrix}\cos((n-m)\theta)k_1-sin((n-m)\theta)k_2 \\\\ \sin((n-m)\theta)k_1+\cos((n-m)\theta)k_2 \end{bmatrix}
$$

$$
\begin{aligned}
&[q_1,q_2]\left( R_{n-m}\begin{bmatrix} k_1 \\\\ k_2 \end{bmatrix} \right) \\\\ &=(q_1 k_1 + q_2 k_2)\cos((n-m)\theta)+(q_2 k_1-q_1 k_2)sin((n-m)\theta)
\end{aligned}
$$

**验证了6.2章节结尾的结论。**

**关键步骤**：计算 $R_{n-m}$ 与向量 $\mathbf{q}$ 和 $\mathbf{k}$ 的乘积，在展开计算内积时，**必然会用到三角函数的和差角公式**（积化和差公式的另一种形式），最终得到只包含 $(m-n)$ 的表达式。

## 三、RoPE 的优势

### 1. 外推性（Extrapolation）
RoPE的一个关键优势是**良好的外推能力**。传统的绝对位置编码（如正弦/余弦编码）在训练时只见过固定长度（如512），当推理时输入长度超过训练长度，模型性能会急剧下降。而RoPE通过旋转操作，即使输入序列长度远超训练长度，相对位置关系仍能保持一定的合理性，因此模型能更好地处理长文本。

### 2. 理论优雅
RoPE建立在复数旋转的数学基础上，形式简洁，与自注意力机制完美结合。其“内积仅依赖相对位置”的性质是严格推导出来的，而非启发式设计。

### 3. 兼容线性注意力
RoPE不改变向量的维度，只是进行旋转，因此可以与线性注意力（Linear Attention）等优化方法兼容。

### 4. 可扩展性
RoPE可以轻松扩展到更长的序列，只需调整旋转角度（如通过NTK-aware缩放）即可在不重新训练的情况下扩展上下文长度。


## 四、RoPE 的变体与改进

### 1. NTK-aware RoPE
为了进一步提升外推能力，苏剑林等人提出了NTK-aware RoPE。其核心思想是：**在扩展上下文长度时，对旋转角频率 $\theta_i$ 进行适当缩放**，避免高频维度（对应小 $i$）的旋转角度变化过快，从而保持模型对近距离token的区分能力。

具体做法是将 $\theta_i$ 修改为：
$$
\theta_i' = \theta_i \cdot \text{scale}^{-2i/(d-2)}
$$
其中 $\text{scale}$ 是缩放因子，通常与目标长度/训练长度的比例相关。

### 2. Dynamic NTK RoPE
在推理时根据输入长度动态调整缩放因子，实现更灵活的外推。

### 3. YaRN (Yet another RoPE extensioN)
YaRN结合了NTK-aware缩放和注意力温度调整，进一步优化了长文本的外推性能，被LLaMA 2等模型采用。


## 五、RoPE 的代码实现（简化版）

以下是一个简化的PyTorch实现，帮助理解RoPE的核心操作：
```python
import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        # 预计算频率：theta_i = base^(-2i/dim) for i in range(0, dim, 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # 不参与训练

    def forward(self, x, seq_len=None):
        # x: [batch_size, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        assert dim == self.dim, f"输入维度{dim}与初始化维度{self.dim}不匹配"

        # 生成位置序列 [0, 1, ..., seq_len-1]
        pos = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)  # [seq_len]

        # 计算每个位置的角度：pos * inv_freq
        # 外积： [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)  # [seq_len, dim/2]

        # 将freqs复制一份，因为每个频率对应两个维度（实部和虚部）
        freqs = freqs.repeat_interleave(2, dim=-1)  # [seq_len, dim]

        # 应用旋转：对x的每个位置，乘以旋转矩阵（通过复数乘法实现）
        # 将freqs扩展为 [1, seq_len, dim]，与x同形状
        freqs = freqs.unsqueeze(0)  # [1, seq_len, dim]

        # 构造旋转矩阵的cos和sin部分
        cos = torch.cos(freqs)  # [1, seq_len, dim]
        sin = torch.sin(freqs)  # [1, seq_len, dim]

        # 旋转操作：x' = [x0*cos - x1*sin, x0*sin + x1*cos, x2*cos - x3*sin, ...]
        # 将x的奇数位和偶数位分开
        x1, x2 = x[..., 0::2], x[..., 1::2]  # 分别取偶数位和奇数位
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # [batch, seq, dim/2, 2]

        # 恢复形状
        x_rotated = x_rotated.flatten(-2)  # [batch, seq, dim]

        return x_rotated
```
在实际应用中，RoPE通常直接集成到注意力层的Query和Key计算中，而不是作为一个独立的层。


## 六、总结

*   **RoPE原理**：通过将词向量的每一对维度视为一个复平面，并根据该词的位置对其进行旋转，从而将位置信息编码到向量中。
*   **与积化和差的关系**：RoPE利用**旋转操作**来编码位置信息，而旋转在数学上由三角函数描述。当计算两个经过不同位置旋转的向量的内积时，必须通过**积化和差公式**来化简表达式。正是这个公式的魔力，使得绝对位置项 $m$ 和 $n$ 相互抵消，最终只留下相对位置差 $(m-n)$。没有积化和差公式，RoPE就无法实现其最核心的特性——**相对位置感知**。

RoPE通过旋转操作将位置信息编码到token嵌入中，其核心优势在于：

- **理论优雅**：严格满足“内积仅依赖相对位置”的性质。
- **外推性强**：能有效处理远超训练长度的序列。
- **兼容性好**：与现有Transformer架构无缝集成。

随着大模型对长文本处理需求的增加，RoPE及其改进版本（如NTK-aware、YaRN）已成为位置编码的主流选择。理解RoPE的原理，对于深入掌握现代大模型的工作原理至关重要。