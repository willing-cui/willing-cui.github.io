M-RoPE（多维旋转位置编码）的数学原理建立在复数域旋转变换的基础上，通过多维解耦和频率分组机制来增强位置编码的表达能力。以下从基础数学、多维扩展、频率分配、注意力计算四个层面进行详细推导。


## 一、基础数学框架：复数旋转表示

### 1.1 复数表示与欧拉公式

在复数域中，任意复数可表示为：
$$ z = a + bi = re^{i\theta} $$

其中$r = \sqrt{a^2 + b^2}$为模长，$\theta = \arctan(b/a)$为幅角。欧拉公式：
$$ e^{i\theta} = \cos\theta + i\sin\theta $$

### 1.2 旋转变换的矩阵形式

对于二维向量$\begin{bmatrix} x \\\\ y \end{bmatrix}$，绕原点旋转角度$\theta$的变换矩阵为：
$$ R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix} $$

在复数域中，旋转操作等价于乘以$e^{i\theta}$。将复数$z = x + yi$旋转角度$\theta$：
$$ z' = z \cdot e^{i\theta} = (x + yi)(\cos\theta + i\sin\theta) = (x\cos\theta - y\sin\theta) + i(x\sin\theta + y\cos\theta) $$

对应实部虚部分离，即：
$$ \begin{bmatrix} x' \\\\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\\\ y \end{bmatrix} $$


## 二、标准RoPE的数学构造

### 2.1 位置编码的基本要求

对于位置$m$和$n$，我们希望位置编码函数$f(\cdot)$满足：
$$ \langle f(q, m), f(k, n) \rangle = g(q, k, m-n) $$

即内积只依赖于相对位置$m-n$，这是相对位置编码的核心性质。

### 2.2 RoPE的构造方法

<span class="image main">
<img class="main img-in-blog" style="max-width: 50%" src="./blogs/41_M_RoPE_Positional_Encoding/RoPE_QK.webp" alt="RoPE Q/K matrix" />
<i>Q/K matrix with RoPE</i>
</span> 

对于$d$维向量，将维度两两分组（$d$为偶数，分得$d/2$组），每组对应一个旋转频率。设第$i$组（对应维度$2i, 2i+1$）的旋转频率为$\theta\_i$，则位置$m$处的旋转矩阵为：

$$ R\_{\Theta}(m) = \text{diag}\left( e^{i\theta\_1 m}, e^{i\theta\_2 m}, \cdots, e^{i\theta\_{d/2} m} \right) $$

实际实现中，对于查询向量$q$和键向量$k$，位置编码为：
$$ q\_m = R\_{\Theta}(m)q,\quad k\_n = R\_{\Theta}(n)k $$

#### 2.2.1 RoPE只对Q和K做位置编码吗？

**是的，RoPE（旋转位置编码）通常只对Q（查询向量）和K（键向量）进行位置编码，而不对V（值向量）做位置编码。**

这是RoPE在Transformer架构中的标准实现方式。原因在于：注意力计算的核心是Q和K之间的相似度（通过点积运算），位置信息主要影响的是"查询与键的匹配关系"。通过将位置编码融入Q和K，可以使模型在计算注意力权重时感知到token之间的相对位置关系。而V向量承载的是内容信息，在注意力机制中只参与加权求和，不参与相似度计算，因此通常不需要额外添加位置编码。

不过需要说明的是，RoPE本身作为一种位置编码方法，理论上可以应用于任何向量。但在实际的大语言模型（如LLaMA、GPT等）中，为了计算效率和设计简洁性，都遵循了"只编码Q和K"的实践方案。这种设计已被证明在保持位置感知能力的同时，不会损失模型表达能力。

#### 2.2.2 记得位置编码是加在模型输入的embedding上？

**加性位置编码**（如原始Transformer的sinusoidal位置编码）确实是直接加到输入embedding上的，但**RoPE（旋转位置编码）的实现方式不同**——它是在计算注意力时通过旋转变换作用于Q和K向量，而不是在输入阶段直接加到embedding上。

**关键区别在于作用时机和方式**：

| 编码类型                     | 作用位置        | 实现方式                             | 特点                                                 |
| ---------------------------- | --------------- | ------------------------------------ | ---------------------------------------------------- |
| 加性位置编码（如sinusoidal） | 输入embedding层 | 将位置编码向量与词向量逐元素相加     | 位置信息在输入阶段就融合，后续所有计算都包含位置信息 |
| RoPE（旋转位置编码）         | 注意力计算层    | 对Q和K向量进行旋转变换（复数域旋转） | 只在计算Q·K时引入位置信息，V和FFN层不涉及            |

**为什么RoPE不直接加在embedding上？**

RoPE的核心思想是通过旋转变换让Q和K的内积结果包含相对位置信息，这种"旋转"操作需要在向量空间中进行，无法简单地用"加"的方式实现。它本质上是一种**乘性变换**（通过旋转矩阵相乘），而不是加性操作。

**简单理解**：加性位置编码是"先加后算"，RoPE是"在计算时通过旋转引入位置信息"。两者都是有效的位置编码方案，但实现路径不同。当前主流大模型（如LLaMA系列）采用RoPE正是因为它在长序列外推、相对位置建模等方面表现更优。

所以关于“位置编码加在embedding上”的说法对应的是原始Transformer方案，而RoPE是另一种更现代的实现方式。

### 2.3 内积性质验证

$$ \langle q\_m, k\_n \rangle = q^\top R\_{\Theta}(m)^\top R\_{\Theta}(n)k = q^\top R\_{\Theta}(n-m)k $$

由于$R\_{\Theta}(m)^\top R\_{\Theta}(n) = R\_{\Theta}(n-m)$（旋转矩阵的共轭转置性质），因此内积只依赖于相对位置$n-m$，满足相对位置编码要求。

### 2.4 频率参数设置

标准RoPE采用几何级数分配频率：
$$ \theta\_i = 10000^{-2i/d} $$

这种分配使得不同维度具有不同的旋转频率，高频维度对位置变化更敏感。


## 三、M-RoPE的多维扩展机制

### 3.1 维度分组策略

<span class="image main">
<img class="main img-in-blog" style="max-width: 60%" src="./blogs/41_M_RoPE_Positional_Encoding/M-RoPE_QK.webp" alt="M-RoPE Q/K matrix" />
<i>Q/K matrix with M-RoPE</i>
</span> 

M-RoPE的核心改进是将$d$维特征空间划分为$K$个维度组（$K$通常为2-4）：
$$ \text{Dim} = G\_1 \cup G\_2 \cup \cdots \cup G\_K $$

每个维度组$G\_k$包含$d\_k$个维度，满足$\sum\_{k=1}^K d\_k = d$。不同维度组采用不同的**基频率**$\omega\_k$。

### 3.2 组内频率分配

对于维度组$G\_k$中的第$j$个维度（$j=1,2,\ldots,d\_k$），其旋转频率为：
$$ \theta\_{k,j} = \omega\_k \cdot \lambda\_j $$

其中$\lambda\_j$是组内频率倍数，通常采用线性或几何级数分配。例如：
- 线性分配：$\lambda\_j = j$
- 几何分配：$\lambda\_j = \gamma^{j-1}$（$\gamma$为公比）

### 3.3 完整的位置编码矩阵

对于位置$m$，M-RoPE的旋转矩阵是一个块对角矩阵：

$$ R\_{\Theta}(m) = \begin{bmatrix}
R\_{G\_1}(m) & & \\\\
& R\_{G\_2}(m) & \\\\
& & \ddots & \\\\
& & & R\_{G\_K}(m)
\end{bmatrix} $$

其中每个子块$R\_{G\_k}(m)$是维度组$G\_k$对应的旋转矩阵，其形式为：
$$ R\_{G\_k}(m) = \text{diag}\left( e^{i\theta\_{k,1}m}, e^{i\theta\_{k,2}m}, \cdots, e^{i\theta\_{k,d\_k}m} \right) $$


## 四、频率参数的可学习机制

### 4.1 可学习参数定义

M-RoPE支持将基频率$\omega\_k$作为可训练参数。设可学习频率向量为：
$$ \boldsymbol{\omega} = [\omega\_1, \omega\_2, \ldots, \omega\_K] $$

在训练过程中，$\boldsymbol{\omega}$通过梯度下降进行优化。

### 4.2 梯度计算

设损失函数为$\mathcal{L}$，则对$\omega\_k$的梯度为：
$$ \frac{\partial \mathcal{L}}{\partial \omega\_k} = \sum\_{j=1}^{d\_k} \frac{\partial \mathcal{L}}{\partial \theta\_{k,j}} \cdot \frac{\partial \theta\_{k,j}}{\partial \omega\_k} = \sum\_{j=1}^{d\_k} \frac{\partial \mathcal{L}}{\partial \theta\_{k,j}} \cdot \lambda\_j $$

其中$\frac{\partial \mathcal{L}}{\partial \theta\_{k,j}}$可通过链式法则从注意力计算反向传播得到。

### 4.3 参数初始化

可学习频率通常采用以下初始化策略：
- 几何级数初始化：$\omega\_k = \text{base}^{-k/K}$（$\text{base}$通常取10000）
- 均匀分布初始化：$\omega\_k \sim \mathcal{U}(0, 1)$
- 固定频率初始化：从标准RoPE的频率分布中采样


## 五、注意力计算中的具体实现

### 5.1 查询和键的旋转

对于位置$m$处的查询向量$q$和位置$n$处的键向量$k$，分别应用旋转：

$$ q\_m = R\_{\Theta}(m)q = \begin{bmatrix}
R\_{G\_1}(m)q\_{G\_1} \\\\
R\_{G\_2}(m)q\_{G\_2} \\\\
\vdots \\\\
R\_{G\_K}(m)q\_{G\_K}
\end{bmatrix},\quad
k\_n = R\_{\Theta}(n)k = \begin{bmatrix}
R\_{G\_1}(n)k\_{G\_1} \\\\
R\_{G\_2}(n)k\_{G\_2} \\\\
\vdots \\\\
R\_{G\_K}(n)k\_{G\_K}
\end{bmatrix} $$

其中$q\_{G\_k}$和$k\_{G\_k}$分别是$q$和$k$在维度组$G\_k$上的子向量。

### 5.2 注意力分数计算

注意力分数为：
$$ \text{Attention}(q\_m, k\_n) = \langle q\_m, k\_n \rangle = \sum\_{k=1}^K \langle R\_{G\_k}(m)q\_{G\_k}, R\_{G\_k}(n)k\_{G\_k} \rangle $$

由于旋转矩阵是酉矩阵，内积可进一步展开：
$$ \langle R\_{G\_k}(m)q\_{G\_k}, R\_{G\_k}(n)k\_{G\_k} \rangle = q\_{G\_k}^\top R\_{G\_k}(n-m) k\_{G\_k} $$

因此最终：
$$ \text{Attention}(q\_m, k\_n) = \sum\_{k=1}^K q\_{G\_k}^\top R\_{G\_k}(n-m) k\_{G\_k} $$

这保持了相对位置编码的性质，且不同维度组贡献不同的位置感知能力。


## 六、数学性质分析

### 6.1 相对位置保持性

从上述推导可知，M-RoPE严格满足：
$$ \langle q\_m, k\_n \rangle = f(q, k, n-m) $$

即注意力分数只依赖于相对位置$n-m$，与绝对位置无关。这是位置编码的核心要求。

### 6.2 多尺度位置感知

不同维度组$G\_k$采用不同的基频率$\omega\_k$：
- 低频组（$\omega\_k$较小）：旋转缓慢，对长距离位置变化敏感
- 高频组（$\omega\_k$较大）：旋转快速，对短距离位置变化敏感

这种设计使得模型能够同时捕捉局部和全局的位置关系。

### 6.3 参数复杂度

设维度$d$，分组数$K$，则：
- 固定频率模式：参数量为$K$（基频率）+ $d$（组内分配）= $O(d)$
- 可学习频率模式：参数量为$K$（可学习基频率），相比标准RoPE增加$K$个参数

计算复杂度：相比标准RoPE增加约$O(K)$的旋转操作，实际计算量增加约$10$-$20\\%$。


## 七、与标准RoPE的数学对比

| 数学特性 | 标准RoPE | M-RoPE |
|---------|---------|--------|
| **频率分配** | 单一几何级数：$\theta\_i = \text{base}^{-2i/d}$ | 多维分组：$\theta\_{k,j} = \omega\_k \cdot \lambda\_j$ |
| **维度耦合** | 所有维度共享频率分布 | 维度组间解耦，组内耦合 |
| **参数数量** | $0$（固定）或$d/2$（可学习） | $K$（可学习基频率） |
| **表达能力** | 单尺度位置感知 | 多尺度位置感知 |
| **外推能力** | 随长度衰减 | 低频组缓解衰减 |


## 八、总结

M-RoPE的数学核心在于：
1. **复数旋转基**：通过$e^{i\theta m}$实现相对位置编码
2. **多维分组**：将特征空间划分为多个频率组，实现多尺度感知
3. **频率可学习**：通过端到端训练优化频率分布
4. **块对角结构**：保持相对位置性质的同时增强表达能力

这种设计在数学上保证了位置编码的合理性，同时通过频率分组机制提升了模型对复杂位置关系的建模能力。实际应用中，需要根据任务特性调整分组策略和频率初始化方式。

**注**：以上数学推导基于公开的M-RoPE论文和开源实现，具体实现细节可能因版本不同而略有差异。建议在实际使用时参考官方代码库的数学实现。