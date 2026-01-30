<span class="image main">
<img class="main img-in-blog" style="max-width: 60%" src="./blogs/14_Activation_Functions/activation_func.webp" alt="Activition Functions" />
<i>激活函数</i>
</span>

### 1. Sigmoid函数

<span class="image main">
<img class="main img-in-blog" style="max-width: 40%" src="./blogs/14_Activation_Functions/Sigmoid.webp" alt="Sigmoid" />
<i>Sigmoid 激活函数, By <a href="//commons.wikimedia.org/wiki/User:Qef" title="User:Qef">Qef</a> (<a href="//commons.wikimedia.org/wiki/User_talk:Qef" title="User talk:Qef">talk</a>) - Created from scratch with gnuplot, Public Domain, <a href="https://commons.wikimedia.org/w/index.php?curid=4310325">Link</a></i>
</span>

**公式**：$σ(x) = 1 / (1 + e^{-x})$

**特点**：

- 输出范围(0,1)，适合表示概率
- 平滑且处处可导
- 容易饱和，梯度消失问题严重

**实用场景**：

- **二分类**问题的输出层
- 需要输出概率值的场景
- 早期神经网络中广泛使用

### 2. Tanh函数

<span class="image main">
<img class="main img-in-blog" style="max-width: 40%" src="./blogs/14_Activation_Functions/Tanh.webp" alt="Tanh" />
<i>Tanh 激活函数</i>
</span>

**公式**：$tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})$

**特点**：

- 输出范围(-1,1)，零中心化
- 比Sigmoid梯度更强
- 同样存在梯度消失问题

**实用场景**：

- 隐藏层激活函数
- RNN、LSTM等循环神经网络
- 需要**零中心化**输出的场景

### 3. ReLU函数

<span class="image main">
<img class="main img-in-blog" style="max-width: 40%" src="./blogs/14_Activation_Functions/ReLU_and_GELU.webp" alt="ReLU and GELU" />
<i>ReLU 和 ELU 激活函数</i>
</span>

**公式**：$f(x) = \text{max}(0, x)$

**特点**：

- 计算简单，收敛速度快
- 解决梯度消失问题（正区间）
- 存在"死神经元"问题（负区间梯度为0）

**实用场景**：

- 深度神经网络的隐藏层（最常用）
- CNN、DNN等前馈网络
- 需要快速训练的场景

### 4. Leaky ReLU

<span class="image main">
<img class="main img-in-blog" style="max-width: 40%" src="./blogs/14_Activation_Functions/ReLU-Leaky-ReLU-ELU.webp" alt="ReLU, Leaky-ReLU and GELU" />
<i>ReLU, Leaky-ReLU 和 GELU 激活函数</i>
</span>

**公式**：$f(x) = \text{max}(αx, x)$，$α$通常取0.01

**特点**：

- 解决ReLU的"死神经元"问题
- 负区间有微小梯度
- 计算依然简单

**实用场景**：

- 替代ReLU的隐藏层激活
- 防止神经元死亡
- 对**负值**敏感的任务

### 5. ELU函数

<span class="image main">
<img class="main img-in-blog" style="max-width: 40%" src="./blogs/14_Activation_Functions/ReLU-Leaky-ReLU-ELU.webp" alt="ReLU, Leaky-ReLU and GELU" />
<i>ReLU, Leaky-ReLU 和 GELU 激活函数</i>
</span>

**公式**：$f(x) = x (x > 0), α(e^x - 1) (x ≤ 0)$

**特点**：

- 负区间平滑，接近零均值
- 缓解梯度消失问题
- 计算成本略高

**实用场景**：

- 需要更好性能的深度网络
- 对噪声敏感的任务
- 图像分类等复杂任务

### 6. GELU函数

<span class="image main">
<img class="main img-in-blog" style="max-width: 40%" src="./blogs/14_Activation_Functions/ReLU_and_GELU.webp" alt="ReLU and GELU" />
<i>ReLU 和 GELU 激活函数</i>
</span>

**公式**：$GELU(x) = x · \Phi(x) = x\cdot \frac{1}{2}[1 + erf(\frac{x}{\sqrt{2}})]$

**特点**：

- 基于高斯分布的平滑近似
- 结合了ReLU的稀疏性和Sigmoid的平滑性
- 在Transformer架构中广泛使用
- 计算成本较高，但性能优异

**实用场景**：

- Transformer架构（BERT、GPT、T5等）
- 自然语言处理任务
- 需要平滑激活的深度网络
- 对性能要求较高的模型

### 7. Softmax函数

**公式**：$σ(z)_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$

**特点**：

- 将输出转换为概率分布
- 所有输出和为1
- 适合多分类问题

**实用场景**：

- 多分类问题的输出层
- 需要输出类别概率的场景
- 分类任务的标准选择

### 综合对比表

| 激活函数       | 输出范围 | 是否平滑 | 梯度消失 | 计算复杂度 | 主要应用场景           |
| -------------- | -------- | -------- | -------- | ---------- | ---------------------- |
| **Sigmoid**    | (0,1)    | ✅ 平滑   | 严重     | 中等       | 二分类输出层           |
| **Tanh**       | (-1,1)   | ✅ 平滑   | 严重     | 中等       | RNN隐藏层              |
| **ReLU**       | [0, +∞)  | ❌ 不连续 | 正区间无 | 极低       | CNN、DNN隐藏层         |
| **Leaky ReLU** | (-∞, +∞) | ❌ 不连续 | 缓解     | 极低       | 替代ReLU               |
| **ELU**        | (-α, +∞) | ✅ 平滑   | 缓解     | 中等       | 深度网络               |
| **GELU**       | (-∞, +∞) | ✅ 平滑   | 缓解     | 较高       | Transformer、BERT、GPT |
| **Softmax**    | (0,1)    | ✅ 平滑   | -        | 高         | 多分类输出层           |

### 选择建议

**隐藏层选择**：

- 通用场景：ReLU/Leaky ReLU（计算简单，速度快）
- 深度网络：ELU/GELU（性能更好，缓解梯度消失）
- 循环网络：Tanh（零中心化）

**输出层选择**：

- 二分类：Sigmoid
- 多分类：Softmax

**特殊场景**：

- Transformer架构：GELU（标准配置）
- 计算资源受限：ReLU系列
- 需要最佳性能：GELU/ELU

**核心原则**：根据任务类型、网络深度和计算资源灵活选择，ReLU系列适合大多数隐藏层，Sigmoid/Softmax用于输出层，GELU在Transformer中表现优异。