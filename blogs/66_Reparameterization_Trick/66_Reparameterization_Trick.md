**重参数化（Reparameterization Trick）** 是深度学习中处理随机变量梯度传播的**核心技术**，尤其在变分自编码器（VAE）、生成模型和连续动作空间的强化学习（如 SAC）中至关重要。它的核心思想是**将“随机采样”过程从计算图中移出，使其变得可导**。

## 一、 问题根源：随机采样为何不可导？

假设网络输出高斯分布的参数 $(\mu, \sigma)$，我们想要从这个分布中采样：
$$
a \sim \mathcal{N}(\mu, \sigma^2)
$$

**问题**：这个采样操作是**非确定性的**。计算图是：
```text
输入 → 网络(μ, σ) → 采样 → 输出 a
```

反向传播时，梯度无法通过“采样”这个随机操作回传到网络参数 $(\mu, \sigma)$，因为采样是**无梯度**的离散操作。

## 二、 解决方案：将随机性“外包”

**重参数化的巧妙之处**：将随机性转移到一个**独立的外部变量** $\epsilon$ 上，让采样过程变成**确定性变换**。

### 1. 数学表述
原始采样：
$$
a \sim \mathcal{N}(\mu, \sigma^2)
$$

重参数化后：
1. 从标准正态分布采样一个噪声：$\epsilon \sim \mathcal{N}(0, 1)$
2. 进行确定性变换：$a = \mu + \sigma \cdot \epsilon$

**关键洞察**：
- $\mu, \sigma$ 是网络的确定输出
- $\epsilon$ 是独立的外部噪声
- 整个表达式 $a = \mu + \sigma \cdot \epsilon$ 是**关于 $\mu, \sigma$ 的确定性函数**

### 2. 计算图变化
**重参数化前**（不可导）：
```text
输入 → 网络 → μ, σ → 随机采样 → 输出 a
                       ↑
                   (梯度中断)
```

**重参数化后**（可导）：
```text
输入 → 网络 → μ, σ → 确定性计算 a=μ+σ·ε → 输出 a
        ↑                ↑
   (梯度可回传)       (梯度可回传)
                     ε ∼ N(0,1) (外部采样，不参与求导)
```

**梯度计算**：
$$
\frac{\partial a}{\partial \mu} = 1, \quad \frac{\partial a}{\partial \sigma} = \epsilon
$$
现在梯度可以通过链式法则顺利传播回网络参数。

## 三、 在 SAC 中的具体应用

在 SAC 中，策略网络输出高斯分布的 $(\mu, \sigma)$，但为了将动作限制在 $[-1, 1]$ 范围内，通常还加上 $\tanh$ 激活：

### 1. 完整流程
```python
# SAC 策略网络的重参数化实现
def sample\_action(s):
    # 网络输出分布的均值和标准差
    mu, log\_sigma = policy\_network(s)  # log\_sigma 保证正定性
    sigma = exp(log\_sigma)
    
    # 重参数化：从标准正态分布采样
    epsilon = torch.randn\_like(mu)  # ε ∼ N(0, I)
    
    # 确定性变换
    raw\_action = mu + sigma * epsilon
    
    # tanh 变换将动作限制在 [-1, 1]
    action = tanh(raw\_action)
    
    return action, mu, sigma, epsilon, raw\_action
```

### 2. 对数概率的计算
由于做了 $\tanh$ 变换，概率密度需要修正。根据概率论中的**变量变换公式**：
$$
\log \pi(a|s) = \log \mathcal{N}(\tanh^{-1}(a) | \mu, \sigma^2) - \log(1 - a^2 + \text{eps})
$$
其中第二项是 $\tanh$ 变换的雅可比行列式的对数。

**代码实现**：
```python
def log\_prob(mu, sigma, raw\_action, action):
    # 原始高斯分布的对数概率
    logp\_gaussian = -0.5  ((raw\_action - mu) / sigma)2 - log(sigma) - 0.5  log(2*pi)
    
    # tanh变换的雅可比修正
    logp\_tanh = log(1 - action2 + 1e-6)
    
    return logp\_gaussian - logp\_tanh
```

## 四、 为什么必须用重参数化？

在 SAC 的 Actor 更新中，我们需要计算：
$$
\nabla\_\phi J(\phi) = \nabla\_\phi \mathbb{E}\_{a \sim \pi\_\phi} [Q(s,a) - \alpha \log \pi\_\phi(a|s)]
$$

### 1. 如果没有重参数化
我们需要使用 REINFORCE（Score Function）估计器：
$$
\nabla\_\phi J(\phi) = \mathbb{E}\_{a \sim \pi\_\phi} [\nabla\_\phi \log \pi\_\phi(a|s) \cdot (Q(s,a) - \alpha \log \pi\_\phi(a|s))]
$$
**问题**：方差极大，收敛缓慢。

### 2. 使用重参数化后
我们将期望内的采样 $a$ 用确定性函数表示：$a = f\_\phi(s, \epsilon)$
$$
\nabla\_\phi J(\phi) = \mathbb{E}\_{\epsilon \sim \mathcal{N}(0,1)} [\nabla\_a (Q(s,a) - \alpha \log \pi\_\phi(a|s)) \cdot \nabla\_\phi f\_\phi(s, \epsilon)]
$$
**优势**：
- 梯度通过 $Q(s,a)$ 直接传播，方差小
- 可以高效使用自动微分
- 训练更稳定、收敛更快

## 五、 物理意义与直观理解

想象训练一个机器人扔飞镖：

1. **传统方法**（无重参数化）：
   - 机器人每次随机扔（采样）
   - 教练说：“刚才那个角度偏了 5 度”
   - 机器人困惑：“但我每次都是随机扔的啊，怎么调？”

2. **重参数化方法**：
   - 机器人有个“基准姿势”$(\mu, \sigma)$
   - 每次扔之前，加一点随机颤抖 $\epsilon$
   - 最终姿势 = 基准姿势 + 颤抖
   - 教练说：“刚才那个角度偏了 5 度”
   - 机器人知道：“哦，是我的基准姿势需要调整 5 度，颤抖程度（$\sigma$）需要调小一点”

**核心**：将“随机性”与“可学习的参数”**解耦**，让网络学习“基准”，而将不确定性委托给外部噪声。

## 六、 通用公式与扩展

对于任意分布，重参数化的一般形式是：
$$
z = g\_\phi(\epsilon), \quad \epsilon \sim p(\epsilon)
$$
其中 $g\_\phi$ 是参数 $\phi$ 的确定性函数。

| 分布类型 | 重参数化形式 | 噪声分布 |
| :--- | :--- | :--- |
| **高斯分布** | $z = \mu + \sigma \cdot \epsilon$ | $\epsilon \sim \mathcal{N}(0,1)$ |
| **均匀分布** | $z = a + (b-a) \cdot \epsilon$ | $\epsilon \sim \mathcal{U}(0,1)$ |
| **指数分布** | $z = -\log(1-\epsilon)/\lambda$ | $\epsilon \sim \mathcal{U}(0,1)$ |

## 七、 总结

**重参数化的本质**：通过**变量变换**，将随机采样 $z \sim p\_\phi(z)$ 重新表述为 $z = g\_\phi(\epsilon), \epsilon \sim p(\epsilon)$，使得：
1. 随机性完全由 $\epsilon$ 承担
2. $g\_\phi$ 是关于 $\phi$ 的确定性、可微函数
3. 梯度可以正常通过计算图回传

**在强化学习中的重要性**：
- 使连续动作空间的策略梯度估计**方差大幅降低**
- 是 SAC、TD3 等先进算法能**稳定训练**的关键
- 让端到端训练随机策略成为可能

**一句话总结**：重参数化是一种“偷梁换柱”的数学技巧，将不可导的随机采样转化为可导的确定性计算，是连接深度学习与概率建模的桥梁。