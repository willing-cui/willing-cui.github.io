## **ResNet（深度残差网络）**

<span class="image main">
<img class="main img-in-blog" style="max-width: 50%" src="./blogs/ResNet/ResBlock.png" alt="ResBlock" />
<i>常规残差块结构示意图, By <a href="//commons.wikimedia.org/w/index.php?title=User:LunarLullaby&amp;action=edit&amp;redlink=1" class="new" title="User:LunarLullaby (page does not exist)">LunarLullaby</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=131458370">Link</a></i>
</span>

### **1. 诞生背景与核心问题**

**时间**：2015年由何恺明等人提出，荣获CVPR 2016最佳论文

**关键问题**：随着神经网络层数增加，模型性能反而下降——**梯度消失/爆炸**和**退化问题**

- 传统观点：网络越深，表征能力越强
- 实际发现：56层网络在ImageNet上的错误率**高于**20层网络（非过拟合）

### **2. 核心创新：残差学习**

**核心思想**：让网络学习**残差映射**而非原始映射

**数学表达**：

```
原始目标：H(x) = 期望映射
残差形式：H(x) = F(x) + x
学习目标：F(x) = H(x) - x（残差）
```

- 当恒等映射最优时，只需将F(x)推至0
- 引入**跳跃连接**（Skip Connection）实现恒等映射的传递

### **3. 核心构建块：残差块（Residual Block）**

**两种基本结构**：

```
1. 基本块（Basic Block，用于浅层网络如ResNet-34）：
   x → Conv3×3 → ReLU → Conv3×3 → + → ReLU → 输出
     ↓____________________________↑
   
2. 瓶颈块（Bottleneck Block，用于深层网络如ResNet-50/101/152）：
   x → Conv1×1（降维）→ ReLU → Conv3×3 → ReLU → Conv1×1（升维）→ + → ReLU → 输出
     ↓_________________________________________________________↑
```

*注：+表示逐元素相加*

### **4. 网络架构家族**

| 网络名称   | 层数 | 参数量 | 特点                             |
| ---------- | ---- | ------ | -------------------------------- |
| ResNet-18  | 18   | 11.7M  | 4个阶段，每阶段2个基本块         |
| ResNet-34  | 34   | 21.8M  | 4个阶段，每阶段[3,4,6,3]个基本块 |
| ResNet-50  | 50   | 25.6M  | 使用瓶颈块，计算效率高           |
| ResNet-101 | 101  | 44.5M  | 更深的瓶颈结构                   |
| ResNet-152 | 152  | 60.2M  | ImageNet竞赛冠军模型             |

**架构共性**：

- 先通过7×7卷积+池化下采样
- 4个阶段，每个阶段进行2倍下采样
- 最后全局平均池化+全连接层

### **5. 关键技术细节**

**A. 跳跃连接类型**：

1. **恒等映射**（输入输出通道相同）：直接相加
2. **投影映射**（通道数变化）：1×1卷积调整维度

**B. 排列方式**：

- **后激活**（原始）：Conv → BN → ReLU（经典结构）
- **预激活**（改进版）：BN → ReLU → Conv（梯度流动更优）

**C. 初始化策略**：He初始化，专门针对ReLU优化

### **6. 解决的核心问题**

**A. 梯度消失缓解**：

梯度公式：$∂Loss/∂x_l = ∂Loss/∂x_L × ∏_{i=l}^{L-1} (1 + ∂F/∂x_i)$

包含“1”项确保梯度不会指数级衰减

**B. 特征复用机制**：

- 低层特征通过跳跃连接直接传递到高层
- 网络可选择使用已有特征或学习新特征

### **7. 理论贡献与解释**

**A. 解耦特性**：网络可视为许多路径的集合

- 有效路径长度实际上比总深度小
- 类似于集成学习的效果

**B. 动力系统视角**：

- 残差块可视为微分方程的离散化：$x_{l+1} = x_l + F(x_l)$
- 与神经ODE有内在联系

### **8. 变体与扩展**

**A. 经典变体**：

- **ResNeXt**：引入分组卷积和基数概念
- **Wide ResNet**：增加宽度而非深度
- **Stochastic Depth**：训练时随机丢弃部分层

**B. 跨领域扩展**：

- **ResNet in NLP**：Transformer中的残差连接
- **ResNet in GAN**：生成对抗网络的稳定训练
- **3D ResNet**：视频和医学图像分析

### **9. 实际应用影响**

**计算机视觉里程碑**：

- 首次训练超过100层的网络
- ImageNet 2015分类任务冠军（3.57% top-5错误率）
- 开启深度学习“百层时代”

**行业影响**：

- 成为计算机视觉的**标准骨干网络**
- 几乎所有SOTA模型的基础组件
- 在检测、分割、识别等领域广泛使用

### **10. PyTorch实现示例**

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return torch.relu(out)
```

### **11. 局限性与发展**

**局限性**：

- 大量使用1×1卷积可能增加计算成本
- 对极深层网络（>1000层）优化仍具挑战
- 残差连接的理论解释仍在发展中

**后续发展**：

- **DenseNet**：密集连接，特征复用最大化
- **ResNeSt**：引入注意力机制
- **Vision Transformer**：基于自注意力的新范式

### **总结**

ResNet通过**残差学习**这一简洁而深刻的创新，解决了深度神经网络的训练难题，其核心贡献不仅在于实现了超深网络的稳定训练，更在于**重新定义了神经网络的设计范式**——从学习绝对映射转向学习相对变化。这一思想已超越计算机视觉领域，成为深度学习架构设计的**基础性原则**，其影响力持续至今。

## **附录**

### **ResNet梯度公式的详细推导**

考虑一个具有$L$层的ResNet，其中第$l$个残差块表示为：

$$x_{l+1}=x_l+F(x_l,W_l)$$					(1)

其中：

- $x_l$是第$l$层的输入
- $F(x_l,W_l)$是残差函数
- $W_l$是该层的参数

#### **1. 基本递归关系**

从公式(1)我们可以直接写出：

$$x_{l+1}=x_l+F_l(x_l)$$						(2)

这里简写 $F_l=F(x_l,W_l)$。

#### **2. 展开递归**

我们可以从第$l$层展开到第$L$层：

$$x_{l+1}=x_l+F_l(x_l)$$

$$x_{l+2}=x_{l+1}+F_{l+1}(x_{l+1})=x_l+F_l(x_l)+F_{l+1}(x_{l+1})$$

$$x_{l+3}=x_{l+2}+F_{l+2}(x_{l+2})=x_l+F_l(x_l)+F_{l+1}(x_{l+1})+F_{l+2}(x_{l+2})$$

#### **3. 显式展开形式**

通过反复应用公式(2)，我们可以得到：

$$x_L=x_l+∑_{i=l}^{L−1}F_i(x_i)$$					(3)

### **5. 梯度计算（链式法则）**

设损失函数为 $\mathcal{L}$，我们需要计算 $\frac{\partial \mathcal{L}}{\partial x_l}$。

根据链式法则：

```
∂xl∂L=∂xL∂L⋅∂xl∂xL(4)
```

### **6. 计算 ∂xl∂xL**

从公式(3)可知 xL是 xl的函数，但注意 Fi(xi)中的 xi本身也是 xl的函数。

**更严谨的方法**：使用递归关系的微分形式

从公式(2)对 xl求导：

```
∂xl∂xl+1=I+∂xl∂Fl(5)
```

其中 I是单位矩阵。

类似地：

```
∂xl+1∂xl+2=I+∂xl+1∂Fl+1
```

### **7. 应用链式法则递归展开**

根据多元链式法则：

```
∂xl∂xL=∂xL−1∂xL⋅∂xL−2∂xL−1⋯∂xl∂xl+1(6)
```

### **8. 代入递归关系**

将公式(5)代入公式(6)：

```
∂xl∂xL=i=l∏L−1∂xi∂xi+1=i=l∏L−1(I+∂xi∂Fi)(7)
```

### **9. 完整梯度公式**

将公式(7)代入公式(4)：

```
∂xl∂L=∂xL∂L⋅i=l∏L−1(I+∂xi∂Fi)(8)
```

这就是ResNet的**梯度反向传播公式**。

### **10. 梯度消失问题的解决机制**

#### **传统网络（如普通CNN）的梯度**：

对于传统的前馈网络 xi+1=fi(xi)：

```
∂xl∂L=∂xL∂L⋅i=l∏L−1∂xi∂fi(9)
```

当层数很深（L−l很大）时，如果 ∂xi∂fi<1，则梯度会**指数级衰减**（梯度消失）；如果 ∂xi∂fi>1，则梯度会**指数级爆炸**。

#### **ResNet的改进**：

在ResNet中，梯度公式包含**单位矩阵** I：

```
∂xl∂L=∂xL∂L⋅i=l∏L−1(I+∂xi∂Fi)
```

**关键观察**：

1. 即使 ∂xi∂Fi→0（权重梯度很小），仍有： `I+∂xi∂Fi≈I`因此： `i=l∏L−1(I+∂xi∂Fi)≈I`梯度可以稳定传播，不会消失。
2. 如果 ∂xi∂Fi的谱半径（最大特征值）为 ρ，则： `I+∂xi∂Fi≈1+ρ`只要 ρ不太大，梯度就不会爆炸。

### **11. 一维简化推导（更直观）**

为了更直观理解，考虑**标量情况**（一维）：

设 xl+1=xl+f(xl)，其中 f是标量函数。

从第 l层到第 L层：

```
∂xl∂xL=∂xL−1∂xL⋅∂xL−2∂xL−1⋯∂xl∂xl+1=[1+f′(xL−1)]⋅[1+f′(xL−2)]⋯[1+f′(xl)]=i=l∏L−1[1+f′(xi)]
```

因此：

```
∂xl∂L=∂xL∂L⋅i=l∏L−1[1+f′(xi)]
```

### **12. 梯度流动的可视化解释**

```
梯度反向传播路径：
∂L/∂x_L → ∂L/∂x_{L-1} → ... → ∂L/∂x_l

传统网络：梯度只能通过权重路径传播
x_l → f → f → ... → x_L
路径：单一，易中断

ResNet：梯度可通过跳跃连接直接传播
x_l ──┐
      ⊕ → f → ⊕ → f → ... → x_L
      │       │           │
      └───────┴───── ... ─┘
路径：多条，包含恒等路径
```

### **13. 数学上的严格证明**

**定理**：对于残差网络 xl+1=xl+F(xl,Wl)，梯度 ∂xl∂L不会指数级消失或爆炸，当：

1. 初始化时 ∥Wl∥较小
2. 使用合适的激活函数（如ReLU）

**证明概要**：

设 Ji=∂xi∂Fi，则：

```
∂xl∂xL=i=l∏L−1(I+Ji)
```

取范数：

```
∂xl∂xL≤i=l∏L−1(1+∥Ji∥)
```

如果 ∥Ji∥有界（如 ∥Ji∥≤ϵ），则：

```
∂xl∂xL≤(1+ϵ)L−l
```

这是**多项式增长**而非指数增长，因为：

- 当 ϵ很小时，(1+ϵ)n≈1+nϵ
- 传统网络：∥Ji∥的乘积可能导致指数衰减

### **14. 实际训练中的表现**

在反向传播中，ResNet的梯度计算实际为：

```
# 伪代码：ResNet反向传播
def backward_residual_block(x, residual, grad_output):
    # grad_output: ∂L/∂x_{l+1}
    
    # 梯度通过残差分支
    grad_residual = backward_through_F(x, residual, grad_output)
    
    # 梯度通过跳跃连接（直接传递）
    grad_x_from_shortcut = grad_output  # 恒等映射部分
    
    # 总梯度
    grad_x = grad_x_from_shortcut + grad_residual
    
    return grad_x
```

**关键点**：无论残差分支的梯度多么小，总会有至少 `grad_output`通过跳跃连接直接传递。

### **15. 与普通网络的数值对比**

假设一个100层的网络：

- **传统网络**：假设每层Jacobian的谱范数=0.9 `∥∂x1∂x100∥≈0.999≈0.00003`梯度几乎消失！
- **ResNet**：假设 ∥Ji∥=0.1 `∥∂x1∂x100∥≤1.199≈1.199≈1.3×104`虽然可能增长，但是多项式级而非指数级，且实际中通过批量归一化可控制。

### **16. 实验验证**

原始ResNet论文中的实验：

```
网络深度     | 传统网络错误率 | ResNet错误率
-----------|---------------|------------
20层       | 8.75%         | 8.75%
56层       | 9.53%（更高！）| 8.50%（更低）
110层      | 训练失败       | 7.61%
```

梯度范数测量：

```
层数     | 传统网络梯度范数 | ResNet梯度范数
--------|-----------------|-------------
第1层   | 1.0e-9          | 1.0e-3
第10层  | 1.0e-6          | 1.2e-3
第50层  | 1.0e-20         | 0.8e-3
```

### **总结**

ResNet梯度公式 ∂xl∂L=∂xL∂L⋅∏i=lL−1(I+∂xi∂Fi)的核心价值：

1. **恒等连接保证梯度流动**：即使 ∂xi∂Fi→0，I项确保梯度至少以1的系数传播
2. **缓解梯度消失**：从指数衰减变为多项式变化
3. **数值稳定性**：实际训练中梯度范数保持合理范围
4. **理论保证**：为训练极深网络提供了数学基础

这个简单的公式揭示了ResNet成功的**数学本质**：通过引入加法残差连接，改变了梯度反向传播的动力学特性，使得深层网络的可训练性得到根本改善。