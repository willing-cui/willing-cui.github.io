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