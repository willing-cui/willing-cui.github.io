深度学习训练中，学习率调度器（Learning Rate Scheduler）通过数学公式动态调整学习率，以平衡收敛速度与精度。以下是主流调度器的数学表达式、核心参数及适用场景。

## 1. 阶梯式衰减 (StepLR / MultiStepLR)
*   **数学表达式**：
    *   **StepLR**：$$lr\_t = lr\_0 \times \gamma^{\lfloor t / s \rfloor}$$
    *   **MultiStepLR**：$$lr\_t = lr\_0 \times \gamma^{k}$$，其中 $k$ 是满足 $t \ge milestones[k]$ 的最大索引
*   **核心参数**：
    *   $lr\_0$：初始学习率
    *   $\gamma$：衰减因子（通常为0.1）
    *   $s$：衰减步长（StepLR）
    *   $milestones$：衰减节点列表（MultiStepLR）
*   **适用场景**：传统CNN训练（如ResNet）、实验初期调试
*   **变化曲线**：**阶梯状**，在特定轮次学习率突然下降

## 2. 指数衰减 (ExponentialLR)
*   **数学表达式**：$$lr\_t = lr\_0 \times \gamma^{t}$$
*   **核心参数**：
    *   $lr\_0$：初始学习率
    *   $\gamma$：衰减因子（通常为0.99）
*   **适用场景**：需要平滑衰减的小模型实验
*   **变化曲线**：**指数下降曲线**，初期下降快，后期缓慢

## 3. 余弦退火 (CosineAnnealingLR)
*   **数学表达式**：$$lr\_t = lr\_{min} + \frac{1}{2}(lr\_0 - lr\_{min}) \times \left(1 + \cos\left(\frac{\pi \times t}{T\_{max}}\right)\right)$$
*   **核心参数**：
    *   $lr\_0$：初始学习率
    *   $lr\_{min}$：最小学习率（通常为0）
    *   $T\_{max}$：半周期长度
*   **适用场景**：现代大模型训练（如Transformer、LLaMA）
*   **变化曲线**：**余弦波形**，平滑下降

## 4. 单周期策略 (OneCycleLR)
*   **数学表达式**（线性预热+余弦退火）：
    *   **预热阶段**（$t < T\_{warmup}$）：$$lr\_t = lr\_{min} + (lr\_{max} - lr\_{min}) \times \frac{t}{T\_{warmup}}$$
    *   **退火阶段**（$t \ge T\_{warmup}$）：$$lr\_t = lr\_{min} + \frac{1}{2}(lr\_{max} - lr\_{min}) \times \left(1 + \cos\left(\frac{\pi \times (t - T\_{warmup})}{T\_{total} - T\_{warmup}}\right)\right)$$
*   **核心参数**：
    *   $lr\_{max}$：峰值学习率
    *   $T\_{warmup}$：预热步数
    *   $T\_{total}$：总训练步数
*   **适用场景**：图像分类、目标检测等需要快速收敛的任务
*   **变化曲线**：**先升后降的三角形曲线**

## 5. 自适应调度 (ReduceLROnPlateau)
*   **数学表达式**：$$lr\_t = lr\_{t-1} \times \gamma$$（当验证指标连续$patience$轮未改善时触发）
*   **核心参数**：
    *   $\gamma$：衰减因子（通常为0.1）
    *   $patience$：容忍轮数
*   **适用场景**：需要根据验证结果动态调整的任务
*   **变化曲线**：**不规则阶梯状**，下降时机取决于模型表现

## 6. 线性衰减 (Linear Scheduler)
*   **数学表达式**：$$lr\_t = lr\_0 \times \left(1 - \frac{t}{T\_{total}}\right)$$
*   **核心参数**：
    *   $lr\_0$：初始学习率
    *   $T\_{total}$：总训练步数
*   **适用场景**：BERT等预训练模型的微调
*   **变化曲线**：**斜向下的直线**

## 7. 预热机制 (Warmup)
*   **数学表达式**（线性预热）：$$lr\_t = lr\_0 \times \frac{t}{T\_{warmup}}$$
*   **核心参数**：
    *   $lr\_0$：目标学习率
    *   $T\_{warmup}$：预热步数
*   **作用**：通常与其他调度器结合，在训练初期逐步提高学习率
*   **变化曲线**：**初始阶段的上坡曲线**

## 8. 总结对比

| 调度器                | 数学特征                                     | 变化曲线   | 推荐场景         |
| :-------------------- | :------------------------------------------- | :--------- | :--------------- |
| **StepLR**            | 幂函数 $f(t)=\gamma^{\lfloor t/s \rfloor}$   | 阶梯状     | 传统CNN          |
| **ExponentialLR**     | 指数函数 $f(t)=\gamma^{t}$                   | 指数下降   | 小模型实验       |
| **CosineAnnealingLR** | 余弦函数 $f(t)=\frac{1}{2}(1+\cos(\pi t/T))$ | 余弦波     | **大模型预训练** |
| **OneCycleLR**        | 分段函数                                     | 三角形     | **快速收敛任务** |
| **ReduceLROnPlateau** | 条件触发                                     | 不规则阶梯 | 自适应优化       |
| **Linear**            | 一次函数 $f(t)=1-t/T$                        | 直线       | BERT微调         |

### 8.1 **OneCycleLR** 和 **Warmup + CosineAnnealingLR**

**OneCycleLR** 和 **Warmup + CosineAnnealingLR** 在视觉上非常相似，都是“先升后降”的曲线。但它们之间存在着**核心的哲学差异**和**参数设置逻辑**的不同，这使得它们适用于完全不同的场景。

简单来说：
- **OneCycleLR**：激进派。为了追求“超级收敛”（Super-Convergence），它会使用极高的学习率（通常比正常值大10-100倍），并强制在训练结束前降到一个极低的值。
- **Warmup + CosineAnnealingLR**：温和派。为了训练大模型（如BERT、GPT）的稳定性，它通常只将学习率提升到一个适中的峰值（如初始值的2-5倍），然后平滑退火。

#### 8.1.1 核心差异对比

| 维度 | OneCycleLR (1Cycle) | Warmup + CosineAnnealing |
| :--- | :--- | :--- |
| **设计哲学** | **超级收敛** (Super-Convergence)<br>通过大学习率快速穿越参数空间 | **稳定训练**<br>防止训练初期梯度爆炸，后期精细调优 |
| **学习率峰值** | **极高** ($\text{max\\_lr}$)<br>通常是正常学习率的10-100倍 | **适中**<br>通常设定为初始学习率的2-5倍 |
| **终点值** | **极低** ($\text{max\\_lr} / \text{final\\_div\\_factor}$)<br>为了“冷却”模型，通常为峰值的万分之一 | **适中** ($\text{min\\_lr}$)<br>通常设定为峰值的十分之一或零 |
| **适用场景** | **小模型、快速实验**<br>（计算机视觉任务，如CIFAR-10） | **大模型预训练/微调**<br>（Transformer, BERT, GPT） |
| **动量策略** | **反向循环** (Inverse Cycle)<br>动量与学习率反向变化（高LR时低动量） | **通常固定**<br>或使用线性/余弦调整，但通常不反向 |

#### 8.1.2 数学公式详解

为了更清晰地展示区别，我们来看它们的数学定义。

##### OneCycleLR (PyTorch 实现)

OneCycleLR 的核心在于它定义了一个非常宽的范围（从极低到极高）。

- **初始学习率** ($lr\_{\text{init}}$): 由 $\text{max\\_lr} / \text{div\\_factor}$ 决定，$\text{div\\_factor}$ 通常设为 25。
- **峰值学习率** ($lr\_{\text{max}}$): 由用户设定的 $\text{max\\_lr}$。
- **终点学习率** ($lr\_{\text{final}}$): 由 $\text{max\\_lr} / \text{final\\_div\\_factor}$ 决定，$\text{final\\_div\\_factor}$ 通常设为 10000。

**分段公式**：
1. **Warmup 阶段** ($0 \le t < T\_{\text{warmup}}$):
   $$lr(t) = lr\_{\text{init}} + (lr\_{\text{max}} - lr\_{\text{init}}) \times \frac{t}{T\_{\text{warmup}}}$$
2. **Annealing 阶段** ($T\_{\text{warmup}} \le t \le T\_{\text{total}}$):
   $$lr(t) = lr\_{\text{final}} + (lr\_{\text{max}} - lr\_{\text{final}}) \times \left( 1 + \cos\left(\pi \times \frac{t - T\_{\text{warmup}}}{T\_{\text{total}} - T\_{\text{warmup}}}\right) \right) / 2$$

##### Warmup + CosineAnnealingLR

这里的参数通常基于初始学习率 ($lr\_0$) 进行设置，范围相对较小。

- **峰值学习率** ($lr\_{\text{max}}$): 通常设为 $lr\_0$ 或 $lr\_0 \times 2$。
- **终点学习率** ($lr\_{\text{min}}$): 通常设为 0 或 $lr\_0 \times 0.1$。

**分段公式**：
1. **Warmup 阶段** ($0 \le t < T\_{\text{warmup}}$):
   $$lr(t) = lr\_0 + (lr\_{\text{max}} - lr\_0) \times \frac{t}{T\_{\text{warmup}}}$$
2. **Cosine Annealing 阶段** ($T\_{\text{warmup}} \le t \le T\_{\text{total}}$):
   $$lr(t) = lr\_{\text{min}} + (lr\_{\text{max}} - lr\_{\text{min}}) \times \left( 1 + \cos\left(\pi \times \frac{t - T\_{\text{warmup}}}{T\_{\text{total}} - T\_{\text{warmup}}}\right) \right) / 2$$

#### 8.1.3 如何选择？

- **选 OneCycleLR**：如果你的目标是快速训练一个中等大小的模型（如ResNet50、EfficientNet）在标准数据集（如ImageNet）上，或者你正在进行快速实验（Fast AI），想要利用“超级收敛”效应。它的高学习率能帮你快速跳过局部极小值。
- **选 Warmup + CosineAnnealingLR**：如果你在训练大语言模型（LLaMA、GPT）或大视觉模型（ViT-Huge）。大模型对学习率极其敏感，过高的学习率会导致梯度爆炸或训练发散，因此需要一个更温和、更稳定的调度策略。

**总结**：虽然它们长得像“双胞胎”，但**OneCycleLR 是“短跑运动员”**（追求爆发力），而**Warmup + Cosine 是“马拉松选手”**（追求稳定性）。

从另一个角度来说：<span style="color: red; font-weight: bold;">一个参数保守的OneCycleLR，可以看作是Warmup + CosineAnnealing的一个高度集成且自动优化的“便捷版本”。它们最终的学习率曲线形态会非常相似。</span>

## 9. 选择建议

1.  **大模型训练**：推荐使用 **Warmup + CosineAnnealingLR** 或 **Linear衰减**
2.  **快速实验**：推荐使用 **OneCycleLR** 实现超级收敛
3.  **传统任务**：可考虑 **StepLR** 或 **ExponentialLR**
4.  **不确定衰减时机**：可尝试 **ReduceLROnPlateau**

**注**：所有调度器中，$t$ 通常表示当前训练步数或轮数，具体实现可能有差异。

## 10. 模型学习率设置参照

较大的 Transformer 模型（如 BERT、GPT、LLaMA、ViT 等）通常使用 **极小的学习率**，具体数值范围根据任务类型（预训练 vs 微调）和模型规模差异较大。

以下是基于主流实践和论文推荐的详细设置：

### 10.1 预训练 (Pretraining)

预训练通常使用相对较大的学习率，因为模型需要从零开始学习语言或视觉知识。

| 模型类型                     | 推荐学习率      | 说明                                                         |
| ---------------------------- | --------------- | ------------------------------------------------------------ |
| **BERT (Base/Large)**        | **1e-4**        | 原始 BERT 论文使用 1e-4，配合线性预热和线性衰减              |
| **大语言模型 (LLaMA, GPT)**  | **3e-4 ~ 5e-4** | 随着模型规模增大，常用 3e-4（如 LLaMA-7B）或 5e-4（如 GPT-3） |
| **Vision Transformer (ViT)** | **3e-4 ~ 5e-4** | 通常配合 AdamW 优化器，权重衰减较高（5e-2），学习率通常设为 3e-4 |

### 10.2 微调 (Fine-tuning / SFT)

微调时模型权重已经接近收敛，为防止破坏预训练知识，学习率必须大幅降低。

| 模型类型                   | 推荐学习率      | 说明                                                         |
| -------------------------- | --------------- | ------------------------------------------------------------ |
| **BERT (全量微调)**        | **2e-5 ~ 5e-5** | **业界标准范围**。BERT 官方推荐 5e-5，实际应用中 2e-5 更常用且稳定 |
| **LLaMA / GPT (全量微调)** | **1e-5 ~ 5e-5** | 通常设为 2e-5。模型越大（如 70B），学习率越小（如 1e-6 ~ 5e-6） |
| **LoRA / QLoRA**           | **1e-4 ~ 5e-4** | **LoRA 的学习率通常比全量微调大 10 倍**。因为 LoRA 只更新少量适配器参数，需要更大的步长来快速收敛 |

### 10.3 增量预训练 / 继续训练 (Continued Pretraining)

介于预训练和微调之间，目标是让模型适应新的领域数据而不遗忘旧知识。

| 场景           | 推荐学习率      | 说明                                                         |
| -------------- | --------------- | ------------------------------------------------------------ |
| **领域自适应** | **3e-5 ~ 5e-5** | 通常取原始预训练学习率的 10%~20%，例如原 LR 为 3e-4，则 CPT LR 设为 3e-5 |

### 10.4 学习率设置通用法则

1. **预训练 > 微调**：预训练学习率通常比微调大 **10 倍**。
2. **LoRA > 全量微调**：LoRA 的学习率通常比全量微调大 **10 倍**。
3. **模型越大，LR 越小**：70B 模型的微调学习率通常比 7B 模型小 **10 倍**（例如 1e-6 vs 1e-5）。
4. **必须有预热**：无论哪种场景，都必须配合 **Warmup** 策略（通常占总步数的 1%~10%），防止初期梯度震荡。