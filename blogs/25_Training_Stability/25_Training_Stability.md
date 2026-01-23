<span class="image main">
<img class="main img-in-blog" style="max-width: 60%" src="./blogs/25_Training_Stability/stability.webp" alt="Stability" />
<i>稳定性, Stability by <a href="http://www.nyphotographic.com/">Nick Youngson</a> <a rel="license" href="https://creativecommons.org/licenses/by-sa/3.0/">CC BY-SA 3.0</a> <a href="http://pix4free.org/">Pix4free</a></i>
</span>

模型训练准确率的波动通常源于**随机性、超参数选择或训练过程的不稳定**。以下是系统性的解决方案，按优先级排序：

### **1. 控制随机性来源**

确保实验可复现，减少非本质波动：

- **固定随机种子**：设置Python、NumPy、随机库（如`random.seed()`）、深度学习框架（如`torch.manual_seed()`）的种子。
- **数据加载顺序**：使用`DataLoader`的`worker_init_fn`固定数据shuffle顺序。
- **CUDA确定性**：设置`torch.backends.cudnn.deterministic = True`和`torch.backends.cudnn.benchmark = False`（可能降低训练速度）。

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### **2. 优化数据加载与增强**

- **数据增强的随机性**：若增强策略过于激进（如随机裁剪、颜色抖动），可能引入噪声。可尝试： 减弱增强强度，或对验证集使用确定性的预处理。 检查数据集的样本均衡性，确保每个batch的分布稳定。
- **数据清洗**：检查验证集中是否有模糊标注或异常样本。

### **3. 调整超参数与训练策略**

- **学习率**：最关键的参数之一。 使用**学习率预热**（Warmup）：如前几轮从较小学习率线性增加，避免初期震荡。 采用**余弦退火**（Cosine Annealing）或**带重启的调度器**（CosineWarmRestarts），替代阶梯下降。 若波动仍大，可适当**降低初始学习率**（如减少30%-50%）。下面提供了一种自定义 Warmup 调度器的实现

  ```python
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.optim.lr_scheduler import _LRScheduler
  
  # 自定义线性Warmup调度器
  class LinearWarmupLR(_LRScheduler):
      def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, last_epoch=-1):
          self.warmup_epochs = torch.tensor(warmup_epochs, dtype=torch.float32)
          self.total_epochs = torch.tensor(total_epochs, dtype=torch.float32)
          self.base_lr = base_lr
          super().__init__(optimizer, last_epoch)
  
      def get_lr(self):
          epoch = self.last_epoch + 1  # last_epoch初始为-1，对应第0轮
          
          # 预热阶段：从0线性增加到base_lr
          if epoch <= self.warmup_epochs:
              lr = self.base_lr * (epoch / self.warmup_epochs)
          # 预热后：余弦退火衰减
          else:
              # 余弦退火公式：lr = base_lr * (1 + cos(pi*(epoch-warmup)/ (total-warmup))) / 2
              cos_factor = (1 + torch.cos(torch.pi * (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))) / 2
              lr = self.base_lr * cos_factor
          
          return [lr for _ in self.optimizer.param_groups]
  
  # 测试自定义调度器
  model = nn.Linear(10, 1)
  base_lr = 0.001
  optimizer = optim.SGD(model.parameters(), lr=base_lr)
  
  # 初始化自定义调度器
  warmup_epochs = 5
  total_epochs = 50
  scheduler = LinearWarmupLR(optimizer, warmup_epochs, total_epochs, base_lr)
  
  # 模拟训练
  for epoch in range(total_epochs):
      # 训练步骤（省略）
      optimizer.step()
      
      # 打印学习率
      current_lr = optimizer.param_groups[0]['lr']
      print(f"Epoch {epoch+1:2d}, Learning Rate: {current_lr:.6f}")
      
      # 更新学习率
      scheduler.step()
  ```

  **代码关键解释**：

  - `LinearWarmupLR`继承自`_LRScheduler`，必须实现`get_lr`方法，返回每轮的学习率列表。

  - 预热阶段：学习率从 0 开始，更适合对初始学习率敏感的场景。

  - 预热后：使用**余弦退火**策略，相比线性衰减更平缓，是业界常用的优化策略。

    <span class="image main">
    <img class="main img-in-blog" style="max-width: 60%" src="./blogs/25_Training_Stability/CosineAnnealingLR.webp" alt="Cosine Annealing Learning Rate" />
    <i>Cosine annealing learning rate. Image taken from <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html">Pytorch CosineAnnealingLR</a></i>
    </span>

  **实现要点**：

  - 预热轮数（`warmup_epochs`）通常设置为 5-10 轮，具体根据数据集大小调整。
  - `scheduler.step()`必须在`optimizer.step()`之后调用，否则学习率更新无效。

- **批量大小**：过小的batch可能导致梯度噪声大，过大会降低泛化能力。可尝试适度增大batch size（需同步调整学习率）。

- **优化器选择**： 使用**AdamW**（Adam with decoupled weight decay）替代原始Adam，更稳定且泛化更好。 可尝试**SGD with momentum**（如0.9）配合Nesterov动量，对许多任务更稳定。

- **梯度裁剪**：尤其是RNN或Transformer中，梯度爆炸可能导致波动（设置`clip_grad_norm_`通常为1.0-5.0）。

### **4. 模型与正则化**

- **模型容量**：若模型过于复杂（参数量远大于数据量），容易对数据噪声过拟合，导致每次训练收敛到不同局部极小值。可尝试： 增加正则化：**Dropout**（全连接层后）、**权重衰减**（AdamW中已包含）、**标签平滑**（Label Smoothing）。 降低模型复杂度（减少层数/通道数）。
- **批归一化（BatchNorm）**：在训练时使用移动平均统计，验证时固定。确保训练/验证模式切换正确，并检查batch size过小导致的统计量不稳定。

### **5. 验证集设计与评估**

- **验证集大小**：若验证集过小（如少于1000样本），评估本身可能有统计误差。扩大验证集或使用**交叉验证**。
- **评估指标**：对于分类任务，若类别不均衡，准确率可能掩盖波动。可同时监控**F1分数**或**AUC**。
- **多次运行取平均**：最终报告时，多次随机初始化训练（如5次），取平均准确率与标准差。

### **6. 高级技巧**

- **集成学习**：训练多个模型并集成预测，可显著稳定最终表现（但增加计算成本）。
- **早停策略**：基于验证集损失的平滑版本（如移动平均）决定早停，避免噪声干扰。
- **SWA（随机权重平均）**：在训练后期对多个检查点权重取平均，可提升稳定性与泛化。

### **快速诊断流程**

1. **固定随机种子**，观察波动是否仍存在。
2. 暂时**关闭数据增强**，检查波动是否减小。
3. 使用**更小的学习率**（如降为1/10）训练1-2个周期，观察损失是否平稳下降。
4. 监控**训练损失曲线**：若训练损失也大幅波动，可能是优化问题（如学习率过高）；若仅验证损失波动，可能是过拟合或验证集问题。

### **总结建议**

- 首先**控制所有随机种子**，确认波动是否显著减少。
- 若仍存在，**优先调整学习率策略**（加入Warmup + 余弦退火）。
- 同时**检查数据加载与增强**，确保训练/验证集预处理一致。
- 最后考虑**模型容量与正则化**的平衡。

通过上述步骤，通常可将验证准确率波动控制在1%以内，并获得更可靠的模型评估结果。