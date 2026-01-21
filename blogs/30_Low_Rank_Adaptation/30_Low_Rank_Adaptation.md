## 1. 背景与核心思想

### 1.1 问题背景
- 随着模型规模变得越来越大（如 GPT-3 有 1750 亿参数），传统的**全参数微调**（Full Fine-Tuning）变得极其昂贵：
  - 需要存储和训练整个模型的副本，每个任务一个副本，存储成本高。
  - 训练时间长，计算资源需求巨大。
  - 难以快速适配多个下游任务。

- 之前的参数高效微调方法（如 Adapter Layers、BitFit 等）虽然减少了参数量，但往往引入了额外的推理延迟，或性能不如全参数微调。

### 1.2 LoRA （Low-Rank Adaptation，低秩自适应）的核心直觉
LoRA 的核心思想基于一个**假设**：
> 在模型适配下游任务时，权重变化（$\Delta W$）具有“低秩”特性。  
> 也就是说，$\Delta W$ 可以用一个低维矩阵来近似表示，而不需要完整的高维矩阵。

换句话说，尽管模型权重矩阵 $W$ 很大（比如 $d \times d$），但任务特定的更新 $\Delta W$ 的“内在维度”其实很低，可以用两个小矩阵的乘积 $B A$ 来表示，其中 $B$ 和 $A$ 的秩 $r \ll d$。

## 2. LoRA 的原理

<span class="image main">
<img class="main img-in-blog" style="max-width: 60%" src="./blogs/30_Low_Rank_Adaptation/LoRA.webp" alt="Robotics Research Production Line" /><i>LoRA. Image taken from <a href="https://kim95175.tistory.com/28">Elsa Tech Blog</a></i></span>

### 2.1 数学形式

假设预训练模型的某一层的前向计算为：
$$
h = W x
$$
其中 $W \in \mathbb{R}^{d \times k}$ 是预训练权重，$x \in \mathbb{R}^{k}$ 是输入，$h \in \mathbb{R}^{d}$ 是输出。

在 LoRA 中，我们**冻结** $W$，不更新它的参数。我们引入一个低秩分解的适配矩阵 $\Delta W = B A$，其中：
- $A \in \mathbb{R}^{r \times k}$，$B \in \mathbb{R}^{d \times r}$，秩 $r \ll \min(d, k)$。
- 前向传播变为：
$$
h = W x + B A x
$$

训练时，只训练 $A$ 和 $B$，而 $W$ 保持不变。  
初始化时：

- $A$ 用随机高斯初始化（或 Kaiming 初始化）。
- $B$ 初始化为零，这样训练开始时 $\Delta W = 0$，不影响原始模型输出。

### 2.2 为什么是低秩？
- 理论支持：Aghajanyan 等人在论文《[Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/pdf/2012.13255)》中发现，预训练模型在下游任务上微调时，其“内在维度”很小，意味着可以用很少的参数捕捉任务所需的变化。
- 工程优势：参数量从 $d \times k$ 降到 $r \times (d + k)$，当 $r \ll d, k$ 时，参数量大幅减少。

### 2.3 可合并性（Inference-time Merging）
训练完成后，我们可以将 $B A$ 加到 $W$ 上：
$$
W_{\text{new}} = W + B A
$$
这样在推理时，不需要额外计算 $B A x$，**不增加任何推理延迟**，和原始模型的计算复杂度一致。  
这是 LoRA 相比于 Adapter 的最大优势之一。

## 3. LoRA 的应用细节

### 3.1 应用到 Transformer 的哪些层？
通常，LoRA 只应用到 Transformer 的**注意力机制**的权重矩阵上（Q、K、V、O 的投影矩阵），因为：
- 这些层通常包含模型的大部分语义和任务相关能力。
- 实验表明，只适配这些层就能达到很好的效果，且参数量最小。

具体来说，对于每个注意力头的 Q、K、V、O 四个投影矩阵，分别加一组 $B A$。  
有时也会应用到 FFN 层的矩阵，但效果提升不如注意力层明显。

### 3.2 超参数选择
- **秩 $r$**：通常取 4、8、16、64 等。对于大多数任务，$r=8$ 已经足够，更大的 $r$ 对性能提升有限。
- $\alpha$（缩放系数）：在合并权重时，通常用 $\frac{\alpha}{r}$ 对 $B A$ 进行缩放，类似于学习率缩放。$\alpha$ 一般与 $r$ 同量级，比如 $r=8, \alpha=16$。
- **Dropout**：在 LoRA 的适配路径（即 $B A x$）中可以加 Dropout 防止过拟合。

## 4. LoRA 的优势

1. **参数高效**：训练参数减少 10,000 倍以上（例如 GPT-3 175B 只需训练 0.01% 的参数）。
2. **无推理延迟**：权重可合并，推理时和原始模型一样快。
3. **内存效率高**：只需存储 LoRA 权重，多个任务可共用同一基础模型，只需切换小的 LoRA 模块。
4. **兼容性**：可与其它方法结合，如前缀微调（Prefix Tuning）、Adapter 等。
5. **易于部署**：LoRA 权重很小，可以像插件一样分发和加载。

## 5. 实际应用场景

### 5.1 大语言模型指令微调
- 例如：用 LoRA 微调 LLaMA、ChatGLM 等模型，使其遵循指令、进行对话。
- 代表性项目：Alpaca-LoRA、Chinese-LLaMA-Alpaca 等。

### 5.2 多任务适配
- 一个基础模型 + 多个 LoRA 权重，分别适配不同任务（翻译、摘要、问答等）。
- 推理时按需加载对应的 LoRA 权重。

### 5.3 跨模态适配
- 在视觉-语言模型（如 BLIP、Flamingo）中，用 LoRA 适配视觉编码器或跨模态注意力层。

### 5.4 领域适配
- 将通用模型快速适配到医疗、法律、金融等垂直领域，只需少量领域数据 + LoRA 微调。

## 6. 变体与扩展

- **QLoRA**：结合量化（4-bit 量化），进一步减少内存占用，使得在单张消费级 GPU 上微调 65B 模型成为可能。
- **LoRA+**：对 $A$ 和 $B$ 使用不同的学习率，加速收敛。
- **DoRA（Weight-Decomposed Low-Rank Adaptation）**：将权重分解为幅度和方向分量，用 LoRA 适配方向，效果更接近全参数微调。
- **VeRA（Vector-based Random Matrix Adaptation）**：冻结随机矩阵，只训练小的缩放向量，进一步减少参数量。

## 7. 总结

LoRA 是一个简单、高效、实用的参数高效微调方法，它利用**低秩分解**的思想，在保持模型性能的同时，大幅降低了训练成本和部署复杂度。  
它已成为大模型微调的事实标准之一，并被集成到 Hugging Face PEFT（Parameter-Efficient Fine-Tuning）库中，方便用户快速使用。
