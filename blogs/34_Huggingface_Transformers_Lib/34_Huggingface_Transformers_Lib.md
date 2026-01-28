Hugging Face Transformers 是一个开源的 Python 库，它封装了大量基于 Transformer 架构的预训练模型，旨在简化自然语言处理（NLP）、多模态（图像、音频等）任务的开发流程。它支持 PyTorch、TensorFlow 和 JAX 三大深度学习框架，是当前大模型开发中最常用的工具之一。

<a href="https://huggingface.co/docs/transformers/v5.0.0/zh/index">官方教程</a>

### 1. 核心功能与特点

- **丰富的预训练模型**：库内集成了数千个预训练模型，覆盖 100 多种语言，包括 BERT、GPT、T5、LLaMA、ViT、Whisper 等，支持文本分类、问答、生成、翻译、图像分类、语音识别等多种任务。
- **统一的 API 接口**：通过 `AutoModel`、`AutoTokenizer`等“自动”类，可以根据模型名称自动匹配结构，实现模型无关的代码编写，降低了使用不同模型的学习成本。
- **开箱即用的 Pipeline**：`pipeline`接口封装了从文本预处理、模型推理到结果后处理的完整流程，仅需几行代码即可完成情感分析、文本生成等常见任务，非常适合快速原型开发。
- **完整的训练与微调工具链**：内置 `Trainer`类，封装了训练循环、评估、日志记录等复杂逻辑，支持分布式训练、混合精度等优化技术，方便用户使用领域数据对预训练模型进行微调。
- **强大的生态协同**：与 Hugging Face 生态中的其他工具无缝集成，如 `datasets`（数据集加载）、`evaluate`（模型评估）、`accelerate`（分布式训练）、`gradio`（快速演示）等，覆盖了从数据准备到模型部署的全流程。

### 2. 核心组件

1. **模型 (Models)**：提供 `AutoModel`及各类任务专用模型（如 `BertForSequenceClassification`），用于加载预训练模型并进行推理或微调。
2. **分词器 (Tokenizers)**：负责将原始文本转换为模型可接受的数字序列，支持 BPE、WordPiece 等主流分词算法。
3. **配置 (Configurations)**：用于定义和修改模型结构参数，如层数、隐藏层大小等。
4. **训练器 (Trainer)**：高阶 API，封装了训练循环、评估、保存等常用功能，简化了模型训练流程。

### 3. 快速入门示例

#### 1. 安装与验证

```bash
pip install transformers[torch] torch
```

验证安装结果：

```python
import torch
import transformers

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
```

#### 2. 使用 Pipeline 进行情感分析

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face Transformers!")
print(result)
```

```python
# 输出: [{'label': 'POSITIVE', 'score': 0.9971315860748291}]
```

#### 3. 使用 AutoModel 进行文本编码

```python
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Hello, Hugging Face!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

```python
# 输出：torch.Size([1, 7, 768])
```

### 4. 应用场景

- **学术研究**：快速复现基线模型，将精力集中于算法创新。
- **工业界**：低成本集成智能客服、内容审核、代码生成等 AI 能力。
- **个人学习**：学习现代 AI、构建个人项目、进入 AI 行业的入门工具。