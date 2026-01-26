Transformer是自然语言处理领域的革命性架构，理解其实现细节对掌握现代大语言模型至关重要。Transformer 的核心是**自注意力机制**，它通过计算序列中所有元素之间的关系，将输入序列（如句子）**转换**为一种新的、富含上下文信息的表示。这种“转换”能力体现在：

- **语义转换**：将原始词向量转换为包含全局上下文信息的表示。
- **任务转换**：将一种序列（如英文）转换为另一种序列（如中文），适用于机器翻译、摘要生成等任务。

> **命名由来**：Jakob Uszkoreit 的创意
>
> “Transformer”这个名字由论文作者之一 Jakob Uszkoreit 提出，主要基于以下两点：
>
> 1. **功能契合**：该词准确描述了模型“**转换**”数据表示的核心功能。
> 2. **个人喜好**：Uszkoreit 本人是《变形金刚》（Transformers）的粉丝，认为这个名字“听起来很酷”，因此将其用于模型命名。

<span class="image main">
<img class="main img-in-blog" style="max-width: 50%" src="./blogs/32_Building_a_Transformer_from_Scratch/Building_a_Transformer.webp" alt="Building a transformer from scratch" />
<i>Building a transformer from scratch.</i>
</span>

本文将从零开始实现一个完整的Transformer模型。

## 0. 项目目标及可行性分析

项目以训练一个**检索增强生成模型 (RAG) **或**问答模型 (QA)**作为目标。对于这个目标我们可以选择Encoder-Decoder (T5风格)架构，Encoder读入 `title`，Decoder生成 `text`。

项目以个人电脑的硬件配置（8GB显存 + 32GB内存）训练一个**微型 (Tiny)** 或**极小 (Mini)** 规模的 Encoder-Decoder Transformer 模型。

虽然你无法训练出像 T5-large 或 mT5 那样拥有数亿参数的大模型，但可以训练一个参数在 **10M 到 50M（1000万到5000万）** 之间的模型，这在个人电脑上绰绰有余，并且完全能够完成“给定标题，生成摘要”的问答任务。

为了在 8GB 显存下流畅训练，我们需要严格控制模型大小。以下是一个推荐的配置（以 **T5-Small** 为蓝本进行缩小）：

- **模型架构**: Encoder-Decoder (标准的 Transformer 结构，最适合做 Text-to-Text 的任务，如问答、摘要)
- **参数规模**: 约 **20M (2000万)** 参数
- **关键超参数**: `d_model`: 256 或 384 (模型的隐藏层维度，原始T5-small是512，我们砍半或取3/4) `n_head`: 4 或 8 (注意力头数) `num_layers`: 4 (Encoder和Decoder都只有4层，足够深来捕捉语义，又足够浅来节省显存) `d_ff`: 1024 (前馈网络维度，通常是 d_model 的 4 倍)
- **词表大小 (Vocab Size)**: 约 30,000 (使用 SentencePiece 在中文维基数据上训练一个专属的词表，或者复用 Hugging Face 上已有的中文T5词表)

这个尺寸的模型在训练时，即使 Batch Size 设为 16 或 32，显存占用也完全可以控制在 6GB 以内，给你留出了足够的显存余量用于梯度计算和优化器状态。

### 本项目中必需的python包

```bash
pip install datasets	# Hugging Face datasets
pip install tqdm
pip install sentencepiece	# 用于训练中文词表

# 避免安装失败,先升级pip
pip install --upgrade pip
pip install --upgrade pip setuptools
nvcc --version	# 查看当前安装的cuda版本
# 按照运行环境安装对应版本的pytorch
# https://pytorch.org/get-started/previous-versions/
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

## 1. 数据集准备

### 1.1 准备训练语料 (Corpus Preparation)

本文中对Transformer的实现以[中文维基百科（Wikipedia 中文版）离线数据集](https://huggingface.co/datasets/fjcanyue/wikipedia-zh-cn/tree/main)作为训练数据。

```python
from datasets import load_dataset

# 加载数据集（自动下载）
dataset = load_dataset("fjcanyue/wikipedia-zh-cn", data_files="wikipedia-zh-cn-20250901.json", split="train")

# 查看数据集信息
print(f"数据集类型: {type(dataset)}")
print(f"数据集大小: {len(dataset)}")
print(f"数据集特征: {dataset.features}")

# 查看样例
print("打印样例: ")
print(dataset[0])

# 尝试查看是否有预定义的分割
try:
    # 尝试加载验证集（如果存在的话）
    val_dataset = load_dataset("fjcanyue/wikipedia-zh-cn",
                              data_files="wikipedia-zh-cn-20250901.json",
                              split="validation")
    print("验证集存在，大小:", len(val_dataset))
except:
    print("验证集不存在，需要手动划分")
```
代码输出如下：

```text
数据集类型: <class 'datasets.arrow_dataset.Dataset'>
数据集大小: 1467344
数据集特征: {'id': Value('string'), 'title': Value('string'), 'tags': Value('string'), 'text': Value('string')}
打印样例: 
{'id': '13', 'title': '数学', 'tags': '数学,形式科学,主要话题条目', 'text': '数学是研究数量、结构以及空间等概念及其变化的一门学科，属于形式科学的一种。...'}
验证集不存在，需要手动划分
```

使用 SentencePiece 训练中文词表是一个不错的选择，特别是对于维基百科这种包含大量专业术语的语料。SentencePiece 可以直接从原始文本中学习子词划分，无需依赖预先分词。

SentencePiece 需要一个纯文本文件作为输入。我们需要从结构化数据中提取出所有文本内容，并将其合并到一个 `.txt`文件中。

```python
from datasets import load_dataset
from tqdm import tqdm
import os  # 导入os模块用于处理目录

# 加载数据集（自动下载）
print("正在加载数据集...")
dataset = load_dataset("fjcanyue/wikipedia-zh-cn", data_files="wikipedia-zh-cn-20250901.json", split="train")

# 提取所有文本：通常我们会使用 'title' 和 'text' 字段
# 为了充分利用信息，我们可以将 title 和 text 拼接起来
print("正在处理文本数据...")
corpus_lines = []
for item in tqdm(dataset, desc="处理进度"):
    # 拼接 title 和 text
    full_text = item['title'] + " " + item['text']
    corpus_lines.append(full_text)

# 3. 将语料写入一个纯文本文件，每行一个文档/句子
corpus_file = './dataset/wiki_corpus.txt'

# 确保目录存在
os.makedirs(os.path.dirname(corpus_file), exist_ok=True)

print("正在写入文件...")
with open(corpus_file, 'w', encoding='utf-8') as f:
    for line in tqdm(corpus_lines, desc="写入进度"):
        f.write(line + '\n')  # 每个条目占一行

print(f"语料文件已生成: {corpus_file}, 共 {len(corpus_lines)} 条文本。")
```

### 1.2 训练词表 (Training the Vocabulary)

```python
import sentencepiece as spm

# 综合符号列表
user_defined_symbols = [
    '。', '，', '、', '！', '？', '；', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '《', '》', '『', '』', '「', '」', '〔', '〕', '…', '—', '～', '·', '．',
    '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '-', '_', '=', '+', '@', '#', '$', '%', '^', '&', '*',
    '×', '÷', '±', '≠', '≈', '≡', '≤', '≥', '≦', '≧', '≪', '≫', '∝', '∞',
    '∈', '∉', '⊂', '⊃', '⊆', '⊇', '∪', '∩', '∅', '∀', '∃', '¬', '∧', '∨', '⊕', '⊗', '⊥', '∥', '∠',
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
    'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
    '∑', '∏', '∫', '∬', '∭', '∮', '∯', '∰', '∇', '∂', '∆', '√', '∛', '∜', '‰', '‱',
    '→', '←', '↑', '↓', '↔', '↕', '⇒', '⇐', '⇔', '↦', '↣', '↪', '↩', '⇀', '⇁',
    '￥', '＄', '€', '£', '¥', '₩', '₽', '₹', '¢', '¤',
    '°', '℃', '℉', '㎜', '㎝', '㎞', '㎡', '㎥', '㎏', '㏄', '㏑', '㏒', 'µ', 'Å', 'Ω',
    '♂', '♀', '⚥', '☿', '♁', '♃', '♄', '♅', '♆', '♇', '♈', '♉', '♊', '♋', '♌', '♍', '♎', '♏', '♐', '♑', '♒', '♓',
    '■', '□', '▲', '△', '▼', '▽', '◆', '◇', '●', '○', '★', '☆', '♠', '♣', '♥', '♦',
    '零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾', '佰', '仟', '萬', '億',
    '甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸', '子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥',
    '©', '®', '™', '§', '¶', '†', '‡', '※', '№', '♯',
]

# 定义训练参数
corpus_file = './dataset/wiki_corpus.txt'	# 语料文件
model_prefix = 'zh_wiki_spm'  # 模型输出文件的前缀
vocab_size = 30000            # 词表大小，根据你的需求调整，3万是一个常用起点
character_coverage = 0.9995   # 字符覆盖率，对于中文建议高一些

# 训练模型
model_prefix = 'zh_wiki_spm_comprehensive'
vocab_size = 30000
character_coverage = 0.9995  # 对中文很重要

# 训练模型
spm.SentencePieceTrainer.train(
    input=corpus_file,        # 输入语料文件
    model_prefix=model_prefix,# 输出模型前缀
    vocab_size=vocab_size,    # 词表大小
    model_type='bpe',         # 模型类型：'bpe', 'unigram', 'char', 'word'
    # 对于中文，BPE 或 Unigram 都是常用选择。Unigram 通常效果更好，BPE 更简单。
    # 这里我们使用 'bpe'
    character_coverage=character_coverage, # 字符覆盖率
    pad_id=0,                 # 定义特殊Token的ID
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='[PAD]',        # 定义特殊Token的字符串表示
    unk_piece='[UNK]',
    bos_piece='[BOS]',
    eos_piece='[EOS]',
    # 控制分词行为的重要参数
    # 将数字、标点也视为独立Token，而不是与字合并
    split_digits=True,        # 将数字拆分为单个数字
    user_defined_symbols=user_defined_symbols,  # 使用我们定义的全面符号列表
)

print("模型训练完成！")
```

**关键参数解释**:

- **`model_type`**: 推荐使用 `'bpe'`(Byte Pair Encoding) 或 `'unigram'`。BPE 更常见，训练快；Unigram 在某些任务上效果更好，但训练稍慢。
- **`character_coverage`**: 对于像中文这样字符集很大的语言，设置一个较高的覆盖率（如0.9995）可以确保模型能处理绝大多数字符。如果遇到未知字符，会回退到 `[UNK]`。
- **`split_digits`**: 将数字（如“123”）拆分为单个数字（“1”, “2”, “3”），这有助于模型处理数字，而不是将它们作为一个巨大的、罕见的词。
- **`input_sentence_size`**: 这是最关键的参数。它限制了用于训练的最大句子数量。通常**不需要**使用全部数据集来训练一个通用的BPE词表。**100万到500万行文本**就足以训练出高质量的词表。
- **`num_threads`**: 增加线程数，充分利用你的 CPU 核心。
- **`user_defined_symbols`**: 为了确保训练出的词表能完美覆盖中文维基百科中的各种符号（包括数学、科学、标点、货币等），我们需要定义一个**更全面、更系统的符号列表**。
  - **防止碎片化**: 如果没有将这些符号定义为 `user_defined_symbols`，SentencePiece 可能会尝试将它们与相邻的汉字或数字合并，产生奇怪的子词（如 `“数`或 `。的`），这会严重影响模型对文本结构的理解。
  - **保留语义**: 数学符号如 `∑`、`∫`在维基百科中具有精确的数学含义，必须作为独立的 Token 存在。
  - **跨语言兼容**: 确保英文标点和中文标点都能被正确处理。

训练过程中打印的部分词表如下

```text
bpe_model_trainer.cc(159) LOG(INFO) Updating active symbols. max_freq=923 min_freq=15
bpe_model_trainer.cc(268) LOG(INFO) Added: freq=921 size=11320 all=12506638 active=634944 piece=欢乐
bpe_model_trainer.cc(268) LOG(INFO) Added: freq=919 size=11340 all=12514818 active=643124 piece=控制了
bpe_model_trainer.cc(268) LOG(INFO) Added: freq=917 size=11360 all=12524315 active=652621 piece=招待
bpe_model_trainer.cc(268) LOG(INFO) Added: freq=915 size=11380 all=12534708 active=663014 piece=日到
bpe_model_trainer.cc(268) LOG(INFO) Added: freq=913 size=11400 all=12544452 active=672758 piece=记述
```

### 1.3 验证和使用词表

测试词表对句子的编码和解码。

```python
import sentencepiece as spm

# 加载训练好的模型
model_prefix = 'zh_wiki_spm_comprehensive'
sp = spm.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")

# 测试句子
test_sentence = "数学是研究数量、结构以及空间等概念的一门学科。"

# 编码：将句子转换为Token ID列表
ids = sp.encode_as_ids(test_sentence)
print(f"Token IDs: {ids}")

# 编码：将句子转换为Token 字符串列表
pieces = sp.encode_as_pieces(test_sentence)
print(f"Tokens: {pieces}")

# 解码：将Token ID列表转换回句子
decoded = sp.decode_ids(ids)
print(f"Decoded: {decoded}")

# 打印词表大小
vocab_size = sp.get_piece_size()
print(f"词表大小: {vocab_size}")

# 查看词表
vocab_list = []
for i in range(sp.get_piece_size()):
    vocab_list.append((i, sp.id_to_piece(i), sp.get_score(i)))

# 打印前50个词条
print("\n词表前50个条目:")
for i, piece, score in vocab_list[:50]:
    print(f"{i}\t{piece}\t{score}")
```
测试结果如下
```text
Token IDs: [23444, 1916, 23457, 324, 685, 6, 596, 308, 920, 23584, 1315, 312, 23762, 3802, 4]
Tokens: ['▁', '数学', '是', '研究', '数量', '、', '结构', '以及', '空间', '等', '概念', '的一', '门', '学科', '。']
Decoded: 数学是研究数量、结构以及空间等概念的一门学科。
词表大小: 30000

词表前50个条目:
0	[PAD]	0.0
1	[UNK]	0.0
2	[BOS]	0.0
3	[EOS]	0.0
```

## Transformer 实现

### 2.1 环境准备与依赖导入

首先导入必要的库并设置随机种子以确保结果可复现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import math
from tqdm import tqdm
from datasets import load_dataset
import os
from dataclasses import dataclass

# 固定随机种子
torch.manual_seed(42)

# 模型配置类
@dataclass
class TransformerConfig:
    # 模型配置
    d_model: int = 256      # 模型维度
    n_heads: int = 8        # 注意力头数
    d_ff: int = 2048        # 前馈网络维度
    n_layers: int = 4       # 编码器/解码器层数
    max_seq_len: int = 1000 # 最大序列长度
    dropout: float = 0.1    # dropout比例
    
    # 训练配置
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    
    # 词表配置
    vocab_size: int = 30000  # 词表大小
    sp_model_path: str = "zh_wiki_spm_comprehensive.model"	# 词表路径
```

## 2. 位置编码实现

Transformer没有循环结构，需要显式添加位置信息：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        pe = torch.zeros(config.max_seq_len, config.d_model)

        # 生成位置序列
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算div_term：10000^(2i/d_model)
        # 根据对数恒等: exp(a * log(b)) = b^a，我们可以进行变换
        # exp( -(2i * log(10000)) / d_model ) = 10000^(-2i / d_model)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2, dtype=torch.float) *
            - (math.log(10000.0) / config.d_model)
        )

        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度并注册为不参与梯度更新的缓冲区
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

**关键细节解释**:

1. **`div_term`**: 为什么不直接写成 `1 / (10000 ** (2*i / d_model))`与原始论文中的公式一致？

   * **数值稳定性**：

     - 当 `d_model`较大时，`10000^(2i/d_model)`可能会产生非常大的数值，导致数值计算上的问题（如溢出）。

     - 使用 `exp`和 `log`的变换，可以将幂运算转化为乘法和指数运算，这在数值上更稳定。

   * **计算效率**：
     - 在底层实现上，`torch.exp`和向量乘法通常经过高度优化，可能比逐个计算大数的幂运算更快。

2. 为什么要加 `Dropout`?

   这个 Dropout 是施加在**“词嵌入”和“位置编码”的和**之上的。这意味着：

   - 词嵌入向量（`x`）在加上位置编码后，**立即**被进行了 Dropout 正则化。
   - 这个 Dropout 会随机将“词嵌入+位置编码”这个联合表示中的一些元素置零。

   对“内容+位置”的联合表示进行 Dropout，能同时扰动两种信息，迫使模型不依赖于特定的内容或特定的位置，从而学习更鲁棒的特征。这与分别对两者做 Dropout 的效果是相似的。

3. **`register_buffer()`**函数的作用

   `register_buffer()`是 PyTorch 的 `nn.Module` 类中一个非常重要且实用的方法。它的作用是将张量（Tensor）注册为模块的“缓冲区”（Buffer）。

   要理解它的作用，我们需要先区分模型中的两种数据：**参数（Parameters）** 和 **缓冲区（Buffers）**。

   1. **核心概念**：参数（Parameters） vs 缓冲区（Buffers）

   - **参数（Parameters）**: 这些是模型需要**学习和更新**的权重。例如，线性层（`nn.Linear`）的 `weight`和 `bias`。 它们通过反向传播和优化器（如SGD, Adam）进行梯度下降更新。 使用 `self.parameter = nn.Parameter(tensor)`来定义。
   - **缓冲区（Buffers）**: 这些是模型的一部分，需要被保存和加载，但**不需要进行梯度下降更新**。 它们是模型的“状态”，但不是“可训练权重”。 使用 `self.register_buffer('buffer_name', tensor)`来定义。

   2. **register_buffer** 的具体作用

      ``` python
      pe = pe.unsqueeze(0)  # 增加batch维度: [1, max_seq_len, d_model]
      self.register_buffer('pe', pe)  # 注册为缓冲区
      ```

      这行代码做了以下几件关键的事情：

      1. 将张量标记为模型的一部分

         `register_buffer`告诉 PyTorch：**“这个 `pe`张量是我这个模型的一部分，请把它和模型一起保存，一起加载。”**

         - **保存模型时 (`torch.save(model.state_dict(), 'model.pt')`)**: 缓冲区 `pe`会和所有参数（`weight`, `bias`等）一起被保存到 `.pt`文件中。

         - **加载模型时 (`model.load_state_dict(...)`)**: 缓冲区 `pe`也会被正确地加载回来。

         **对比**：如果你只是简单地写成 `self.pe = pe`（作为一个普通的实例属性），那么这个张量**不会**被自动包含在 `state_dict`中。当你保存模型后再加载，`self.pe`会是 `None`，导致模型出错。

      2. 不计算梯度，不进行更新

         位置编码 `pe`是预先计算好的固定值（正弦和余弦函数），**不应该**被梯度下降算法更新。它只是一个固定的位置偏置。

         - 如果使用 `nn.Parameter(pe)`，PyTorch 会为 `pe`计算梯度，优化器会尝试更新它，这完全是浪费计算资源，而且会破坏预定义的位置信息。

         - 使用 `register_buffer`注册后，`pe`会被自动排除在梯度计算之外，优化器也会忽略它。

      3. 自动设备移动（Device Agnostic）

         这是一个非常方便的特性。当你将模型移动到 GPU 或 CPU 时：

         ```python
         model = model.to('cuda')
         # 或者
         model = model.cuda()
         ```

         所有通过 `register_buffer`注册的张量（如 `pe`）**会自动**被移动到与模型参数相同的设备上。你不需要手动管理 `pe`的设备位置。

## 3. 缩放点积注意力

这是Transformer的**核心机制**，详细解析，参考 <a href="index.html?part=blogs&id=8">Attention Mechanism </a>.

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        # query, key, value形状: [batch_size, n_heads, seq_len, d_k]
        d_k = query.size(-1)
        
        # 计算注意力分数: Q*K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（防止解码器看到未来信息）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, value)
        return output, attn_weights
```

## 4. 多头注意力机制

让模型能够同时关注不同表示子空间的信息，详细解析，参考 <a href="index.html?part=blogs&id=24">Multi-Head Attention Mechanism</a>.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        assert config.d_model % config.n_heads == 0, "d_model必须能被n_heads整除"
        
        # 线性变换层
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        self.attention = ScaledDotProductAttention(config.dropout)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性投影并分割成多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        x, attn_weights = self.attention(Q, K, V, mask)
        
        # 合并多头
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出投影
        return self.w_o(x)
```

**关键细节解释**:

**`.contiguous()`**: 这是一个**关键但容易被忽略的操作**，它解决了 PyTorch 张量在内存布局上的一个潜在问题，确保了后续的 `view()`操作能够正确执行。

1. **问题的根源**：`transpose()`操作改变了内存布局

   - **初始形状**：假设 `Q`经过 `view`后的形状是 `[batch_size, seq_len, n_heads, d_k]`。这是一个在内存中**连续存储**的张量。
   - **转置后**：`transpose(1, 2)`将第1维（`seq_len`）和第2维（`n_heads`）交换。现在的形状是 `[batch_size, n_heads, seq_len, d_k]`。

   **关键点**：`transpose()`操作是**视图操作 (View Operation)**。它不会实际移动数据在内存中的位置，只会改变访问数据的**步长 (Stride)**。转置后的张量在内存中变成了**非连续 (non-contiguous)** 存储。

2. 为什么需要 `contiguous()`？

   接下来，代码需要将多头合并回去：

   ```python
   x = x.transpose(1, 2)  # 形状变回 [batch_size, seq_len, n_heads, d_k]
   x = x.contiguous()     # 确保内存布局连续
   x = x.view(batch_size, -1, self.d_model)  # 合并最后两维: [batch_size, seq_len, d_model]
   ```

   * view()的限制：PyTorch 的 view()函数要求底层的张量在内存中必须是**连续存储 (contiguous) 的**。它只能作用于“看起来像”一维数组的张量。
   * 非连续张量的问题：由于之前的 transpose()，张量在内存中的排列顺序是“**跳跃**”的。如果你试图直接对一个非连续张量调用 view()，PyTorch 会报错：RuntimeError: view size is not compatible with input tensor's size and stride...。

3. `.contiguous()`的**作用**

   `.contiguous()`函数的作用是：创建一个新的、在物理内存中连续存储的张量，其数据与原始张量相同，但内存布局是连续的。

   * **调用前**：`x`是转置后的视图，数据在内存中是“交错”的，步长不规则。
   * **调用后**：`x.contiguous()`返回一个新的张量，数据被实际复制（或重新排列）成内存中连续的一块。这个新张量的步长是规则的，可以被 `view()` 安全地操作。

4. **替代方案**：`reshape()`

   在现代 PyTorch 中，你可以使用 `.reshape()`函数来替代 `.view()`。`.reshape()`更智能，它会**自动判断**是否需要复制数据：

   - 如果张量是连续的，`.reshape()`的行为和 `.view()`完全一样，不复制数据。
   - 如果张量是非连续的，`.reshape()`会自动调用 `.contiguous()`来创建连续副本，然后进行形状改变。

   因此，代码可以简化为：

   ```python
   # 使用 reshape 替代 contiguous + view
   x = x.transpose(1, 2).reshape(batch_size, -1, self.d_model)
   ```

   这行代码与原始代码是等价的，但更加简洁和安全。

## 5. 前馈网络

对每个位置独立应用相同的**全连接网络**：

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # 两层线性变换 + ReLU激活
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

## 6. 编码器层

包含多头自注意力和前馈网络，都有残差连接和层归一化：

```python
class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## 7. 解码器层

比编码器层多一个**编码器-解码器**注意力机制，详细解析，参考 <a href="index.html?part=blogs&id=21">Cross Attention Mechanism</a>.

```python
class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 掩码自注意力（防止看到未来信息）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力, Q来自解码器, K V来自编码器
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

## 8. 完整编码器和解码器

```python
class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, src, src_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return self.norm(x)
```

**关键细节解释**:

`nn.Embedding`是 PyTorch 中用于词嵌入（Word Embedding） 的核心模块。它的功能非常简单但至关重要：将一个整数索引（通常代表一个词或一个Token）映射为一个固定大小的密集向量（Dense Vector）。

1. **核心功能**：查表（Look-up Table）

   你可以把 `nn.Embedding`想象成一个**巨大的查找表（Look-up Table）** 或**字典**：

   - **键（Key）**：整数索引（例如 0, 1, 2, ..., vocab_size-1）。每个索引对应词表中的一个词（Token）。

   - **值（Value）**：固定长度的浮点数向量（例如长度为 256, 512 等）。这就是所谓的“词向量”或“嵌入向量”。

2. **结构解析**

   `nn.Embedding`的结构非常简单，它本质上是一个**可训练的**权重矩阵。在模型训练过程中，通过梯度下降和反向传播，这个矩阵中的每一个数值都会被更新。

   - **初始状态**：权重通常是随机初始化的（例如，从均值为0，方差为1的正态分布中采样）。
   - **训练过程**：模型通过观察大量的文本数据，学习调整每个词对应的向量。语义相近的词（如“猫”和“狗”）的向量在空间中的位置会逐渐变得接近。

   当创建 `nn.Embedding`时，需要指定两个关键参数：

   ```python
   embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)
   ```

   - **`num_embeddings`**：词表的大小。例如，如果你的词表有 10000 个不同的词，这里就设为 10000。索引的有效范围是 `0`到 `9999`。
   - **`embedding_dim`**：每个嵌入向量的维度。这个值通常是你模型配置中的 `d_model`（如 256, 512, 768）。维度越高，能编码的词义信息越丰富，但参数也越多。

3. **处理填充索引**（Padding）

   在处理变长序列时，我们通常会用 `0`来填充（Pad）短句子的末尾。`nn.Embedding`会为索引 `0`也分配一个向量。在训练过程中，这个“填充”位置的向量也会被更新，但通常我们会在计算注意力时通过掩码（Mask）来忽略这些位置。

## 9. 完整的Transformer模型

```python
class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        # 参数初始化
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器前向传播
        enc_output = self.encoder(src, src_mask)
        
        # 解码器前向传播
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.output_layer(dec_output)
        return output
    
    def generate_mask(self, src, tgt):
        # 源序列掩码（处理padding）
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # 目标序列掩码（防止解码器看到未来信息）
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        tgt_mask = tgt_mask & subsequent_mask.to(tgt.device)
        
        return src_mask, tgt_mask
```

**关键细节解释**:

1. **`nn.init.xavier_uniform_()`**: 这是 PyTorch 中一个非常重要的**权重初始化函数**。它的作用是根据 **Xavier (Glorot) 初始化** 方法，使用均匀分布来初始化神经网络层的权重。

   Xavier 初始化的核心思想是：**在初始化时，确保每一层输出的方差等于其输入的方差**。这样可以使得信号在前向传播过程中保持稳定的幅度，同时梯度在反向传播过程中也能保持稳定。

   **为什么在 Transformer 中很重要？**

   Transformer 模型通常很深（有很多层），并且包含自注意力机制和前馈网络。正确的初始化对于模型的收敛至关重要。

   - **注意力层的线性投影**：`W_Q`, `W_K`, `W_V`, `W_O`都是线性层，需要 Xavier 初始化。
   - **前馈网络**：两个线性层 `W1`和 `W2`也需要正确初始化。
   - **嵌入层**：虽然 `nn.Embedding`的输入是 one-hot 向量（方差为1），但 Xavier 初始化也能提供一个合理的起始点。

2. **`torch.tril`**函数的作用: `torch.tril`是 PyTorch 中一个非常实用的函数，它的作用是返回一个矩阵的下三角部分（lower triangular part），并将上三角部分设置为0。

   **函数定义**: 

   ```python
   torch.tril(input, diagonal=0) → Tensor
   ```

   * input(Tensor): 输入张量，至少是2维的
   * diagonal(int, optional): 对角线偏移量，默认为0

   **核心功能**: 

   `torch.tril`会返回一个与输入张量相同形状的新张量，其中：

   * **下三角部分（包括对角线）** 保持原值
   * **上三角部分** 设置为0

## 10. 训练模型

定义数据集

```python
class WikipediaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 构建输入：使用title作为源序列
        source_text = item['title']

        # 构建目标：使用text作为目标序列（用于文本生成任务）
        target_text = item['text']

        # 使用SentencePiece进行编码
        source_ids = self.tokenizer.encode_as_ids(source_text)
        target_ids = self.tokenizer.encode_as_ids(target_text)

        # 截断或填充到固定长度
        source_ids = source_ids[:self.max_length - 1] + [self.tokenizer.eos_id()]
        target_ids = target_ids[:self.max_length - 1] + [self.tokenizer.eos_id()]
		
        # 保存 padding 前的源序列长度
        len_source_ids = len(source_ids)

        # 填充到相同长度
        source_padding = [self.tokenizer.pad_id()] * (self.max_length - len_source_ids)
        target_padding = [self.tokenizer.pad_id()] * (self.max_length - len(target_ids))

        # source_ids 此处长度发生变化
        source_ids.extend(source_padding)
        target_ids.extend(target_padding)

        return {
            'source_ids': torch.LongTensor(source_ids),
            'target_ids': torch.LongTensor(target_ids),
            'source_mask': torch.LongTensor([1] * len_source_ids + [0] * len(source_padding))
        }

```

下面是完整的的训练流程：

```python
def train_transformer():
    # 配置参数
    config = TransformerConfig()
    
    # 加载数据集
    print("加载数据集中...")
    dataset = load_dataset("fjcanyue/wikipedia-zh-cn", 
                          data_files="wikipedia-zh-cn-20250901.json", 
                          split="train")
    
    # 加载分词器
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(config.sp_model_path)
    
    # 创建数据集和数据加载器
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        WikipediaDataset(train_dataset, tokenizer, config.max_seq_len),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        WikipediaDataset(val_dataset, tokenizer, config.max_seq_len),
        batch_size=config.batch_size
    )
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    
    # 学习率调度器
    def lr_lambda(step):
        step = max(step, 1)
        return min(step ** -0.5, step * (config.warmup_steps ** -1.5))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"开始训练，设备: {device}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"词表大小: {config.vocab_size}")
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        # 训练阶段
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [训练]')
        for batch in train_bar:
            optimizer.zero_grad()
            
            src = batch['source_ids'].to(device)
            tgt = batch['target_ids'].to(device)
            
            # 创建掩码
            src_mask, tgt_mask = model.generate_mask(src, tgt)
            
            # 前向传播
            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
            
            # 计算损失
            loss = criterion(output.contiguous().view(-1, config.vocab_size), 
                            tgt[:, 1:].contiguous().view(-1))
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [验证]')
            for batch in val_bar:
                src = batch['source_ids'].to(device)
                tgt = batch['target_ids'].to(device)
                
                src_mask, tgt_mask = model.generate_mask(src, tgt)
                output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
                
                loss = criterion(output.contiguous().view(-1, config.vocab_size), 
                                tgt[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: 训练损失 = {avg_train_loss:.4f}, 验证损失 = {avg_val_loss:.4f}')
        
        # 保存模型检查点
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': config
            }
            torch.save(checkpoint, f'transformer_epoch_{epoch+1}.pth')
            print(f"模型已保存: transformer_epoch_{epoch+1}.pth")
```

**关键细节解释**:

1. `load_dataset`函数的行为

   ```python
   dataset = load_dataset("fjcanyue/wikipedia-zh-cn", 
                         data_files="wikipedia-zh-cn-20250901.json", 
                         split="train")
   ```

   * `split="train"` 参数只是指定了要加载数据集的哪个分割（split），但在这个特定的数据集中，**实际上只有一个整体的数据集文件**，没有预定义的分割。
   * Hugging Face datasets库中的很多数据集确实提供了预划分的 train、validation、test分割，但需要数据集作者在创建时就已经定义好。
   * 对于 fjcanyue/wikipedia-zh-cn这个维基百科数据集，从文件名 wikipedia-zh-cn-20250901.json可以看出，它应该是完整的数据集，没有预划分（通过代码验证数据集**确实没有划分**，请参照**1.1**小节）。

## 11. 完整模型及训练代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import math
from tqdm import tqdm
from datasets import load_dataset
import os
from dataclasses import dataclass

# 固定随机种子
torch.manual_seed(42)


# 模型配置类
@dataclass
class TransformerConfig:
    # 模型配置
    d_model: int = 256  # 模型维度
    n_heads: int = 8  # 注意力头数
    d_ff: int = 1024  # 前馈网络维度
    n_layers: int = 4  # 编码器/解码器层数
    max_seq_len: int = 512  # 最大序列长度
    dropout: float = 0.1  # dropout比例

    # 训练配置
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000

    # 词表配置
    vocab_size: int = 30000  # 词表大小
    sp_model_path: str = "zh_wiki_spm_comprehensive.model"  # 词表路径

class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        pe = torch.zeros(config.max_seq_len, config.d_model)

        # 生成位置序列
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算div_term：10000^(2i/d_model)
        # 根据对数恒等: exp(a * log(b)) = b^a，我们可以进行变换
        # exp( -(2i * log(10000)) / d_model ) = 10000^(-2i / d_model)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2, dtype=torch.float) *
            - (math.log(10000.0) / config.d_model)
        )

        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度并注册为不参与梯度更新的缓冲区
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value形状: [batch_size, n_heads, seq_len, d_k]
        d_k = query.size(-1)

        # 计算注意力分数: Q*K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用掩码（防止解码器看到未来信息）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        assert config.d_model % config.n_heads == 0, "d_model必须能被n_heads整除"

        # 线性变换层
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)

        self.attention = ScaledDotProductAttention(config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性投影并分割成多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        x, attn_weights = self.attention(Q, K, V, mask)

        # 合并多头
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 输出投影
        return self.w_o(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # 两层线性变换 + ReLU激活
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 掩码自注意力（防止看到未来信息）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 编码器-解码器注意力, Q来自解码器, K V来自编码器
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, src, src_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器前向传播
        enc_output = self.encoder(src, src_mask)

        # 解码器前向传播
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # 输出层
        output = self.output_layer(dec_output)
        return output

    def generate_mask(self, src, tgt):
        # 源序列掩码（处理padding）
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # 目标序列掩码（防止解码器看到未来信息）
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        tgt_mask = tgt_mask & subsequent_mask.to(tgt.device)

        return src_mask, tgt_mask


class WikipediaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 构建输入：使用title作为源序列
        source_text = item['title']

        # 构建目标：使用text作为目标序列（用于文本生成任务）
        target_text = item['text']

        # 使用SentencePiece进行编码
        source_ids = self.tokenizer.encode_as_ids(source_text)
        target_ids = self.tokenizer.encode_as_ids(target_text)

        # 截断或填充到固定长度
        source_ids = source_ids[:self.max_length - 1] + [self.tokenizer.eos_id()]
        target_ids = target_ids[:self.max_length - 1] + [self.tokenizer.eos_id()]

        # 保存 padding 前的源序列长度
        len_source_ids = len(source_ids)

        # 填充到相同长度
        source_padding = [self.tokenizer.pad_id()] * (self.max_length - len_source_ids)
        target_padding = [self.tokenizer.pad_id()] * (self.max_length - len(target_ids))

        # source_ids 此处长度发生变化
        source_ids.extend(source_padding)
        target_ids.extend(target_padding)

        return {
            'source_ids': torch.LongTensor(source_ids),
            'target_ids': torch.LongTensor(target_ids),
            'source_mask': torch.LongTensor([1] * len_source_ids + [0] * len(source_padding))
        }


def train_transformer():
    # 配置参数
    config = TransformerConfig()

    # 加载数据集
    print("加载数据集中...")
    dataset = load_dataset("fjcanyue/wikipedia-zh-cn",
                           data_files="wikipedia-zh-cn-20250901.json",
                           split="train")

    # 加载分词器
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(config.sp_model_path)

    # 创建数据集和数据加载器
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        WikipediaDataset(train_dataset, tokenizer, config.max_seq_len),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        WikipediaDataset(val_dataset, tokenizer, config.max_seq_len),
        batch_size=config.batch_size
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())

    # 学习率调度器
    def lr_lambda(step):
        step = max(step, 1)
        return min(step ** -0.5, step * (config.warmup_steps ** -1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"开始训练，设备: {device}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"词表大小: {config.vocab_size}")

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        # 训练阶段
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs} [训练]')
        for batch in train_bar:
            optimizer.zero_grad()

            src = batch['source_ids'].to(device)
            tgt = batch['target_ids'].to(device)

            # 创建掩码
            src_mask, tgt_mask = model.generate_mask(src, tgt)

            # 前向传播
            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])

            # 计算损失
            loss = criterion(output.contiguous().view(-1, config.vocab_size),
                             tgt[:, 1:].contiguous().view(-1))

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs} [验证]')
            for batch in val_bar:
                src = batch['source_ids'].to(device)
                tgt = batch['target_ids'].to(device)

                src_mask, tgt_mask = model.generate_mask(src, tgt)
                output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1])

                loss = criterion(output.contiguous().view(-1, config.vocab_size),
                                 tgt[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch + 1}: 训练损失 = {avg_train_loss:.4f}, 验证损失 = {avg_val_loss:.4f}')

        # 保存模型检查点
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': config
            }
            torch.save(checkpoint, f'transformer_epoch_{epoch + 1}.pth')
            print(f"模型已保存: transformer_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train_transformer()
```

## 12. 关键实现要点

1. **位置编码**：使用正弦和余弦函数为每个位置生成独特的编码
2. **掩码机制**：解码器使用因果掩码防止信息泄漏
3. **残差连接**：每个子层都有残差连接，缓解梯度消失
4. **层归一化**：在每个子层后应用，稳定训练过程
5. **缩放注意力**：注意力分数除以$\sqrt{d_k}$防止梯度消失

这篇文章中的实现包含了Transformer的核心组件，实际应用中还需要添加数据预处理、学习率调度等组件。