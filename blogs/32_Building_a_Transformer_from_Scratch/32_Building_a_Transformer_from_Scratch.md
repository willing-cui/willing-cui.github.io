Transformer是自然语言处理领域的革命性架构，理解其实现细节对掌握现代大语言模型至关重要。我们将从零开始实现一个完整的Transformer模型。

## 1. 环境准备与依赖导入

首先导入必要的库并设置随机种子以确保结果可复现：

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass

# 固定随机种子
torch.manual_seed(42)

# 模型配置类
@dataclass
class TransformerConfig:
    vocab_size: int = 10000  # 词表大小
    d_model: int = 512      # 模型维度
    n_heads: int = 8        # 注意力头数
    d_ff: int = 2048        # 前馈网络维度
    n_layers: int = 6       # 编码器/解码器层数
    max_seq_len: int = 1000 # 最大序列长度
    dropout: float = 0.1    # dropout比例
```

## 2. 位置编码实现

Transformer没有循环结构，需要显式添加位置信息：

```
class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        pe = torch.zeros(config.max_seq_len, config.d_model)
        
        # 生成位置序列
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term：10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float() * 
            (-math.log(10000.0) / config.d_model)
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

## 3. 缩放点积注意力

这是Transformer的核心机制：

```
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

让模型能够同时关注不同表示子空间的信息：

```
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

## 5. 前馈网络

对每个位置独立应用相同的全连接网络：

```
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

```
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

比编码器层多一个编码器-解码器注意力机制：

```
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
        
        # 编码器-解码器注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

## 8. 完整编码器和解码器

```
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

## 9. 完整的Transformer模型

```
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

## 10. 训练示例

下面是一个简单的训练流程：

```
def train_transformer():
    config = TransformerConfig(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        max_seq_len=1000,
        dropout=0.1
    )
    
    model = Transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 模拟训练数据
    batch_size = 32
    seq_len = 50
    for epoch in range(10):
        # 模拟训练数据
        src = torch.randint(1, config.vocab_size, (batch_size, seq_len))
        tgt = torch.randint(1, config.vocab_size, (batch_size, seq_len))
        
        # 生成掩码
        src_mask, tgt_mask = model.generate_mask(src, tgt[:, :-1])
        
        # 前向传播
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        
        # 计算损失
        loss = criterion(output.reshape(-1, config.vocab_size), tgt[:, 1:].reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_transformer()
```

## 关键实现要点

1. **位置编码**：使用正弦和余弦函数为每个位置生成独特的编码
2. **掩码机制**：解码器使用因果掩码防止信息泄漏
3. **残差连接**：每个子层都有残差连接，缓解梯度消失
4. **层归一化**：在每个子层后应用，稳定训练过程
5. **缩放注意力**：注意力分数除以√d_k防止梯度消失

这个实现包含了Transformer的核心组件，你可以根据需要调整超参数或修改架构。实际应用中还需要添加数据预处理、学习率调度等组件。