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
                output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])

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