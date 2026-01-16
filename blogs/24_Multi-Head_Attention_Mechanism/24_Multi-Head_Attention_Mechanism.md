Transformer中“多头注意力”的实现机制的核心在于：**让模型在不同的表示子空间里，并行地学习并关注输入序列的不同方面**。

## 1. 核心思想：为什么需要“多头”？

单一的自注意力机制虽然能学习序列元素间的依赖关系，但每一次计算只形成一种“注意力模式”。这就像人看一幅画，如果只看一次，可能只关注了颜色，而忽略了构图、纹理或细节。

“多头”机制的设计思想是：

- **并行化**：让模型同时、独立地以多种方式“审视”输入。
- **增强表达能力**：不同的“头”可以学习到不同的关系模式。例如，有的头关注局部语法依赖（如形容词修饰最近的名词），有的头关注长程指代关系（如代词指代远处的实体），有的头关注语义角色等。
- **稳定性**：类似于集成学习，多个头的组合比单一头更稳定，降低了模型对某个特定头权重的敏感性。

## 2. 实现机制：从“单头”到“多头”的分解步骤

我们以一个输入序列为例，假设：

- 序列长度：$L$
- 输入向量维度：$d_{\text{model}}$ (例如 512)
- 头数：$h$ (例如 8)
- 每个头的维度：$d_k = d_v = d_{\text{model}} / h$ (例如 512/8=64)

**步骤 1：初始化投影权重矩阵**

模型会初始化三组可学习的权重矩阵，用于将输入投影到查询、键、值的“子空间”：

- $W^Q_i \in \mathbb{R}^{(d_{\text{model}} \times d_k)}$： 第 $i$ 个头的`查询`投影矩阵
- $W^K_i \in \mathbb{R}^{(d_{\text{model}} \times d_k)}$ ： 第 $i$ 个头的`键`投影矩阵
- $W^V_i \in \mathbb{R}^{(d_{\text{model}} \times d_v)}$： 第 $i$ 个头的`值`投影矩阵 *(其中 $i = 1, 2, ..., h$)* 通常会有一个大的组合权重矩阵 $W^Q, W^K, W^V \in \mathbb{R}^{(d_{\text{model}} \times d_{\text{model}})}$，然后在计算时拆分成 $h$ 个头。

**步骤 2：为每个头生成独立的 Q, K, V**

对于输入矩阵 $X \in \mathbb{R}^{L \times d_{\text{model}}}$，我们为**每个头**计算其独有的 $Q$, $K$, $V$：

$Q_i = X W^Q_i \quad$ (形状：$(L, d_k)$)
$K_i = X W^K_i \quad$ (形状：$(L, d_k)$)
$V_i = X W^V_i \quad$ (形状：$(L, d_v)$)

这样，我们就得到了 $h$ 组 $(Q_i, K_i, V_i)$ 三元组，每组对应一个注意力头，专注于 $d_k$ 维的子空间。

**步骤 3：在每个头上并行执行缩放点积注意力**

对每个头 $i$，独立运行标准的自注意力公式：

$$
\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left( \frac{Q_i K_i^T}{\sqrt{d_k}} \right) V_i
$$

- $Q_i K_i^T$： 计算头 $i$ 的注意力分数矩阵，形状为 $(L, L)$，表示在该子空间下序列元素间的关联强度。
- $\text{softmax}(\cdots)$： 沿键的维度归一化，得到注意力权重。
- 乘以 $V_i$： 用权重对值向量进行加权求和，得到头 $i$ 的输出 $\text{head}_i$，形状为 $(L, d_v)$。

**关键**：这 $h$ 个注意力计算是**完全并行**的，可以高效地在GPU上执行。

**步骤 4：拼接所有头的输出**

将所有 $h$ 个头的输出矩阵 $\text{head}_i$ 在特征维度上拼接起来：

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)
$$

拼接后的矩阵形状为 $(L, h \cdot d_v) = (L, d_{\text{model}})$。

**步骤 5：线性投影**

将拼接后的结果通过一个可学习的输出投影矩阵 $W^O \in \mathbb{R}^{(d_{\text{model}} \times d_{\text{model}})}$ 进行变换：

$$
\text{Output} = \text{MultiHead}(X) \cdot W^O \quad \text{形状：}(L, d_{\text{model}})
$$

这一步的作用是：

1. **融合信息**：将来自不同子空间的信息进行混合和重组。
2. **保持维度**：确保输出维度与输入 $d_{\text{model}}$ 一致，以便与残差连接和前馈网络无缝衔接。

## 3. 核心技巧：高效实现与张量操作

在实际代码（如PyTorch）中，**不会真的用for循环处理每个头**，而是利用张量运算一次性完成所有计算。

**高效实现流程**：

1. **线性投影**：将输入 $X$ 通过三个大矩阵 $W^Q$, $W^K$, $W^V$ 投影，得到 $Q$, $K$, $V$，形状均为 $(L, d_{\text{model}})$。
2. **重塑（Reshape）**：将 $Q$, $K$, $V$ 重塑为 $(L, h, d_k)$。这相当于在特征维度上“分出”了 $h$ 个头。
3. **转置**：将形状转为 $(h, L, d_k)$，使“头”的维度在最前面，便于批处理。
4. **批量矩阵乘法**：使用 `torch.bmm` 等函数，一次性计算 $h$ 个头的注意力。计算 $(h, L, d_k)$ 与 $(h, d_k, L)$ 的批矩阵乘法，得到 $(h, L, L)$ 的注意力分数矩阵。
5. **计算注意力权重并加权求和**：在 $h$ 个头上并行完成softmax和与 $V$ 的乘法，得到 $(h, L, d_v)$ 的输出。
6. **重塑回原状**：将输出转置回 $(L, h, d_v)$，然后重塑为 $(L, d_{\text{model}})$。
7. **输出投影**：通过 $W^O$ 进行线性变换。

**核心代码逻辑（伪代码）**：
```
def multi_head_attention(X, W_Q, W_K, W_V, W_O, h):

batch_size, L, d_model = X.shape

d_k = d_v = d_model // h

# 1. 线性投影并分头
Q = X @ W_Q  # (L, d_model)
K = X @ W_K
V = X @ W_V

# 2. 重塑为多头形式
Q = Q.reshape(batch_size, L, h, d_k).transpose(1, 2)  # (batch_size, h, L, d_k)
K = K.reshape(batch_size, L, h, d_k).transpose(1, 2)
V = V.reshape(batch_size, L, h, d_v).transpose(1, 2)

# 3. 并行计算缩放点积注意力（Scaled Dot-Product Attention）
attn_scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)  # (batch_size, h, L, L)
attn_weights = F.softmax(attn_scores, dim=-1)
context = attn_weights @ V  # (batch_size, h, L, d_v)

# 4. 合并多头
context = context.transpose(1, 2).reshape(batch_size, L, d_model)  # (L, d_model)

# 5. 输出投影
output = context @ W_O  # (L, d_model)
return output
```

## 4. 多头注意力的优势总结

1. **表达能力的提升**：模型能同时关注来自不同位置的不同信息，增强了其建模复杂依赖关系的能力。
2. **计算效率**：由于 $d_k$ 和 $d_v$ 是 $d_{\text{model}}/h$，点积注意力（复杂度 $O(L^2 \cdot d_k)$）的总计算量在 $h$ 个头上与单头大矩阵（$O(L^2 \cdot d_{\text{model}})$）相近甚至更低，但并行性更好。
3. **表征的多样性**：为后续的前馈网络提供了更丰富、多视角的中间表示。

**一个直观比喻**：

多头注意力就像让一组专家（多个头）同时审阅同一份文档。每位专家有自己的专长（由不同的投影矩阵决定），他们分别写下自己的分析报告（每个头的输出），最后由主编（$W^O$ 投影）汇总成一份全面的最终报告。这比只让一位专家审阅要全面和稳健得多。