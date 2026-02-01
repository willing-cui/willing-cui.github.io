## 0. 项目目标及可行性分析

在个人电脑（32GB RAM + 8GB VRAM）上微调Qwen模型是可行的，但需要选择合适的微调方法和参数。我们选择<a href="https://huggingface.co/Qwen/Qwen3-0.6B" target="_blank" rel="noopener noreferrer">Qwen3-0.6B</a>模型。Qwen3-0.6B 是一个轻量级模型，微调时

* **显存占用**：在 FP16 精度下，全参数微调约需 1.2GB 显存；使用 4-bit 量化（QLoRA）后，显存占用可降至约 0.5GB。
* **内存需求**：推荐配置为 16GB 系统内存。

在此项目中，我们验证两种微调方案：

**全参数微调 (Full Fine-tuning)**

- **适用场景**：显存充足，希望获得最佳微调效果。
- **可行性**：在 8GB 显存上可以轻松运行，甚至能支持较大的批次大小（batch size）。

**LoRA / QLoRA 微调**

- **适用场景**：显存有限，或希望快速实验、节省训练时间。
- **可行性**：在 4GB 显存的入门级显卡上即可流畅运行， 8GB 显存配置非常宽裕。

我们的目标是使用 <a href="https://huggingface.co/datasets/larryvrh/Chinese-Poems" target="_blank" rel="noopener noreferrer">larryvrh/Chinese-Poems</a> 数据集微调 Qwen-0.6B 以生成**古诗词风格文本**。

### 本项目中必需的python包

```bash
pip install datasets	# Hugging Face datasets
pip install transformers
pip install accelerate

# 避免安装失败,先升级pip
python.exe -m pip install --upgrade pip
python.exe -m pip install --upgrade pip setuptools

nvcc --version	# 查看当前安装的cuda版本
# 按照运行环境安装对应版本的pytorch
# https://pytorch.org/get-started/previous-versions/
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

## 方案1：全参数微调

混合精度训练本身**不会直接引起梯度爆炸**，但它会**放大并暴露**原本就存在的梯度爆炸问题，使其更容易导致训练崩溃。

##### 核心原因：数值范围缩小

梯度爆炸的根本原因在于深层网络的反向传播过程中，梯度值被逐层放大，最终超出浮点数的可表示范围。混合精度训练（如使用FP16）会加剧这一问题，因为FP16的数值表示范围远小于FP32。

- **FP32 范围**：约 `1.18e-38`到 `3.4e+38`
- **FP16 范围**：约 `5.96e-8`到 `6.55e+4`

在FP32下可能只是“大梯度”的问题，在FP16下会直接变为**数值溢出（NaN/inf）**，导致训练崩溃