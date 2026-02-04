from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from torch import bfloat16
import torch
from PIL import Image
import os


def resize_image_long_edge(image, max_size=1024):
    """
    按长边缩放图片

    参数:
        image: PIL Image对象
        max_size: 长边的最大尺寸（默认1024）

    返回:
        缩放后的PIL Image对象
    """
    # 获取原始尺寸
    original_width, original_height = image.size

    # 确定长边
    if original_width >= original_height:
        # 宽是长边
        new_width = max_size
        new_height = int(original_height * (max_size / original_width))
    else:
        # 高是长边
        new_height = max_size
        new_width = int(original_width * (max_size / original_height))

    # 使用高质量的重采样方法
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    print(f"图片尺寸从 {original_width}x{original_height} 缩放为 {new_width}x{new_height}")
    return resized_image


# 1. 配置量化参数 - 用于减少模型内存占用的4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4-bit量化加载，显著减少显存使用
    bnb_4bit_quant_type="nf4",  # 使用NF4量化类型，针对正态分布权重优化
    bnb_4bit_compute_dtype=bfloat16,  # 计算时使用bfloat16精度，平衡精度和速度
    bnb_4bit_use_double_quant=True,  # 启用双重量化，进一步压缩量化参数
)

# 2. 指定模型名称并加载
model_name = "Qwen/Qwen3-VL-2B-Instruct"  # 通义千问的多模态视觉语言模型，20亿参数

# 加载处理器 - 负责文本分词和图像预处理
processor = AutoProcessor.from_pretrained(model_name)

# 加载模型并应用量化配置
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 应用上述量化配置
    device_map="auto",  # 自动分配模型层到可用设备（GPU/CPU）
    torch_dtype=bfloat16,  # 模型权重使用bfloat16精度
    trust_remote_code=True  # 信任远程代码执行（对于自定义模型需要）
)

# 3. 准备输入：图片和问题
image_path = "./test.jpg"
question = "请详细描述这张图片的内容。"

# 设置长边最大尺寸（可调整）
MAX_LONG_EDGE_SIZE = 1024  # 根据需要修改这个值，如512, 768, 1024, 1536等

# 确保图片文件存在
if not os.path.exists(image_path):
    raise FileNotFoundError(f"图片文件不存在: {image_path}")

# 使用PIL打开图片
try:
    image = Image.open(image_path)
    # 确保图片格式正确，转换为RGB三通道格式
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 获取原始尺寸
    original_size = image.size
    print(f"原始图片尺寸: {original_size[0]}x{original_size[1]}")

    # 缩放图片（长边不超过MAX_LONG_EDGE_SIZE）
    if max(original_size) > MAX_LONG_EDGE_SIZE:
        image = resize_image_long_edge(image, MAX_LONG_EDGE_SIZE)
    else:
        print(f"图片尺寸已小于等于设置的长边限制({MAX_LONG_EDGE_SIZE})，无需缩放")

except Exception as e:
    raise ValueError(f"无法打开或处理图片: {e}")

# 构建多模态输入消息，符合模型要求的格式
messages = [
    {
        "role": "user",  # 用户角色
        "content": [
            {"type": "image", "image": image},  # 图像输入，直接传递PIL Image对象
            {"type": "text", "text": question}  # 文本输入，用户问题
        ]
    }
]

# 4. 处理输入并生成文本
try:
    # 应用聊天模板处理多模态输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,  # 对文本进行分词，转换为token IDs
        add_generation_prompt=True,  # 添加生成提示，告诉模型开始生成回复
        return_dict=True,  # 返回字典格式
        return_tensors="pt"  # 返回PyTorch张量
    )
    inputs = inputs.to(model.device)  # 将输入移动到模型所在的设备（GPU/CPU）

    # 生成参数配置
    generated_ids = model.generate(
        **inputs,  # 解包输入字典
        max_new_tokens=500,  # 最大生成token数量，控制回复长度
        do_sample=True,  # 启用采样生成，而非贪婪解码，使输出更多样
        temperature=0.7,  # 温度参数：值越高输出越随机，越低越确定
        top_p=0.9  # 核采样参数：仅从概率质量前90%的token中采样
    )

    # 解码并打印结果
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True  # 跳过特殊token（如<EOS>结束符）
    )[0]
    print("模型生成的描述：", generated_text)

except Exception as e:
    print(f"处理过程中发生错误: {e}")