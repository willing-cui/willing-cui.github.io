from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from torch import bfloat16
import torch

# 1. 配置量化参数 - 以4-bit量化为例
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4-bit量化
    bnb_4bit_quant_type="nf4",  # 使用NF4数据类型，对正态分布权重更友好
    bnb_4bit_compute_dtype=bfloat16,  # 计算时使用bfloat16以加速并节省内存
    bnb_4bit_use_double_quant=True,  # 启用双重量化，进一步压缩模型大小
)

# 2. 指定模型名称并加载（以2B模型为例）
model_name = "Qwen/Qwen3-VL-2B-Instruct"

# 加载处理器
processor = AutoProcessor.from_pretrained(model_name)

# 加载模型并应用量化配置
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 应用上面的量化配置
    device_map="auto",  # 自动将模型分配到可用的GPU或CPU上
    torch_dtype=bfloat16,
    trust_remote_code=True
)

# 3. 准备输入：图片和问题
image_path = "./test.jpg"
question = "详细描述这张图片的内容。"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]
    }
]

# 4. 处理输入并生成文本
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 生成参数
generated_ids = model.generate(
    **inputs,
    max_new_tokens=500,  # 设置生成文本的最大长度
    do_sample=True,  # 启用采样，使生成结果更多样
    temperature=0.7,  # 控制随机性：值越低输出越确定，越高越随机
    top_p=0.9  # 核采样（Nucleus Sampling）：从累积概率超过top_p的最小词集合中采样
)

# 解码并打印结果
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("模型生成的描述：", generated_text)