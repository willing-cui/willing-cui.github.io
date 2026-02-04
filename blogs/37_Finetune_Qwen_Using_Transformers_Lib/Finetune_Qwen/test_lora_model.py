# example_usage.py
"""
使用微调后的模型进行古诗创作
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def create_poetry(model_path, theme, max_length=100, temperature=0.1):
    """
    使用微调后的模型创作古诗

    Args:
        model_path: 模型路径
        theme: 古诗主题
        max_length: 最大生成长度
        temperature: 温度参数，控制创造性
    """
    # 加载模型和分词器
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # 构建提示词
    prompt = f"请创作一首七言绝句，主题关于{theme}："
    print(f"\n主题: {theme}")
    print("=" * 40)

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成古诗
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码输出
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    poetry = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text

    return poetry


# 使用示例
if __name__ == "__main__":
    # 模型路径（使用合并后的模型）
    model_path = "./qwen3-0.6B-poetry-lora/merged-model"

    # 测试不同主题
    themes = ["春天", "山水", "思乡", "梅花", "秋夜", "边塞"]

    for theme in themes:
        try:
            poetry = create_poetry(model_path, theme)
            print(f"\n主题: {theme}")
            print("-" * 30)
            print(poetry)
            print()
        except Exception as e:
            print(f"生成{theme}主题古诗时出错: {e}")