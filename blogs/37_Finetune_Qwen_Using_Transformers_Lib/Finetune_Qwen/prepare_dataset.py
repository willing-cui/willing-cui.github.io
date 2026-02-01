from datasets import load_dataset

# 加载古诗数据集
dataset = load_dataset("larryvrh/Chinese-Poems") 

# 转换为指令微调格式
def format_poem(example):
    return {
        "text": f"创作一首古诗，风格类似《{example['title']}》：\n{example['content']}"
    }

dataset = dataset.map(format_poem)

# 分割训练/验证集
dataset = dataset['train'].train_test_split(test_size=0.1)
print(dataset["train"][0])  # 检查样例