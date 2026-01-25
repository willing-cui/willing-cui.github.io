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