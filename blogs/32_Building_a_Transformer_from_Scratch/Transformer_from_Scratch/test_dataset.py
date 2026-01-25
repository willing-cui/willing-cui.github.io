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