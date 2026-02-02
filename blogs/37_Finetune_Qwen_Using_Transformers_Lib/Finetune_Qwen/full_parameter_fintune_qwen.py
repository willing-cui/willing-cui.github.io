import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
import os

print(f"pytorch版本：{torch.__version__}")
print(f"transformers版本：{transformers.__version__}")

# 1. 加载模型和分词器
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 确保有填充token
tokenizer.pad_token = tokenizer.eos_token

# 加载模型（全精度）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 2. 加载古诗数据集
def load_poetry_dataset():
    dataset = load_dataset("larryvrh/Chinese-Poems")

    # 将数据集转换为指令格式
    def format_instruction(example):
        return {
            "text": f"请创作一首古诗，主题关于{example['title']}：\n{example['content']}"
        }

    formatted_dataset = dataset.map(format_instruction)
    return formatted_dataset

dataset = load_poetry_dataset()

# 3. 数据预处理和分词
def tokenize_function(examples):
    # 分词处理
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors=None  # 确保返回的是列表而不是张量，以便后续批处理
    )

    # 对于因果语言建模，labels就是input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# 应用分词函数
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

# 分割训练集和验证集
train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 4. 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 不使用掩码语言建模
    pad_to_multiple_of=8    # 优化GPU内存对齐
)

# 5. 训练参数配置
training_args = TrainingArguments(
    output_dir="./qwen3-0.6B-poetry-finetuned",

    # 训练参数
    num_train_epochs=5,
    per_device_train_batch_size=4,  # 根据显存调整
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # 有效批次大小 = 4 * 4 = 16

    # 优化器参数
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_steps=10,

    # 训练调度
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",

    # 精度和性能
    fp16=False,  # 关闭混合精度训练
    dataloader_pin_memory=False,  # 启用内存固定加速数据传输（仅支持GPU）
    gradient_checkpointing=False,  # 使用梯度检查点节省显存（时间换空间）

    # 模型保存
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,

    # 报告设置
    report_to="none"
)

# 6. 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# 7. 开始训练
print("开始全量微调训练...")
print(f"训练样本数: {len(train_dataset)}")
print(f"验证样本数: {len(eval_dataset)}")
print(f"模型可训练参数数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

train_result = trainer.train()

# 8. 保存最终模型
trainer.save_model()
trainer.save_state()

print("训练完成！")
print(f"训练损失: {train_result.metrics['train_loss']}")
print(f"验证损失: {train_result.metrics['eval_loss']}")