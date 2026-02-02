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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from peft import PeftModel

print(f"pytorch版本：{torch.__version__}")
print(f"transformers版本：{transformers.__version__}")

# 1. 加载模型和分词器
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 确保有填充token
tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 准备模型用于LoRA训练
model = prepare_model_for_kbit_training(model)

# 3. 配置LoRA参数
lora_config = LoraConfig(
    r=8,  # LoRA秩
    lora_alpha=32,  # LoRA alpha参数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块
    lora_dropout=0.1,  # Dropout率
    bias="none",  # 不训练偏置
    task_type=TaskType.CAUSAL_LM,  # 因果语言建模任务
)

# 应用LoRA配置
model = get_peft_model(model, lora_config)

# 打印可训练参数数量
trainable_params = 0
all_params = 0
for name, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()

print(f"可训练参数数量: {trainable_params:,}")
print(f"总参数数量: {all_params:,}")
print(f"可训练参数占比: {100 * trainable_params / all_params:.2f}%")


# 4. 加载古诗数据集
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

# 5. 数据预处理和分词
def tokenize_function(examples):
    # 分词处理
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None
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

# 6. 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 不使用掩码语言建模
    pad_to_multiple_of=16    # 优化GPU内存对齐
)

# 7. 训练参数配置
training_args = TrainingArguments(
    output_dir="./qwen3-0.6B-poetry-lora",

    # 训练参数
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,

    # 优化器参数
    learning_rate=5e-5,  # LoRA可以使用更高的学习率
    weight_decay=0.01,
    warmup_steps=10,

    # 训练调度
    logging_steps=100,
    eval_steps=1000,
    save_steps=10000,
    eval_strategy="steps",
    save_strategy="steps",

    # 精度和性能
    fp16=False,
    bf16=True,
    dataloader_pin_memory=True, # 启用内存固定加速数据传输（仅支持GPU）
    gradient_checkpointing=False,   # 使用梯度检查点节省显存（时间换空间）

    # 模型保存
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,

    # 报告设置
    report_to="swanlab",
    run_name="lora_finetune_qwen_try_1",
)

# 8. 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,  # 使用tokenizer参数
)

# 9. 开始训练
print("开始LoRA微调训练...")
print(f"训练样本数: {len(train_dataset)}")
print(f"验证样本数: {len(eval_dataset)}")

train_result = trainer.train()

# 10. 保存模型
# 保存LoRA适配器
print("\n保存LoRA适配器...")
model.save_pretrained("./qwen3-0.6B-poetry-lora/lora-adapter")

# 保存训练状态
trainer.save_state()

# 可选的：合并模型并保存
print("合并LoRA权重到基础模型...")
model = model.merge_and_unload()
model.save_pretrained("./qwen3-0.6B-poetry-lora/merged-model")
tokenizer.save_pretrained("./qwen3-0.6B-poetry-lora/merged-model")

print("\n训练完成！")
print(f"训练损失: {train_result.metrics.get('train_loss', 'N/A')}")
print(f"验证损失: {train_result.metrics.get('eval_loss', 'N/A')}")