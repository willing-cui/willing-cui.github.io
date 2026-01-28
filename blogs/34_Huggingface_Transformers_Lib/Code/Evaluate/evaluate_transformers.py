import torch
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel

# Test installation
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")

# Evaluation case 1
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face Transformers!")
print(result)

# Evaluation case 2
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Hello, Hugging Face!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)