from email.policy import strict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os

print("当前工作目录:", os.getcwd())
checkpoint = "/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B"

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")

# 查看模型结构
print("=== 模型结构 ===")
print(model)

# 查看关键的输出层参数
print("\n=== 关键输出层参数 ===")
for name, param in model.named_parameters():
    if any(key in name.lower() for key in ['lm_head', 'embed_out', 'output', 'norm']):
        print(f"Parameter name: {name}, shape: {param.size()}")

print(f"\nLoaded in {time.time() - start_time:.2f} seconds")

