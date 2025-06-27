import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import argparse
from tqdm import tqdm
from qwen3_classification_direct import Qwen3ForSequenceClassification

# 自定义数据集类
class ClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = pd.read_excel(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 定义标签映射
        self.label_map = {
            '正常': 0,
            '歧视': 1,
            '违法违规': 2,
            '政治安全': 3,
            '暴恐': 4,
            '色情低俗': 5
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text_cn'])
        label_text = str(row['extracted_label']).strip()
        
        # 将文本标签转换为数字
        if label_text in self.label_map:
            label = self.label_map[label_text]
        else:
            # 如果标签不在映射中，尝试将其作为数字处理
            try:
                label = int(label_text)
            except ValueError:
                raise ValueError(f"未知的标签: '{label_text}'。支持的标签: {list(self.label_map.keys())}")
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 计算每个类别的准确率
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # 计算每个类别的准确率
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = []
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            per_class_accuracy.append(cm[i][i] / cm[i].sum())
        else:
            per_class_accuracy.append(0.0)
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class,
        'per_class_f1': f1_per_class,
        'per_class_support': support_per_class
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False, default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B",
                        help="原始模型的路径")
    parser.add_argument('--lora_model', type=str, required=False, default="./lora_model/checkpoint-3000",
                        help="保存的LoRA模型路径")
    parser.add_argument('--test_data', type=str, required=False, default="./val-r456-6-3000.xlsx",
                        help="测试数据集路径")
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--max_length', type=int, default=256)
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = Qwen3ForSequenceClassification(args.checkpoint, num_labels=6)
    
    # 直接从保存的LoRA模型加载
    print(f"加载LoRA模型: {args.lora_model}")
    model = PeftModel.from_pretrained(base_model, args.lora_model)
    
    # 将模型移到设备
    model.to(device)
    model.eval()
    
    print("LoRA模型加载完成！")
    print(f"模型已加载到 {device}")
    
    # 准备测试数据集
    print("加载测试数据集...")
    test_dataset = ClassificationDataset(args.test_data, tokenizer, args.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 评估模型
    print("\n开始评估...")
    test_results = evaluate(model, test_loader, device)
    
    # 打印结果
    print("\n测试集结果:")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1: {test_results['f1']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    
    # 定义标签名称
    label_names = ['正常', '歧视', '违法违规', '政治安全', '暴恐', '色情低俗']
    
    # 打印每个类别的详细指标
    print("\n每个类别的详细指标:")
    print(f"{'类别':<10} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'样本数':<8}")
    print("-" * 60)
    for i, label_name in enumerate(label_names):
        if i < len(test_results['per_class_accuracy']):
            acc = test_results['per_class_accuracy'][i]
            prec = test_results['per_class_precision'][i]
            rec = test_results['per_class_recall'][i]
            f1 = test_results['per_class_f1'][i]
            sup = test_results['per_class_support'][i]
            print(f"{label_name:<10} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f} {sup:<8}")
    
    # 打印混淆矩阵
    cm = confusion_matrix(test_results['labels'], test_results['predictions'])
    print("\n混淆矩阵:")
    print(cm)
    
    # 保存结果
    import json
    output_dir = os.path.dirname(args.lora_model)
    results_file = os.path.join(output_dir, 'test_results_lora.json')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'loss': test_results['loss'],
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'per_class_metrics': {
                label_names[i]: {
                    'accuracy': test_results['per_class_accuracy'][i],
                    'precision': test_results['per_class_precision'][i],
                    'recall': test_results['per_class_recall'][i],
                    'f1': test_results['per_class_f1'][i],
                    'support': int(test_results['per_class_support'][i])
                } for i in range(len(label_names))
            }
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")

if __name__ == "__main__":
    main()