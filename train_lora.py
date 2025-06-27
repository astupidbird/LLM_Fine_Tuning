import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
import time
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from qwen3_classification_direct import Qwen3ForSequenceClassification

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str,
                       default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B")
    parser.add_argument('--train_data', type=str, default="./balanced_train.xlsx")
    parser.add_argument('--val_data', type=str, default="./balanced_val.xlsx")
    parser.add_argument('--test_data', type=str, default="./val-r456-6-3000.xlsx")
    parser.add_argument('--output_dir', type=str, default="./lora_model")
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=18)  # 减小批次大小
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=256)  # 减小最大长度
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)  # 增加梯度累积步数
    parser.add_argument('--fp16', action='store_true', default=True)  # 默认启用混合精度训练
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化分布式训练
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        rank = 0
    
    # 只在主进程打印信息
    is_main_process = rank == 0
    
    if is_main_process:
        print(f"使用设备: {device}")
        print(f"世界大小: {world_size}")
        print(f"当前排名: {rank}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    if is_main_process:
        print("正在加载模型...")
    
    model = Qwen3ForSequenceClassification(args.checkpoint, num_labels=6)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # 减小LoRA秩以节省内存
        lora_alpha=16,  # 相应减小缩放参数
        lora_dropout=0.1,  # LoRA dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 减少目标模块
        modules_to_save=["lm_head"]  # 只保存lm_head层，因为它被替换为分类头
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    if is_main_process:
        print("LoRA配置:")
        model.print_trainable_parameters()
    
    # 将模型移到设备
    model.to(device)
    
    # 如果使用分布式训练，包装模型
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # 准备数据集
    if is_main_process:
        print("加载数据集...")
    
    train_dataset = ClassificationDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = ClassificationDataset(args.val_data, tokenizer, args.max_length)
    
    # 创建数据加载器
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,  # 减少工作进程数
        pin_memory=False  # 关闭pin_memory以节省内存
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,  # 减少工作进程数
        pin_memory=False  # 关闭pin_memory以节省内存
    )
    
    # 设置优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 混合精度训练 - 注意：BFloat16 不需要 GradScaler
    # 检查模型是否使用 BFloat16
    model_dtype = next(model.parameters()).dtype
    use_amp_scaler = args.fp16 and model_dtype != torch.bfloat16
    scaler = torch.amp.GradScaler('cuda') if use_amp_scaler else None
    
    # 创建输出目录
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    best_val_f1 = 0
    global_step = 0
    
    for epoch in range(args.num_epochs):
        if is_main_process:
            print(f"\n===== Epoch {epoch + 1}/{args.num_epochs} =====")
        
        # 设置分布式采样器的epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc="Training", disable=not is_main_process)
        
        for step, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度训练
            # BFloat16 模型使用 autocast 但不使用 scaler
            with torch.amp.autocast('cuda', enabled=args.fp16, dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.gradient_accumulation_steps
            
            if use_amp_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item() * args.gradient_accumulation_steps
            
            # 梯度累积
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if use_amp_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 日志记录
                if global_step % args.logging_steps == 0 and is_main_process:
                    avg_loss = train_loss / (step + 1)
                    train_pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # 评估
                if global_step % args.eval_steps == 0:
                    val_results = evaluate(model, val_loader, device)
                    
                    if is_main_process:
                        print(f"\n验证集结果 - Step {global_step}:")
                        print(f"Loss: {val_results['loss']:.4f}")
                        print(f"Accuracy: {val_results['accuracy']:.4f}")
                        print(f"F1: {val_results['f1']:.4f}")
                        print(f"Precision: {val_results['precision']:.4f}")
                        print(f"Recall: {val_results['recall']:.4f}")
                        
                        # 保存最佳模型
                        if val_results['f1'] > best_val_f1:
                            best_val_f1 = val_results['f1']
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                            print(f"保存最佳模型，F1: {best_val_f1:.4f}")
                    
                    model.train()
                
                # 定期保存检查点
                if global_step % args.save_steps == 0 and is_main_process:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(os.path.join(args.output_dir, f'checkpoint-{global_step}'))
# Epoch结束后的评估
        val_results = evaluate(model, val_loader, device)
        
        if is_main_process:
            print(f"\nEpoch {epoch + 1} 验证集结果:")
            print(f"Loss: {val_results['loss']:.4f}")
            print(f"Accuracy: {val_results['accuracy']:.4f}")
            print(f"F1: {val_results['f1']:.4f}")
    
    # 训练结束，在测试集上评估
    if is_main_process:
        print("\n===== 在测试集上评估 =====")
        test_dataset = ClassificationDataset(args.test_data, tokenizer, args.max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 加载最佳模型
        best_model_path = os.path.join(args.output_dir, 'best_model')
        if os.path.exists(best_model_path):
            # 重新初始化基础模型（使用原始checkpoint路径）
            model = Qwen3ForSequenceClassification(args.checkpoint, num_labels=6)
            # 应用LoRA配置
            model = get_peft_model(model, lora_config)
            # 加载保存的LoRA权重
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, best_model_path)
            model.to(device)
        
        test_results = evaluate(model, test_loader, device)
        
        print("\n测试集最终结果:")
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
        
        # 保存测试结果
        import json
        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump({
                'loss': test_results['loss'],
                'accuracy': test_results['accuracy'],
                'f1': test_results['f1'],
                'precision': test_results['precision'],
                'recall': test_results['recall']
            }, f, indent=4)

if __name__ == "__main__":
    main()
                    