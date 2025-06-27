# LLM Fine-Tuning 项目

基于 Qwen3-1.7B 的大语言模型微调项目，专门用于中文文本分类任务。该项目实现了使用 LoRA（Low-Rank Adaptation）技术对 Qwen3 模型进行高效微调，用于识别文本内容的安全分类。

## 项目概述

本项目专注于中文文本内容安全分类，支持以下6个类别：
- 正常 (0)
- 歧视 (1)
- 违法违规 (2)
- 政治安全 (3)
- 暴恐 (4)
- 色情低俗 (5)

## 项目架构

### 核心文件说明

| 文件名 | 功能描述 |
|--------|----------|
| [`train_lora.py`](train_lora.py) | LoRA微调训练脚本，支持分布式训练和混合精度 |
| [`test_saved_lora.py`](test_saved_lora.py) | 保存的LoRA模型测试评估脚本 |
| [`qwen3_classification_direct.py`](qwen3_classification_direct.py) | Qwen3分类模型实现，直接替换lm_head为分类头 |
| [`classification_model.py`](classification_model.py) | 自定义分类模型实现（备选方案） |
| [`get_data_by_labels.py`](get_data_by_labels.py) | MongoDB数据提取工具，支持按标签采样数据 |
| [`load_model.py`](load_model.py) | 模型加载和结构查看工具 |

### 技术特点

- **高效微调**: 使用LoRA技术，只训练少量参数，大幅减少计算资源需求
- **分布式训练**: 支持多GPU分布式训练，提高训练效率
- **混合精度**: 支持FP16/BF16混合精度训练，节省显存
- **数据平衡**: 提供数据采样工具，支持类别平衡的数据集构建
- **详细评估**: 提供完整的分类指标评估，包括混淆矩阵和各类别详细指标

## 环境要求

### 硬件要求
- GPU: 建议使用具有至少24GB显存的GPU（如A100、V100等）
- CPU: 多核处理器
- 内存: 建议32GB以上

### 软件依赖
```bash
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
tqdm>=4.64.0
pymongo>=4.0.0  # 用于数据提取
openpyxl>=3.0.0  # 用于Excel文件处理
```

## 安装指南

1. **克隆项目**
```bash
git clone <repository-url>
cd LLM_Fine_Tuning
```

2. **安装依赖**
```bash
pip install torch transformers peft pandas numpy scikit-learn tqdm pymongo openpyxl
```

3. **准备模型**
下载 Qwen3-1.7B 模型到本地目录，或使用 Hugging Face 模型路径。

## 使用指南

### 1. 数据准备

#### 从MongoDB提取数据
使用 [`get_data_by_labels.py`](get_data_by_labels.py) 从MongoDB数据库中提取训练数据：

```bash
# 提取平衡的训练数据集
python get_data_by_labels.py \
  -u admin \
  --auth-database scanadata \
  -l "正常:15000,政治安全:6000,暴恐:6000,歧视:6000,色情低俗:6000,违法违规:6000" \
  -f text_cn,n_labels \
  --random \
  --seed 123 \
  -o balanced_train.xlsx

# 提取验证数据集
python get_data_by_labels.py \
  -u admin \
  --auth-database scanadata \
  -l "正常:3000,政治安全:3000,暴恐:3000,歧视:3000,色情低俗:3000,违法违规:3000" \
  -f text_cn,n_labels \
  --random \
  --seed 456 \
  -o balanced_val.xlsx
```

#### 数据格式要求
数据文件应为Excel格式，包含以下列：
- `text_cn`: 中文文本内容
- `extracted_label`: 文本标签（正常/歧视/违法违规/政治安全/暴恐/色情低俗）

### 2. 模型训练

使用 [`train_lora.py`](train_lora.py) 进行LoRA微调训练：

#### 单GPU训练
```bash
python train_lora.py \
  --checkpoint /path/to/Qwen3-1.7B \
  --train_data ./balanced_train.xlsx \
  --val_data ./balanced_val.xlsx \
  --test_data ./test_data.xlsx \
  --output_dir ./lora_model \
  --num_epochs 3 \
  --batch_size 18 \
  --learning_rate 5e-5 \
  --max_length 256 \
  --fp16
```

#### 分布式训练
```bash
torchrun --nproc_per_node=4 train_lora.py \
  --checkpoint /path/to/Qwen3-1.7B \
  --train_data ./balanced_train.xlsx \
  --val_data ./balanced_val.xlsx \
  --output_dir ./lora_model \
  --num_epochs 3 \
  --batch_size 18 \
  --learning_rate 5e-5
```

#### 训练参数说明
- `--checkpoint`: Qwen3模型路径
- `--train_data`: 训练数据文件路径
- `--val_data`: 验证数据文件路径
- `--test_data`: 测试数据文件路径
- `--output_dir`: 模型输出目录
- `--num_epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--max_length`: 最大序列长度
- `--fp16`: 启用混合精度训练

### 3. 模型评估

使用 [`test_saved_lora.py`](test_saved_lora.py) 评估训练好的LoRA模型：

```bash
python test_saved_lora.py \
  --checkpoint /path/to/Qwen3-1.7B \
  --lora_model ./lora_model/best_model \
  --test_data ./test_data.xlsx \
  --batch_size 18 \
  --max_length 256
```

评估脚本将输出：
- 整体准确率、精确率、召回率、F1分数
- 各类别详细指标
- 混淆矩阵
- 结果保存到JSON文件

## 模型架构

### Qwen3ForSequenceClassification
[`qwen3_classification_direct.py`](qwen3_classification_direct.py) 实现的核心分类模型：

- **基础模型**: Qwen3-1.7B (2048维隐藏层)
- **分类头**: 直接替换原始lm_head层 (151936 → 6 个类别)
- **池化策略**: 使用序列最后一个有效token的表示
- **正则化**: Dropout(0.1)
- **初始化**: Xavier均匀初始化

### LoRA配置
```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # LoRA秩
    lora_alpha=16,          # 缩放参数
    lora_dropout=0.1,       # LoRA dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["lm_head"]
)
```

## 性能优化

### 内存优化
1. **梯度累积**: 使用 `gradient_accumulation_steps=2` 减少显存使用
2. **混合精度**: 启用FP16训练，节省约50%显存
3. **序列长度**: 限制 `max_length=256`，适合大多数文本
4. **批次大小**: 调整 `batch_size=18`，平衡速度和显存

### 训练策略
1. **学习率调度**: 线性预热 + 线性衰减
2. **梯度裁剪**: 防止梯度爆炸
3. **早停机制**: 基于验证集F1分数保存最佳模型
4. **定期评估**: 每200步进行一次验证集评估

## 项目结构

```
LLM_Fine_Tuning/
├── train_lora.py                    # LoRA训练脚本
├── test_saved_lora.py              # 模型测试脚本
├── qwen3_classification_direct.py  # Qwen3分类模型
├── classification_model.py         # 备选分类模型
├── get_data_by_labels.py           # 数据提取工具
├── load_model.py                   # 模型加载工具
├── README.md                       # 项目文档
├── balanced_train.xlsx             # 训练数据（需生成）
├── balanced_val.xlsx               # 验证数据（需生成）
└── lora_model/                     # 训练输出目录
    ├── best_model/                 # 最佳模型
    ├── checkpoint-*/               # 训练检查点
    └── test_results.json           # 测试结果
```

## 数据集说明

### 标签分布建议
- **训练集**: 正常(15000) + 其他类别各6000条
- **验证集**: 各类别各3000条
- **测试集**: 各类别各3000条

### 数据质量要求
1. **文本长度**: 建议10-500字符
2. **标签准确性**: 确保标签与内容匹配
3. **类别平衡**: 避免严重的类别不平衡
4. **数据去重**: 避免训练集和测试集重复

## 常见问题

### Q1: 显存不足怎么办？
**A**: 尝试以下方法：
- 减小 `batch_size`
- 增加 `gradient_accumulation_steps`
- 减小 `max_length`
- 启用 `fp16` 混合精度

### Q2: 训练收敛慢？
**A**: 检查以下设置：
- 学习率是否合适（建议5e-5）
- 数据是否平衡
- LoRA参数是否合理

### Q3: 如何提高分类性能？
**A**: 考虑以下优化：
- 增加训练数据量
- 调整LoRA配置（r, alpha）
- 尝试不同的池化策略
- 数据增强技术

### Q4: 如何部署模型？
**A**: 参考以下步骤：
1. 保存最佳模型检查点
2. 使用 `test_saved_lora.py` 验证加载
3. 编写推理接口
4. 考虑模型量化加速

## 更新日志

### v1.0.0 (当前版本)
- 实现基于Qwen3-1.7B的LoRA微调
- 支持6类文本安全分类
- 提供完整的训练和评估流程
- 包含数据提取和预处理工具
- 支持分布式训练和混合精度

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 请确保在使用前仔细阅读所有文档，并根据实际环境调整配置参数。
