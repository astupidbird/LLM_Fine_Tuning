import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

class CustomModel(nn.Module):
  def __init__(self, checkpoint, num_labels=6): 
    super(CustomModel, self).__init__() 
    self.num_labels = num_labels 
 
    # 加载 Qwen3 模型并提取其基础模型
    self.base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_attentions=True,
        output_hidden_states=True
    )
    
    # 获取模型配置以确定隐藏层维度
    config = self.base_model.config
    self.hidden_size = config.hidden_size  # Qwen3-1.7B 的隐藏维度是 2048
    
    # 分类层
    self.dropout = nn.Dropout(0.1)
    self.classifier = nn.Linear(self.hidden_size, num_labels)  # 2048 -> 6
    
    # 将分类器移到与基础模型相同的设备上
    self.classifier = self.classifier.to(self.base_model.device)
 
  def forward(self, input_ids=None, attention_mask=None, labels=None):
    # 获取基础模型的输出，包括隐藏状态
    outputs = self.base_model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        output_hidden_states=True
    )
 
    # 获取最后一层的隐藏状态
    hidden_states = outputs.hidden_states[-1]  # 形状: [batch_size, seq_len, hidden_size]
    
    # 使用 [CLS] token 的表示（第一个 token）进行分类
    # 或者可以使用最后一个非填充 token 的表示
    cls_hidden_state = hidden_states[:, 0, :]  # 形状: [batch_size, hidden_size]
    
    # 应用 dropout 和分类器
    pooled_output = self.dropout(cls_hidden_state)
    logits = self.classifier(pooled_output)  # 形状: [batch_size, num_labels]
    
    # 计算损失
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    # 返回输出
    return SequenceClassifierOutputWithPast(
        loss=loss, 
        logits=logits, 
        past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
    )