import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import time
import os

class Qwen3ForSequenceClassification(nn.Module):
    """
    直接修改Qwen3模型的lm_head层，将其从语言模型头（151936个词元）
    转换为分类头（6个类别）
    """
    def __init__(self, checkpoint, num_labels=6):
        super().__init__()
        
        # 加载预训练的Qwen3模型
        print("正在加载Qwen3模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 保存配置信息
        self.config = self.base_model.config
        self.num_labels = num_labels
        self.hidden_size = self.config.hidden_size  # 2048
        
        # 直接替换lm_head层
        # 原始: Linear(2048, 151936, bias=False)
        # 新的: Linear(2048, 6, bias=False)
        print(f"替换lm_head层: {self.hidden_size} -> {num_labels}")
        self.base_model.lm_head = nn.Linear(
            self.hidden_size,
            num_labels,
            bias=False
        )
        
        # 初始化新的分类头权重
        nn.init.xavier_uniform_(self.base_model.lm_head.weight)
        
        # 添加dropout层用于正则化
        self.dropout = nn.Dropout(0.1)
        
        # 将新层移到正确的设备和数据类型
        device = next(self.base_model.parameters()).device
        dtype = next(self.base_model.parameters()).dtype
        self.base_model.lm_head = self.base_model.lm_head.to(device=device, dtype=dtype)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs  # 接受其他可能的参数
    ):
        # 获取Qwen3模型的基础输出（不包括lm_head）
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 确保输出隐藏状态
            return_dict=True
        )
        
        # 获取最后一层的隐藏状态
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 方法1：使用序列的最后一个有效token（考虑padding）
        if attention_mask is not None:
            # 找到每个序列的最后一个有效token位置
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            sequence_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]
        else:
            # 如果没有attention_mask，使用最后一个token
            sequence_hidden_states = hidden_states[:, -1, :]
            
        # 方法2：使用平均池化（可选）
        # if attention_mask is not None:
        #     mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        #     sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        #     sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        #     sequence_hidden_states = sum_hidden / sum_mask
        # else:
        #     sequence_hidden_states = hidden_states.mean(dim=1)
        
        # 应用dropout
        pooled_output = self.dropout(sequence_hidden_states)
        
        # 通过新的分类头获取logits
        logits = self.base_model.lm_head(pooled_output)  # [batch_size, num_labels]
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # 返回分类输出
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None
        )