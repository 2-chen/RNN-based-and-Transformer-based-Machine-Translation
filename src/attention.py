"""
注意力机制实现
包括点积、乘法、加法三种对齐函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductAttention(nn.Module):
    """点积注意力"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, tgt_len, hidden_dim]
            key: [batch_size, src_len, hidden_dim]
            value: [batch_size, src_len, hidden_dim]
            mask: [batch_size, tgt_len, src_len]
        """
        # 计算注意力分数
        scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, tgt_len, src_len]
        scores = scores / math.sqrt(self.hidden_dim)  # 缩放
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, tgt_len, src_len]
        
        # 应用注意力权重
        context = torch.bmm(attn_weights, value)  # [batch_size, tgt_len, hidden_dim]
        
        return context, attn_weights


class MultiplicativeAttention(nn.Module):
    """乘法注意力（General）"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, tgt_len, hidden_dim]
            key: [batch_size, src_len, hidden_dim]
            value: [batch_size, src_len, hidden_dim]
            mask: [batch_size, tgt_len, src_len]
        """
        # 对key进行线性变换
        key_transformed = self.W(key)  # [batch_size, src_len, hidden_dim]
        
        # 计算注意力分数
        scores = torch.bmm(query, key_transformed.transpose(1, 2))  # [batch_size, tgt_len, src_len]
        scores = scores / math.sqrt(self.hidden_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        context = torch.bmm(attn_weights, value)
        
        return context, attn_weights


class AdditiveAttention(nn.Module):
    """加法注意力（Bahdanau）"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, tgt_len, hidden_dim]
            key: [batch_size, src_len, hidden_dim]
            value: [batch_size, src_len, hidden_dim]
            mask: [batch_size, tgt_len, src_len]
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # 扩展维度以便计算所有组合
        query_expanded = self.W_q(query).unsqueeze(2)  # [batch_size, tgt_len, 1, hidden_dim]
        key_expanded = self.W_k(key).unsqueeze(1)  # [batch_size, 1, src_len, hidden_dim]
        
        # 计算注意力分数
        scores = self.v(torch.tanh(query_expanded + key_expanded)).squeeze(-1)  # [batch_size, tgt_len, src_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        context = torch.bmm(attn_weights, value)
        
        return context, attn_weights


def get_attention(attention_type: str, hidden_dim: int) -> nn.Module:
    """获取注意力机制"""
    if attention_type == 'dot':
        return DotProductAttention(hidden_dim)
    elif attention_type == 'multiplicative':
        return MultiplicativeAttention(hidden_dim)
    elif attention_type == 'additive':
        return AdditiveAttention(hidden_dim)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
