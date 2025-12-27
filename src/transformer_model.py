"""
Transformer-based NMT模型实现
从零开始实现Transformer架构
支持绝对和相对位置编码，LayerNorm和RMSNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """绝对位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """相对位置编码（简化版）"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # 相对位置嵌入
        self.rel_pos_embed = nn.Parameter(torch.randn(2 * max_len - 1, d_model))
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        batch_size = x.size(1)
        
        # 计算相对位置
        positions = torch.arange(seq_len, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
        rel_pos = rel_pos + self.max_len - 1  # 转换为索引
        # 确保索引在有效范围内
        rel_pos = torch.clamp(rel_pos, 0, 2 * self.max_len - 2)
        
        # 获取相对位置嵌入
        rel_pos_emb = self.rel_pos_embed[rel_pos]  # [seq_len, seq_len, d_model]
        
        # 对每个位置，计算其相对于所有其他位置的平均相对位置编码
        # rel_pos_emb.mean(dim=1): [seq_len, d_model] - 对每个位置，平均所有相对位置
        # 然后扩展维度以匹配x的形状
        pos_emb = rel_pos_emb.mean(dim=1, keepdim=True)  # [seq_len, 1, d_model]
        # 扩展以匹配batch维度: [seq_len, 1, d_model] -> [seq_len, batch_size, d_model]
        pos_emb = pos_emb.expand(-1, batch_size, -1)  # [seq_len, batch_size, d_model]
        
        # 添加到输入
        x = x + pos_emb
        
        return self.dropout(x)


class RMSNorm(nn.Module):
    """RMS归一化"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [..., d_model]
        """
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 线性变换并分割为多头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output, attn_weights


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_type: str = 'layernorm'
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_type: str = 'layernorm'
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ):
        # 掩码自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力 + 残差连接
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        pos_encoding: str = 'absolute',
        norm_type: str = 'layernorm'
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        if pos_encoding == 'absolute':
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == 'relative':
            self.pos_encoding = RelativePositionalEncoding(d_model, max_len, dropout)
        else:
            raise ValueError(f"Unknown pos encoding type: {pos_encoding}")
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            src: [batch_size, src_len]
            mask: [batch_size, src_len, src_len]
        """
        # 嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [src_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, src_len, d_model]
        x = self.dropout(x)
        
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        pos_encoding: str = 'absolute',
        norm_type: str = 'layernorm'
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        if pos_encoding == 'absolute':
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == 'relative':
            self.pos_encoding = RelativePositionalEncoding(d_model, max_len, dropout)
        else:
            raise ValueError(f"Unknown pos encoding type: {pos_encoding}")
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ):
        """
        Args:
            tgt: [batch_size, tgt_len]
            encoder_output: [batch_size, src_len, d_model]
            src_mask: [batch_size, tgt_len, src_len]
            tgt_mask: [batch_size, tgt_len, tgt_len]
        """
        # 嵌入 + 位置编码
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [tgt_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, tgt_len, d_model]
        x = self.dropout(x)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x


class Transformer(nn.Module):
    """Transformer模型"""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pos_encoding: str = 'absolute',
        norm_type: str = 'layernorm'
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_layers, d_ff,
            max_len, dropout, pos_encoding, norm_type
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_layers, d_ff,
            max_len, dropout, pos_encoding, norm_type
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ):
        """
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            src_mask: [batch_size, src_len, src_len]
            tgt_mask: [batch_size, tgt_len, tgt_len]
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_proj(decoder_output)
        return output
    
    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor = None):
        """生成掩码"""
        # src_mask: [batch_size, 1, 1, src_len]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        if tgt is not None:
            # tgt_mask: [batch_size, 1, tgt_len, tgt_len]
            tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
            seq_len = tgt.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool()
            tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            tgt_mask = None
        
        return src_mask, tgt_mask
