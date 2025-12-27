"""
RNN-based NMT模型实现
支持GRU和LSTM，编码器-解码器架构，带注意力机制
"""

import torch
import torch.nn as nn

# 同时兼容包内调用（src.）和脚本直接调用（通过sys.path加入src）
try:
    from .attention import get_attention
except ImportError:  # 作为普通脚本被引用时
    from attention import get_attention


class Encoder(nn.Module):
    """编码器"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = 'gru',
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor):
        """
        Args:
            src: [batch_size, src_len]
        Returns:
            outputs: [batch_size, src_len, hidden_dim]
            hidden: 隐藏状态
        """
        embedded = self.embedding(src)  # [batch_size, src_len, embed_dim]
        embedded = self.dropout(embedded)
        
        outputs, hidden = self.rnn(embedded)
        outputs = self.dropout(outputs)
        
        return outputs, hidden


class Decoder(nn.Module):
    """解码器（带注意力机制）"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = 'gru',
        attention_type: str = 'dot',
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                embed_dim + hidden_dim,  # 输入包含embedding和attention context
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_dim + hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        self.attention = get_attention(attention_type, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)  # hidden + context
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        encoder_outputs: torch.Tensor,
        hidden: tuple,
        teacher_forcing: bool = True,
        max_length: int = 50,
        sos_idx: int = 2
    ):
        """
        Args:
            tgt: [batch_size, tgt_len] (训练时使用，teacher forcing)
            encoder_outputs: [batch_size, src_len, hidden_dim]
            hidden: 编码器隐藏状态
            teacher_forcing: 是否使用teacher forcing
            max_length: 最大生成长度
            sos_idx: SOS token的索引，默认为2
        Returns:
            outputs: [batch_size, tgt_len, vocab_size]
            hidden: 最后的隐藏状态
            attentions: 注意力权重
        """
        batch_size = tgt.shape[0] if tgt is not None else encoder_outputs.shape[0]
        device = encoder_outputs.device
        
        # 创建mask（忽略padding）
        src_mask = (encoder_outputs.sum(dim=-1) != 0).unsqueeze(1)  # [batch_size, 1, src_len]
        
        if teacher_forcing and tgt is not None:
            # Teacher Forcing: 使用真实目标序列
            tgt_len = tgt.shape[1]
            embedded = self.embedding(tgt)  # [batch_size, tgt_len, embed_dim]
            embedded = self.dropout(embedded)
            
            outputs = []
            attentions = []
            
            for t in range(tgt_len):
                # 当前输入
                input_t = embedded[:, t:t+1, :]  # [batch_size, 1, embed_dim]
                
                # 注意力
                query = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
                query = query.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                
                context, attn_weights = self.attention(
                    query, encoder_outputs, encoder_outputs, src_mask
                )
                
                # 拼接输入和context
                rnn_input = torch.cat([input_t, context], dim=-1)  # [batch_size, 1, embed_dim + hidden_dim]
                
                # RNN
                rnn_output, hidden = self.rnn(rnn_input, hidden)
                
                # 输出投影
                output = torch.cat([rnn_output, context], dim=-1)  # [batch_size, 1, hidden_dim * 2]
                output = self.output_proj(output)  # [batch_size, 1, vocab_size]
                
                outputs.append(output)
                attentions.append(attn_weights)
            
            outputs = torch.cat(outputs, dim=1)  # [batch_size, tgt_len, vocab_size]
            attentions = torch.cat(attentions, dim=1)  # [batch_size, tgt_len, src_len]
            
        else:
            # Free Running: 自回归生成
            outputs = []
            attentions = []
            
            # 初始输入：如果传入了tgt，使用tgt的最后一个token；否则使用SOS token
            if tgt is not None and tgt.shape[1] > 0:
                # 使用传入的tgt的最后一个token
                input_t = self.embedding(tgt[:, -1:])  # [batch_size, 1, embed_dim]
            else:
                # 使用SOS token
                input_t = self.embedding(torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device))
            
            for t in range(max_length):
                # 注意力
                query = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
                query = query.unsqueeze(1)
                
                context, attn_weights = self.attention(
                    query, encoder_outputs, encoder_outputs, src_mask
                )
                
                # 拼接输入和context
                rnn_input = torch.cat([input_t, context], dim=-1)
                
                # RNN
                rnn_output, hidden = self.rnn(rnn_input, hidden)
                
                # 输出投影
                output = torch.cat([rnn_output, context], dim=-1)
                output = self.output_proj(output)  # [batch_size, 1, vocab_size]
                
                outputs.append(output)
                attentions.append(attn_weights)
                
                # 下一个输入是当前输出的argmax
                next_token = output.argmax(dim=-1)
                input_t = self.embedding(next_token)
            
            outputs = torch.cat(outputs, dim=1)
            attentions = torch.cat(attentions, dim=1)
        
        return outputs, hidden, attentions


class RNNSeq2Seq(nn.Module):
    """RNN-based Seq2Seq模型"""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        rnn_type: str = 'gru',
        attention_type: str = 'dot',
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_dim,
            hidden_dim,
            num_layers,
            rnn_type,
            dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            embed_dim,
            hidden_dim,
            num_layers,
            rnn_type,
            attention_type,
            dropout
        )
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor = None,
        teacher_forcing: bool = True,
        max_length: int = 50,
        sos_idx: int = 2
    ):
        """
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len] (可选，用于训练)
            teacher_forcing: 是否使用teacher forcing
            max_length: 最大生成长度
            sos_idx: SOS token的索引
        """
        # 编码
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # 解码
        decoder_outputs, final_hidden, attentions = self.decoder(
            tgt,
            encoder_outputs,
            encoder_hidden,
            teacher_forcing,
            max_length,
            sos_idx
        )
        
        return decoder_outputs, attentions
