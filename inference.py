"""
一键推理脚本
支持RNN、Transformer和T5模型的推理
"""

import argparse
import torch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_utils import tokenize_zh, tokenize_en, Vocabulary, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from rnn_model import RNNSeq2Seq
from transformer_model import Transformer
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_rnn_model(checkpoint_path, device):
    """加载RNN模型"""
    # PyTorch 2.6+ 默认 weights_only=True，这里需要加载自定义类，显式关闭
    # 如果指定了cuda但cuda不可用，自动降级到cpu
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"警告: CUDA不可用，使用CPU加载模型")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    model = RNNSeq2Seq(
        len(src_vocab),
        len(tgt_vocab),
        args.get('embed_dim', 256),
        args.get('hidden_dim', 512),
        args.get('num_layers', 2),
        args.get('rnn_type', 'gru'),
        args.get('attention_type', 'dot')
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, src_vocab, tgt_vocab


def load_transformer_model(checkpoint_path, device):
    """加载Transformer模型"""
    # 同上，显式关闭 weights_only 以允许加载自定义类
    # 如果指定了cuda但cuda不可用，自动降级到cpu
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"警告: CUDA不可用，使用CPU加载模型")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    model = Transformer(
        len(src_vocab),
        len(tgt_vocab),
        args.get('d_model', 512),
        args.get('n_heads', 8),
        args.get('n_layers', 6),
        args.get('d_ff', 2048),
        args.get('max_length', 50) * 2,
        pos_encoding=args.get('pos_encoding', 'absolute'),
        norm_type=args.get('norm_type', 'layernorm')
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, src_vocab, tgt_vocab


def load_t5_model(checkpoint_path, device):
    """加载T5模型"""
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    model.eval()
    return model, tokenizer


def rnn_inference(model, src_vocab, tgt_vocab, text, device, max_length=50):
    """RNN模型推理"""
    from data_utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
    
    # 分词和编码
    src_tokens = tokenize_zh(text)
    src_ids = [src_vocab.word2idx.get(w, src_vocab.word2idx[UNK_TOKEN]) for w in src_tokens]
    src_ids = [src_vocab.word2idx[SOS_TOKEN]] + src_ids + [src_vocab.word2idx[EOS_TOKEN]]
    
    # 填充到max_length
    if len(src_ids) > max_length:
        src_ids = src_ids[:max_length]
    else:
        src_ids = src_ids + [src_vocab.word2idx[PAD_TOKEN]] * (max_length - len(src_ids))
    
    src_tensor = torch.LongTensor([src_ids]).to(device)
    
    # 编码
    encoder_outputs, encoder_hidden = model.encoder(src_tensor)
    
    # 贪婪解码 (使用模型内部的Free Running模式)
    sos_idx = tgt_vocab.word2idx[SOS_TOKEN]
    with torch.no_grad():
        decoder_outputs, _, _ = model.decoder(
            None,  # tgt=None 触发自回归生成
            encoder_outputs,
            encoder_hidden,
            teacher_forcing=False,
            max_length=max_length,
            sos_idx=sos_idx
        )
    
    # 获取预测的token序列
    predictions = decoder_outputs.argmax(dim=-1)[0].cpu().tolist()
    
    sequences = []
    for token in predictions:
        if token == tgt_vocab.word2idx[EOS_TOKEN]:
            break
        sequences.append(token)
    
    # 解码为文本
    tgt_words = tgt_vocab.decode(sequences)
    return ' '.join(tgt_words)


def transformer_inference(model, src_vocab, tgt_vocab, text, device, max_length=50):
    """Transformer模型推理"""
    # 分词和编码
    src_tokens = tokenize_zh(text)
    src_ids = [src_vocab.word2idx.get(w, src_vocab.word2idx[UNK_TOKEN]) for w in src_tokens]
    src_ids = [src_vocab.word2idx[SOS_TOKEN]] + src_ids + [src_vocab.word2idx[EOS_TOKEN]]
    
    # 填充到max_length
    if len(src_ids) > max_length:
        src_ids = src_ids[:max_length]
    else:
        src_ids = src_ids + [src_vocab.word2idx[PAD_TOKEN]] * (max_length - len(src_ids))
    
    src_tensor = torch.LongTensor([src_ids]).to(device)
    
    # 生成掩码
    src_mask, _ = model.generate_mask(src_tensor)
    
    # 编码
    encoder_output = model.encoder(src_tensor, src_mask)
    
    # 贪婪解码
    batch_size = 1
    tgt = torch.full((batch_size, 1), tgt_vocab.word2idx[SOS_TOKEN], device=device)
    
    for step in range(max_length):
        _, tgt_mask = model.generate_mask(src_tensor, tgt)
        
        with torch.no_grad():
            decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
            output = model.output_proj(decoder_output)
        
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_token.item() == tgt_vocab.word2idx[EOS_TOKEN]:
            break
        
        tgt = torch.cat([tgt, next_token], dim=1)
    
    # 解码为文本
    tgt_ids = tgt[0, 1:].cpu().tolist()  # 去掉SOS
    tgt_ids = [idx for idx in tgt_ids if idx != tgt_vocab.word2idx[EOS_TOKEN] and idx != tgt_vocab.word2idx[PAD_TOKEN]]
    tgt_words = tgt_vocab.decode(tgt_ids)
    return ' '.join(tgt_words)


def t5_inference(model, tokenizer, text, device):
    """T5模型推理"""
    source = f"translate Chinese to English: {text}"
    
    input_ids = tokenizer.encode(source, return_tensors='pt', max_length=128, truncation=True).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return translation


def main():
    parser = argparse.ArgumentParser(description='机器翻译推理脚本')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rnn', 'transformer', 't5'],
                       help='模型类型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入文本（中文）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--max_length', type=int, default=50,
                       help='最大生成长度')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    # 不直接打印输入文本，避免编码错误
    print("-" * 50)
    
    # 加载模型
    if args.model_type == 'rnn':
        model, src_vocab, tgt_vocab = load_rnn_model(args.checkpoint, device)
        translation = rnn_inference(model, src_vocab, tgt_vocab, args.input, device, args.max_length)
    elif args.model_type == 'transformer':
        model, src_vocab, tgt_vocab = load_transformer_model(args.checkpoint, device)
        translation = transformer_inference(model, src_vocab, tgt_vocab, args.input, device, args.max_length)
    elif args.model_type == 't5':
        model, tokenizer = load_t5_model(args.checkpoint, device)
        translation = t5_inference(model, tokenizer, args.input, device)
    
    # 使用 utf-8 编码输出
    try:
        print(f"Translation: {translation}")
    except UnicodeEncodeError:
        print(f"Translation: {translation.encode('utf-8')}")


if __name__ == '__main__':
    main()
