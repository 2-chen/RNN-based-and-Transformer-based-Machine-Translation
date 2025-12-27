"""
RNN模型训练脚本
支持Teacher Forcing和Free Running训练策略
支持贪婪解码和束搜索解码
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import yaml

try:
    from .data_utils import (
        prepare_data, TranslationDataset, collate_fn,
        PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, tokenize_en
    )
    from .rnn_model import RNNSeq2Seq
    from .metrics import calculate_bleu
except ImportError:
    from data_utils import (
        prepare_data, TranslationDataset, collate_fn,
        PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, tokenize_en
    )
    from rnn_model import RNNSeq2Seq
    from metrics import calculate_bleu


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    tgt_vocab,
    beam_size: int = 5,
    max_length: int = 50,
    device: torch.device = None
):
    """束搜索解码（逐个样本处理）"""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = src.size(0)
    src = src.to(device)
    
    # 逐个样本处理
    results = []
    for b_idx in range(batch_size):
        src_single = src[b_idx:b_idx+1].to(device)  # [1, src_len]
        
        # 编码
        encoder_outputs, encoder_hidden = model.encoder(src_single)
        
        sos_idx = tgt_vocab.word2idx[SOS_TOKEN]
        eos_idx = tgt_vocab.word2idx[EOS_TOKEN]
        
        # 初始化束
        beams = [{
            'sequence': [sos_idx],
            'score': 0.0,
            'hidden': encoder_hidden
        }]
        
        for step in range(max_length):
            if all(beam['sequence'][-1] == eos_idx for beam in beams):
                break
            
            candidates = []
            for beam in beams:
                if beam['sequence'][-1] == eos_idx:
                    candidates.append(beam)
                    continue
                
                # 准备输入（只使用最后一个token）
                last_token = torch.LongTensor([[beam['sequence'][-1]]]).to(device)  # [1, 1]
                
                # 解码一步
                with torch.no_grad():
                    decoder_outputs, next_hidden, _ = model.decoder(
                        last_token,
                        encoder_outputs,
                        beam['hidden'],
                        teacher_forcing=False,
                        max_length=1
                    )
                
                # 获取top-k候选
                log_probs = torch.log_softmax(decoder_outputs[0, -1], dim=-1)  # [vocab_size]
                top_k_scores, top_k_indices = torch.topk(log_probs, min(beam_size, len(log_probs)))
                
                for score, idx in zip(top_k_scores, top_k_indices):
                    new_seq = beam['sequence'] + [idx.item()]
                    new_score = beam['score'] + score.item()
                    
                    candidates.append({
                        'sequence': new_seq,
                        'score': new_score,
                        'hidden': next_hidden  # 使用更新后的hidden状态
                    })
            
            # 选择top-k束（使用长度归一化）
            candidates.sort(key=lambda x: x['score'] / max(len(x['sequence']), 1), reverse=True)
            beams = candidates[:beam_size]
        
        # 返回最佳序列（去掉SOS）
        best_beam = max(beams, key=lambda x: x['score'] / max(len(x['sequence']), 1))
        results.append(best_beam['sequence'][1:])  # 去掉SOS token
    
    return results


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    tgt_vocab,
    max_length: int = 50,
    device: torch.device = None
):
    """贪婪解码"""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = src.size(0)
    src = src.to(device)
    
    # 编码
    encoder_outputs, encoder_hidden = model.encoder(src)
    
    # 初始输入
    tgt = torch.full((batch_size, 1), tgt_vocab.word2idx[SOS_TOKEN], device=device)
    
    sequences = [[] for _ in range(batch_size)]
    finished = [False] * batch_size  # 标记每个序列是否已完成
    
    for step in range(max_length):
        # 如果所有序列都完成了，提前退出
        if all(finished):
            break
            
        with torch.no_grad():
            decoder_outputs, encoder_hidden, _ = model.decoder(
                tgt,
                encoder_outputs,
                encoder_hidden,
                teacher_forcing=False,
                max_length=1
            )
        
        # 获取下一个词
        next_tokens = decoder_outputs.argmax(dim=-1)[:, -1]
        tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)
        
        # 收集序列
        for i in range(batch_size):
            if not finished[i]:
                if next_tokens[i].item() != tgt_vocab.word2idx[EOS_TOKEN]:
                    sequences[i].append(next_tokens[i].item())
                else:
                    finished[i] = True  # 标记这个序列已完成
    
    return sequences


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing: bool = True,
    clip_grad: float = 5.0
):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        decoder_outputs, _ = model(src, tgt, teacher_forcing=teacher_forcing)
        
        # 计算损失（对齐时间步：用 t 时刻预测 t+1 时刻）
        # decoder_outputs 包含从每个输入得到的预测
        # outputs[0] 是从 SOS 预测的，对应 tgt[1]
        # outputs[tgt_len-2] 是从最后一个词预测的，对应 tgt[tgt_len-1] (EOS)
        decoder_outputs = decoder_outputs[:, :-1, :].contiguous()
        decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
        tgt_flat = tgt[:, 1:].contiguous().view(-1)  # 去掉第一个 token (SOS)
        
        loss = criterion(decoder_outputs, tgt_flat)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    tgt_vocab,
    device: torch.device,
    decode_method: str = 'greedy',
    beam_size: int = 5
):
    """评估模型"""
    model.eval()
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch['src'].to(device)
            tgt_text = batch['tgt_text']
            
            # 解码
            if decode_method == 'greedy':
                sequences = greedy_decode(model, src, tgt_vocab, device=device)
            elif decode_method == 'beam':
                sequences = beam_search_decode(model, src, tgt_vocab, beam_size, device=device)
            else:
                raise ValueError(f"Unknown decode method: {decode_method}")
            
            # 转换为文本
            for i, seq in enumerate(sequences):
                # 移除SOS和EOS（beam_search_decode已经去掉了SOS，但可能还有EOS）
                seq = [idx for idx in seq if idx not in [
                    tgt_vocab.word2idx[SOS_TOKEN],
                    tgt_vocab.word2idx[EOS_TOKEN],
                    tgt_vocab.word2idx[PAD_TOKEN]
                ]]
                hyp_words = tgt_vocab.decode(seq)
                all_hypotheses.append(hyp_words)
            
            # 准备参考翻译（使用和训练时相同的分词方法）
            for text in tgt_text:
                ref_words = tokenize_en(text)  # 使用和训练时相同的分词方法
                all_references.append(ref_words)
    
    # 计算BLEU
    bleu_score = calculate_bleu(all_references, all_hypotheses)
    return bleu_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--train_file', type=str, default='data/train_10k.jsonl')
    parser.add_argument('--valid_file', type=str, default='data/valid.jsonl')
    parser.add_argument('--test_file', type=str, default='data/test.jsonl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--rnn_type', type=str, default='gru', choices=['gru', 'lstm'])
    parser.add_argument('--attention_type', type=str, default='dot',
                       choices=['dot', 'multiplicative', 'additive'])
    parser.add_argument('--teacher_forcing', action='store_true', default=True)
    parser.add_argument('--free_running', action='store_true', default=False)
    parser.add_argument('--decode_method', type=str, default='greedy',
                       choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--min_freq', type=int, default=2, help='词汇表最小词频')
    parser.add_argument('--bpe_vocab_size', type=int, default=30000, help='BPE词汇表大小')
    parser.add_argument('--bpe_model_path', type=str, default='checkpoints/bpe_tokenizer.json', help='BPE模型保存/加载路径')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2正则化')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--use_scheduler', action='store_true', help='使用学习率调度器')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='学习率调度器patience')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='学习率衰减因子')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，加载并覆盖默认参数
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                setattr(args, k, v)
    
    device = torch.device(args.device)
    
    # 准备数据
    print("准备数据...")
    src_vocab, tgt_vocab, train_data, valid_data, test_data = prepare_data(
        args.train_file, args.valid_file, args.test_file, args.max_length, args.min_freq,
        bpe_vocab_size=getattr(args, 'bpe_vocab_size', 30000),
        bpe_model_path=getattr(args, 'bpe_model_path', None)
    )
    
    # 创建数据集
    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, args.max_length)
    valid_dataset = TranslationDataset(valid_data, src_vocab, tgt_vocab, args.max_length)
    test_dataset = TranslationDataset(test_data, src_vocab, tgt_vocab, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    model = RNNSeq2Seq(
        len(src_vocab),
        len(tgt_vocab),
        args.embed_dim,
        args.hidden_dim,
        args.num_layers,
        args.rnn_type,
        args.attention_type,
        args.dropout
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"源语言词汇表大小: {len(src_vocab):,}")
    print(f"目标语言词汇表大小: {len(tgt_vocab):,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.word2idx[PAD_TOKEN])
    
    # 学习率调度器
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.scheduler_factor, 
            patience=args.scheduler_patience
        )
    
    # 训练
    best_bleu = 0
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查是否存在已有模型并加载
    checkpoint_path = os.path.join(args.save_dir, 'rnn_best.pt')
    results_path = os.path.join(args.save_dir, 'rnn_results.json')
    if os.path.exists(checkpoint_path):
        print(f"发现已有模型 {checkpoint_path}，尝试加载...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 承接之前的最佳 BLEU 分数
            if 'best_valid_bleu' in checkpoint:
                best_bleu = checkpoint['best_valid_bleu']
            elif os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    best_bleu = results.get('best_valid_bleu', 0)
            
            print(f"✓ 成功加载模型，当前最佳 Valid BLEU: {best_bleu:.4f}")
        except (RuntimeError, KeyError) as e:
            error_msg = str(e)
            if 'size mismatch' in error_msg or 'Missing key' in error_msg or 'Unexpected key' in error_msg:
                print(f"\n⚠ 警告: 检测到模型架构已改变（可能是修改了分词方法或词汇表）")
                print(f"   旧模型的权重形状与新架构不匹配，无法加载。")
                print(f"   将从零开始重新训练新架构的模型...\n")
                best_bleu = 0
            else:
                # 如果是其他错误，重新抛出
                print(f"加载模型时发生错误: {error_msg}")
                raise
    
    # 确定训练策略
    use_teacher_forcing = args.teacher_forcing and not args.free_running
    
    print(f"训练策略: {'Teacher Forcing' if use_teacher_forcing else 'Free Running'}")
    print(f"解码方法: {args.decode_method}")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, use_teacher_forcing, args.clip_grad
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        valid_bleu = evaluate(model, valid_loader, tgt_vocab, device, args.decode_method, args.beam_size)
        print(f"Valid BLEU: {valid_bleu:.4f}")
        
        # 学习率调度
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(valid_bleu)
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != old_lr:
                print(f"学习率已调整: {old_lr:.6f} -> {current_lr:.6f}")
            else:
                print(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            torch.save({
                'model_state_dict': model.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'args': vars(args),
                'best_valid_bleu': best_bleu  # 保存最佳分数以便续传
            }, os.path.join(args.save_dir, 'rnn_best.pt'))
            print(f"✓ 保存最佳模型，BLEU: {best_bleu:.4f}")
    
    # 测试
    print("\n在测试集上评估...")
    # PyTorch 2.6+ 默认使用 weights_only=True，这里需要加载自定义类，显式关闭
    checkpoint_path = os.path.join(args.save_dir, 'rnn_best.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_bleu = evaluate(model, test_loader, tgt_vocab, device, args.decode_method, args.beam_size)
        print(f"Test BLEU: {test_bleu:.4f}")
        
        # 保存结果
        results = {
            'best_valid_bleu': best_bleu,
            'test_bleu': test_bleu,
            'args': vars(args)
        }
        with open(os.path.join(args.save_dir, 'rnn_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        print("未发现最佳模型文件，跳过测试。")


if __name__ == '__main__':
    main()
