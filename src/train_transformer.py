"""
Transformer模型训练脚本
支持架构消融实验（位置编码、归一化方法）
支持超参数敏感性分析
"""

import argparse
import sys
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
    from .transformer_model import Transformer
    from .metrics import calculate_bleu
except ImportError:
    from data_utils import (
        prepare_data, TranslationDataset, collate_fn,
        PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, tokenize_en
    )
    from transformer_model import Transformer
    from metrics import calculate_bleu


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
    scheduler=None,
    global_step: int = 0
):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    current_step = global_step
    
    for batch in tqdm(dataloader, desc="Training"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播（使用teacher forcing）
        tgt_input = tgt[:, :-1]  # 去掉最后一个token
        tgt_output = tgt[:, 1:]  # 去掉第一个token
        
        # 生成掩码
        src_mask, tgt_mask = model.generate_mask(src, tgt_input)
        
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        # 计算损失
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        # 更新学习率（如果使用warmup调度器）
        if scheduler is not None and hasattr(scheduler, 'step'):
            # 检查是否是LambdaLR（warmup调度器）
            if isinstance(scheduler, optim.lr_scheduler.LambdaLR):
                scheduler.step()
            # 如果是ReduceLROnPlateau，不在这里调用
        
        total_loss += loss.item()
        current_step += 1
    
    return total_loss / len(dataloader), current_step


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
    src_mask, _ = model.generate_mask(src)
    encoder_output = model.encoder(src, src_mask)
    
    # 初始输入
    sos_idx = tgt_vocab.word2idx[SOS_TOKEN]
    eos_idx = tgt_vocab.word2idx[EOS_TOKEN]
    tgt = torch.full((batch_size, 1), sos_idx, device=device)
    
    for step in range(max_length):
        # 生成掩码
        _, tgt_mask = model.generate_mask(src, tgt)
        
        # 解码
        with torch.no_grad():
            decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
            output = model.output_proj(decoder_output)
        
        # 获取下一个词
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # 检查是否全部生成EOS
        if (next_token == eos_idx).all():
            break
    
    return tgt[:, 1:].cpu().tolist()  # 去掉SOS token


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    tgt_vocab,
    beam_size: int = 5,
    max_length: int = 50,
    device: torch.device = None
):
    """束搜索解码（简化版本，逐个样本处理）"""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = src.size(0)
    results = []
    
    # 逐个样本处理（简化实现）
    for b_idx in range(batch_size):
        src_single = src[b_idx:b_idx+1].to(device)  # [1, src_len]
        
        # 编码
        src_mask, _ = model.generate_mask(src_single)
        encoder_output = model.encoder(src_single, src_mask)  # [1, src_len, d_model]
        
        sos_idx = tgt_vocab.word2idx[SOS_TOKEN]
        eos_idx = tgt_vocab.word2idx[EOS_TOKEN]
        
        # 初始化束
        beams = [{
            'sequence': [sos_idx],
            'score': 0.0,
            'finished': False
        }]
        
        for step in range(max_length):
            if all(beam['finished'] for beam in beams):
                break
            
            candidates = []
            for beam in beams:
                if beam['finished']:
                    candidates.append(beam)
                    continue
                
                # 准备输入
                tgt_seq = torch.tensor([beam['sequence']], device=device)  # [1, seq_len]
                _, tgt_mask = model.generate_mask(src_single, tgt_seq)
                
                # 解码
                with torch.no_grad():
                    decoder_output = model.decoder(tgt_seq, encoder_output, src_mask, tgt_mask)
                    output = model.output_proj(decoder_output)  # [1, seq_len, vocab_size]
                
                # 获取最后一个位置的log概率
                log_probs = torch.log_softmax(output[0, -1, :], dim=-1)  # [vocab_size]
                
                # 获取top-k候选
                top_k_scores, top_k_indices = torch.topk(log_probs, min(beam_size, len(log_probs)))
                
                for score, idx in zip(top_k_scores, top_k_indices):
                    new_seq = beam['sequence'] + [idx.item()]
                    new_score = beam['score'] + score.item()
                    finished = (idx.item() == eos_idx)
                    
                    candidates.append({
                        'sequence': new_seq,
                        'score': new_score,
                        'finished': finished
                    })
            
            # 选择top-k（使用长度归一化）
            candidates.sort(key=lambda x: x['score'] / max(len(x['sequence']), 1), reverse=True)
            beams = candidates[:beam_size]
        
        # 选择最佳序列
        best_beam = max(beams, key=lambda x: x['score'] / max(len(x['sequence']), 1))
        # 去掉SOS和EOS
        sequence = [idx for idx in best_beam['sequence'][1:] 
                   if idx != eos_idx and idx != tgt_vocab.word2idx[PAD_TOKEN]]
        results.append(sequence)
    
    return results


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
            eos_idx = tgt_vocab.word2idx[EOS_TOKEN]
            pad_idx = tgt_vocab.word2idx[PAD_TOKEN]
            
            for seq in sequences:
                # 移除EOS和PAD
                clean_seq = []
                for idx in seq:
                    if idx == eos_idx:
                        break
                    if idx != pad_idx:
                        clean_seq.append(idx)
                
                hyp_words = tgt_vocab.decode(clean_seq)
                all_hypotheses.append(hyp_words)
            
            # 准备参考翻译
            for text in tgt_text:
                ref_words = tokenize_en(text)
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
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--pos_encoding', type=str, default='absolute',
                       choices=['absolute', 'relative'])
    parser.add_argument('--norm_type', type=str, default='layernorm',
                       choices=['layernorm', 'rmsnorm'])
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--min_freq', type=int, default=2)
    parser.add_argument('--bpe_vocab_size', type=int, default=30000, help='BPE词汇表大小')
    parser.add_argument('--bpe_model_path', type=str, default='checkpoints/bpe_tokenizer.json', help='BPE模型保存/加载路径')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup步数')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='学习率调度器patience')
    parser.add_argument('--scheduler_factor', type=float, default=0.7, help='学习率调度器衰减因子')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--decode_method', type=str, default='greedy',
                       choices=['greedy', 'beam'], help='解码方法')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam Search的beam size')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检查命令行中是否提供了特定参数
    def is_arg_provided(arg_name):
        """检查命令行中是否提供了某个参数"""
        # 检查短参数和长参数
        short_arg = f'-{arg_name[0]}' if len(arg_name) > 0 else None
        long_arg = f'--{arg_name}'
        for i, arg in enumerate(sys.argv):
            if arg == long_arg or (short_arg and arg == short_arg):
                # 检查下一个参数是否是值（不是另一个参数）
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                    return True
                # 或者参数本身是布尔标志
                return True
        return False
    
    # 先解析命令行参数
    args = parser.parse_args()
    
    # 如果提供了配置文件，加载配置文件
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 加载配置文件的值到args
            for k, v in config.items():
                # 如果命令行中没有提供该参数，才使用配置文件的值
                if not is_arg_provided(k):
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
    model = Transformer(
        len(src_vocab),
        len(tgt_vocab),
        args.d_model,
        args.n_heads,
        args.n_layers,
        args.d_ff,
        args.max_length * 2,
        pos_encoding=args.pos_encoding,
        norm_type=args.norm_type
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.word2idx[PAD_TOKEN])
    
    # 学习率调度器：使用warmup + ReduceLROnPlateau的组合
    warmup_steps = getattr(args, 'warmup_steps', 4000)
    total_steps = len(train_loader) * args.epochs
    
    # Warmup学习率调度器
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup阶段：线性增长
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Warmup后：保持1.0（由ReduceLROnPlateau控制）
            return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 验证集上的学习率调度器（在warmup后使用）
    # 确保 min_lr 是浮点数类型
    min_lr_value = getattr(args, 'min_lr', 1e-6)
    if isinstance(min_lr_value, str):
        min_lr_value = float(min_lr_value)
    
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=float(getattr(args, 'scheduler_factor', 0.7)),
        patience=int(getattr(args, 'scheduler_patience', 5)),
        min_lr=min_lr_value,
        verbose=True
    )
    
    # 使用warmup调度器作为主调度器
    scheduler = warmup_scheduler
    
    # 训练进度承接
    best_bleu = 0
    os.makedirs(args.save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(args.save_dir, 'transformer_best.pt')
    results_path = os.path.join(args.save_dir, 'transformer_results.json')
    start_epoch = 0
    global_step_start = 0
    if os.path.exists(checkpoint_path):
        print(f"发现已有模型 {checkpoint_path}，尝试加载...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            # 尝试加载权重
            model.load_state_dict(checkpoint['model_state_dict'])
            # 如果成功加载，读取最佳 BLEU
            if 'best_valid_bleu' in checkpoint:
                best_bleu = checkpoint['best_valid_bleu']
            elif os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    best_bleu = results.get('best_valid_bleu', 0)
            # 恢复训练步数
            if 'global_step' in checkpoint:
                global_step_start = checkpoint['global_step']
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            print(f"✓ 成功加载模型，当前最佳 Valid BLEU: {best_bleu:.4f}")
            if global_step_start > 0:
                print(f"  从第 {start_epoch} 个epoch，步数 {global_step_start} 继续训练")
        except (RuntimeError, KeyError) as e:
            error_msg = str(e)
            if 'size mismatch' in error_msg or 'Missing key' in error_msg or 'Unexpected key' in error_msg:
                print(f"\n⚠ 警告: 检测到模型架构已改变（可能是修改了 d_model 或 n_layers）")
                print(f"   旧模型的权重形状与新架构不匹配，无法加载。")
                print(f"   将从零开始重新训练新架构的模型...\n")
                best_bleu = 0
            else:
                # 如果是其他错误，重新抛出
                print(f"加载模型时发生错误: {error_msg}")
                raise
    
    print(f"位置编码: {args.pos_encoding}")
    print(f"归一化方法: {args.norm_type}")
    print(f"模型参数: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"训练配置: batch_size={args.batch_size}, lr={args.lr}, warmup_steps={warmup_steps}")
    print(f"总训练步数: {total_steps}, 每个epoch步数: {len(train_loader)}")
    
    # 如果从checkpoint恢复，需要将warmup调度器快进到正确的步数
    global_step = global_step_start
    if global_step_start > 0 and global_step_start < warmup_steps:
        # 快进warmup调度器到当前步数
        for _ in range(global_step_start):
            warmup_scheduler.step()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练（返回步数）
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=getattr(args, 'clip_grad', 1.0),
            scheduler=scheduler,
            global_step=global_step
        )
        print(f"Train Loss: {train_loss:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f} (Step: {global_step}/{total_steps})")
        
        # 验证
        decode_method = getattr(args, 'decode_method', 'greedy')
        beam_size = getattr(args, 'beam_size', 5)
        valid_bleu = evaluate(model, valid_loader, tgt_vocab, device, decode_method, beam_size)
        print(f"Valid BLEU: {valid_bleu:.4f}")
        
        # 学习率调度
        # 如果还在warmup阶段，warmup_scheduler已经在train_epoch中更新
        # 如果warmup结束，使用plateau_scheduler
        if global_step >= warmup_steps:
            plateau_scheduler.step(valid_bleu)
        
        # 保存最佳模型
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            torch.save({
                'model_state_dict': model.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'args': vars(args),
                'best_valid_bleu': best_bleu,
                'global_step': global_step,
                'epoch': epoch + 1
            }, checkpoint_path)
            print(f"✓ 保存最佳模型，BLEU: {best_bleu:.4f}")
    
    # 测试
    print("\n在测试集上评估...")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        decode_method = getattr(args, 'decode_method', 'greedy')
        beam_size = getattr(args, 'beam_size', 5)
        test_bleu = evaluate(model, test_loader, tgt_vocab, device, decode_method, beam_size)
        print(f"Test BLEU: {test_bleu:.4f}")
        
        # 保存结果
        results = {
            'best_valid_bleu': best_bleu,
            'test_bleu': test_bleu,
            'args': vars(args)
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        print("未发现最佳模型文件，跳过测试。")


if __name__ == '__main__':
    main()
