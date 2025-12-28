#!/usr/bin/env python3
"""
Demo模型训练脚本
训练小规模模型用于快速测试和GitHub上传（文件大小<100MB）

使用方法：
    python train_demo.py --model_type rnn
    python train_demo.py --model_type transformer
"""

import argparse
import subprocess
import sys
import os


def train_rnn_demo():
    """训练RNN demo模型（小规模）"""
    print("="*60)
    print("训练RNN Demo模型（小规模，用于测试）")
    print("="*60)
    print("\n配置说明：")
    print("  - 数据集: train_10k.jsonl (10,000条)")
    print("  - 模型参数: embed_dim=128, hidden_dim=256, num_layers=1")
    print("  - 训练轮数: 10 epochs")
    print("  - 预期模型大小: <50MB")
    print("\n开始训练...\n")
    
    cmd = [
        'python', 'src/train_rnn.py',
        '--train_file', 'data/train_10k.jsonl',
        '--valid_file', 'data/valid.jsonl',
        '--test_file', 'data/test.jsonl',
        '--batch_size', '32',
        '--epochs', '10',
        '--lr', '0.001',
        '--embed_dim', '128',      # 减小embedding维度
        '--hidden_dim', '256',     # 减小hidden维度
        '--num_layers', '1',       # 减少层数
        '--rnn_type', 'gru',
        '--attention_type', 'dot',
        '--teacher_forcing',
        '--max_length', '50',
        '--min_freq', '2',
        '--bpe_vocab_size', '10000',  # 减小词汇表大小
        '--save_dir', 'checkpoints/demo_rnn',
        '--device', 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    ]
    
    print(f"运行命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ RNN Demo模型训练完成！")
        print("模型保存位置: checkpoints/demo_rnn/rnn_best.pt")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("✗ RNN Demo模型训练失败")
        print("="*60)
        return False


def train_transformer_demo():
    """训练Transformer demo模型（小规模）"""
    print("="*60)
    print("训练Transformer Demo模型（小规模，用于测试）")
    print("="*60)
    print("\n配置说明：")
    print("  - 数据集: train_10k.jsonl (10,000条)")
    print("  - 模型参数: d_model=256, n_layers=2, n_heads=4")
    print("  - 训练轮数: 10 epochs")
    print("  - 预期模型大小: <80MB")
    print("\n开始训练...\n")
    
    cmd = [
        'python', 'src/train_transformer.py',
        '--train_file', 'data/train_10k.jsonl',
        '--valid_file', 'data/valid.jsonl',
        '--test_file', 'data/test.jsonl',
        '--batch_size', '32',
        '--epochs', '10',
        '--lr', '0.0003',
        '--d_model', '256',        # 减小模型维度
        '--n_layers', '2',         # 减少层数
        '--n_heads', '4',          # 减少注意力头数
        '--d_ff', '512',           # 减小前馈网络维度
        '--pos_encoding', 'absolute',
        '--norm_type', 'layernorm',
        '--max_length', '50',
        '--warmup_steps', '100',   # 减少warmup步数
        '--min_freq', '2',
        '--bpe_vocab_size', '10000',  # 减小词汇表大小
        '--save_dir', 'checkpoints/demo_transformer',
        '--device', 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    ]
    
    print(f"运行命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ Transformer Demo模型训练完成！")
        print("模型保存位置: checkpoints/demo_transformer/transformer_best.pt")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("✗ Transformer Demo模型训练失败")
        print("="*60)
        return False


def main():
    parser = argparse.ArgumentParser(description='训练Demo模型（小规模，用于测试）')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rnn', 'transformer', 'all'],
                       help='要训练的模型类型')
    
    args = parser.parse_args()
    
    success = True
    
    if args.model_type == 'rnn':
        success = train_rnn_demo()
    elif args.model_type == 'transformer':
        success = train_transformer_demo()
    elif args.model_type == 'all':
        print("训练所有Demo模型...\n")
        success_rnn = train_rnn_demo()
        print("\n" + "="*60 + "\n")
        success_transformer = train_transformer_demo()
        success = success_rnn and success_transformer
    
    if success:
        print("\n" + "="*60)
        print("所有Demo模型训练完成！")
        print("="*60)
        print("\n模型文件位置：")
        if args.model_type in ['rnn', 'all']:
            print("  - RNN: checkpoints/demo_rnn/rnn_best.pt")
        if args.model_type in ['transformer', 'all']:
            print("  - Transformer: checkpoints/demo_transformer/transformer_best.pt")
        print("\n可以使用以下命令测试推理：")
        if args.model_type in ['rnn', 'all']:
            print("  python inference.py --model_type rnn --checkpoint checkpoints/demo_rnn/rnn_best.pt --input '问题在于为什么'")
        if args.model_type in ['transformer', 'all']:
            print("  python inference.py --model_type transformer --checkpoint checkpoints/demo_transformer/transformer_best.pt --input '问题在于为什么'")
        print("\n注意：Demo模型仅用于测试，性能不如完整训练的模型。")
    else:
        print("\n训练过程中出现错误，请检查日志。")
        sys.exit(1)


if __name__ == '__main__':
    main()

