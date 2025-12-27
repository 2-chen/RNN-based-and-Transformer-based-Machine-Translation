"""
RNN解码策略对比评估脚本
使用已训练好的模型，分别用Greedy和Beam Search解码，对比性能
"""

import argparse
import torch
import json
import os
import sys
from torch.utils.data import DataLoader

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.data_utils import (
        prepare_data, TranslationDataset, collate_fn,
        PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, tokenize_en
    )
    from src.rnn_model import RNNSeq2Seq
    from src.metrics import calculate_bleu
    from src.train_rnn import evaluate
except ImportError:
    try:
        from data_utils import (
            prepare_data, TranslationDataset, collate_fn,
            PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, tokenize_en
        )
        from rnn_model import RNNSeq2Seq
        from metrics import calculate_bleu
        from train_rnn import evaluate
    except ImportError:
        # 如果还是失败，尝试从src目录导入
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_utils", "src/data_utils.py")
        data_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_utils)
        
        spec = importlib.util.spec_from_file_location("rnn_model", "src/rnn_model.py")
        rnn_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rnn_model)
        
        spec = importlib.util.spec_from_file_location("metrics", "src/metrics.py")
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        
        spec = importlib.util.spec_from_file_location("train_rnn", "src/train_rnn.py")
        train_rnn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_rnn)
        
        prepare_data = data_utils.prepare_data
        TranslationDataset = data_utils.TranslationDataset
        collate_fn = data_utils.collate_fn
        PAD_TOKEN = data_utils.PAD_TOKEN
        SOS_TOKEN = data_utils.SOS_TOKEN
        EOS_TOKEN = data_utils.EOS_TOKEN
        tokenize_en = data_utils.tokenize_en
        RNNSeq2Seq = rnn_model.RNNSeq2Seq
        calculate_bleu = metrics.calculate_bleu
        evaluate = train_rnn.evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/rnn_best.pt',
                       help='训练好的RNN模型路径')
    parser.add_argument('--valid_file', type=str, default='data/valid.jsonl')
    parser.add_argument('--test_file', type=str, default='data/test.jsonl')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_file', type=str, default='checkpoints/rnn_decoding_comparison.json')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        print("请先训练RNN模型")
        return
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    model_args = checkpoint.get('args', {})
    
    # 创建模型
    model = RNNSeq2Seq(
        len(src_vocab),
        len(tgt_vocab),
        model_args.get('embed_dim', 256),
        model_args.get('hidden_dim', 512),
        model_args.get('num_layers', 2),
        model_args.get('rnn_type', 'gru'),
        model_args.get('attention_type', 'dot'),
        model_args.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("模型加载成功")
    
    # 准备数据（使用相同的预处理）
    print("准备数据...")
    _, _, _, valid_data, test_data = prepare_data(
        model_args.get('train_file', 'data/train_100k.jsonl'),
        args.valid_file,
        args.test_file,
        args.max_length,
        model_args.get('min_freq', 2),
        bpe_vocab_size=model_args.get('bpe_vocab_size', 30000),
        bpe_model_path=model_args.get('bpe_model_path', None)
    )
    
    valid_dataset = TranslationDataset(valid_data, src_vocab, tgt_vocab, args.max_length)
    test_dataset = TranslationDataset(test_data, src_vocab, tgt_vocab, args.max_length)
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    results = {}
    
    # 评估Greedy解码
    print("\n" + "="*60)
    print("使用Greedy解码评估...")
    print("="*60)
    valid_bleu_greedy = evaluate(model, valid_loader, tgt_vocab, device, 'greedy', args.beam_size)
    test_bleu_greedy = evaluate(model, test_loader, tgt_vocab, device, 'greedy', args.beam_size)
    
    results['greedy'] = {
        'valid_bleu': valid_bleu_greedy,
        'test_bleu': test_bleu_greedy
    }
    
    print(f"Greedy - Valid BLEU: {valid_bleu_greedy:.4f}, Test BLEU: {test_bleu_greedy:.4f}")
    
    # 评估Beam Search解码
    print("\n" + "="*60)
    print("使用Beam Search解码评估...")
    print("="*60)
    valid_bleu_beam = evaluate(model, valid_loader, tgt_vocab, device, 'beam', args.beam_size)
    test_bleu_beam = evaluate(model, test_loader, tgt_vocab, device, 'beam', args.beam_size)
    
    results['beam_search'] = {
        'valid_bleu': valid_bleu_beam,
        'test_bleu': test_bleu_beam,
        'beam_size': args.beam_size
    }
    
    print(f"Beam Search (beam_size={args.beam_size}) - Valid BLEU: {valid_bleu_beam:.4f}, Test BLEU: {test_bleu_beam:.4f}")
    
    # 计算提升
    valid_improvement = valid_bleu_beam - valid_bleu_greedy
    test_improvement = test_bleu_beam - test_bleu_greedy
    
    results['comparison'] = {
        'valid_bleu_improvement': valid_improvement,
        'test_bleu_improvement': test_improvement,
        'valid_bleu_improvement_percent': (valid_improvement / valid_bleu_greedy * 100) if valid_bleu_greedy > 0 else 0,
        'test_bleu_improvement_percent': (test_improvement / test_bleu_greedy * 100) if test_bleu_greedy > 0 else 0
    }
    
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"Valid BLEU提升: {valid_improvement:.4f} ({results['comparison']['valid_bleu_improvement_percent']:.2f}%)")
    print(f"Test BLEU提升: {test_improvement:.4f} ({results['comparison']['test_bleu_improvement_percent']:.2f}%)")
    
    # 保存结果
    results['args'] = vars(args)
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {args.output_file}")


if __name__ == '__main__':
    main()

