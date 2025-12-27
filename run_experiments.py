#!/usr/bin/env python3
"""
完整的对比实验脚本
自动运行所有要求的对比实验并收集结果
"""

import os
import subprocess
import json
import yaml
from datetime import datetime
from pathlib import Path

# 实验结果存储目录
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_command(cmd, experiment_name):
    """运行命令并记录结果"""
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False, e.stderr

def collect_results(model_type, experiment_name, save_dir=None):
    """收集实验结果"""
    if save_dir:
        # 如果提供了save_dir，从该目录读取结果
        results_file = os.path.join(save_dir, f"{model_type}_results.json")
    else:
        # 兼容旧的方式：尝试从checkpoints根目录读取
        results_file = f"checkpoints/{model_type}_results.json"
    
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    return None

# ==================== RNN 对比实验 ====================

def run_rnn_attention_experiments():
    """RNN: 不同注意力机制对比"""
    print("\n" + "="*60)
    print("RNN模型 - 注意力机制对比实验")
    print("="*60)
    
    attention_types = ['dot', 'multiplicative', 'additive']
    results = []
    
    for attn_type in attention_types:
        exp_name = f"RNN_attention_{attn_type}"
        config = {
            'train_file': 'data/train_100k.jsonl',
            'valid_file': 'data/valid.jsonl',
            'test_file': 'data/test.jsonl',
            'batch_size': 64,
            'epochs': 10,  # 可以根据需要调整
            'lr': 0.0005,
            'max_length': 50,
            'embed_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'rnn_type': 'gru',
            'attention_type': attn_type,
            'teacher_forcing': True,
            'free_running': False,
            'decode_method': 'greedy',
            'beam_size': 5,
            'save_dir': f'checkpoints/rnn_attn_{attn_type}',
            'device': 'cuda'
        }
        
        # 保存临时配置
        config_file = f"{RESULTS_DIR}/rnn_attn_{attn_type}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        # 运行训练
        cmd = [
            'python', 'src/train_rnn.py',
            '--config', config_file
        ]
        
        success, output = run_command(cmd, exp_name)
        
        if success:
            save_dir = f'checkpoints/rnn_attn_{attn_type}'
            result = collect_results('rnn', exp_name, save_dir=save_dir)
            if result:
                result['attention_type'] = attn_type
                results.append(result)
    
    # 保存对比结果
    comparison_file = f"{RESULTS_DIR}/rnn_attention_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n注意力机制对比结果已保存到: {comparison_file}")
    return results

def run_rnn_training_policy_experiments():
    """RNN: 训练策略对比 (Teacher Forcing vs Free Running)"""
    print("\n" + "="*60)
    print("RNN模型 - 训练策略对比实验")
    print("="*60)
    
    policies = [
        {'name': 'teacher_forcing', 'teacher_forcing': True, 'free_running': False},
        {'name': 'free_running', 'teacher_forcing': False, 'free_running': True}
    ]
    
    results = []
    
    for policy in policies:
        exp_name = f"RNN_{policy['name']}"
        config = {
            'train_file': 'data/train_100k.jsonl',
            'valid_file': 'data/valid.jsonl',
            'test_file': 'data/test.jsonl',
            'batch_size': 64,
            'epochs': 10,
            'lr': 0.0005,
            'max_length': 50,
            'embed_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'rnn_type': 'gru',
            'attention_type': 'dot',
            'teacher_forcing': policy['teacher_forcing'],
            'free_running': policy['free_running'],
            'decode_method': 'greedy',
            'beam_size': 5,
            'save_dir': f"checkpoints/rnn_{policy['name']}",
            'device': 'cuda'
        }
        
        config_file = f"{RESULTS_DIR}/rnn_{policy['name']}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        cmd = ['python', 'src/train_rnn.py', '--config', config_file]
        success, output = run_command(cmd, exp_name)
        
        if success:
            save_dir = f"checkpoints/rnn_{policy['name']}"
            result = collect_results('rnn', exp_name, save_dir=save_dir)
            if result:
                result['training_policy'] = policy['name']
                results.append(result)
    
    comparison_file = f"{RESULTS_DIR}/rnn_training_policy_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练策略对比结果已保存到: {comparison_file}")
    return results

def run_rnn_decoding_experiments():
    """RNN: 解码策略对比 (Greedy vs Beam Search)"""
    print("\n" + "="*60)
    print("RNN模型 - 解码策略对比实验")
    print("="*60)
    
    # 使用已训练好的模型进行解码对比
    model_path = "checkpoints/rnn_best.pt"
    if not os.path.exists(model_path):
        print(f"警告: 未找到训练好的模型 {model_path}")
        print("请先训练RNN模型")
        return []
    
    decode_methods = ['greedy', 'beam']
    results = []
    
    for method in decode_methods:
        exp_name = f"RNN_decode_{method}"
        
        # 创建评估脚本配置
        config = {
            'train_file': 'data/train_100k.jsonl',
            'valid_file': 'data/valid.jsonl',
            'test_file': 'data/test.jsonl',
            'batch_size': 64,
            'epochs': 0,  # 不训练，只评估
            'decode_method': method,
            'beam_size': 5,
            'save_dir': 'checkpoints',
            'device': 'cuda'
        }
        
        config_file = f"{RESULTS_DIR}/rnn_decode_{method}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        # 注意：需要修改train_rnn.py支持仅评估模式
        # 或者直接使用inference.py进行评估
        print(f"使用 {method} 解码方法评估模型...")
        # 这里需要根据实际代码调整
    
    return results

# ==================== Transformer 对比实验 ====================

def run_transformer_pos_encoding_experiments():
    """Transformer: 位置编码对比"""
    print("\n" + "="*60)
    print("Transformer模型 - 位置编码对比实验")
    print("="*60)
    
    pos_encodings = ['absolute', 'relative']
    results = []
    
    for pos_enc in pos_encodings:
        exp_name = f"Transformer_pos_{pos_enc}"
        config = {
            'train_file': 'data/train_100k.jsonl',
            'valid_file': 'data/valid.jsonl',
            'test_file': 'data/test.jsonl',
            'batch_size': 128,
            'epochs': 10,
            'lr': 0.0003,
            'warmup_steps': 4000,
            'max_length': 50,
            'd_model': 1024,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'pos_encoding': pos_enc,
            'norm_type': 'layernorm',
            'save_dir': f'checkpoints/transformer_pos_{pos_enc}',
            'device': 'cuda'
        }
        
        config_file = f"{RESULTS_DIR}/transformer_pos_{pos_enc}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        cmd = ['python', 'src/train_transformer.py', '--config', config_file]
        success, output = run_command(cmd, exp_name)
        
        if success:
            save_dir = f'checkpoints/transformer_pos_{pos_enc}'
            result = collect_results('transformer', exp_name, save_dir=save_dir)
            if result:
                result['pos_encoding'] = pos_enc
                results.append(result)
    
    comparison_file = f"{RESULTS_DIR}/transformer_pos_encoding_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n位置编码对比结果已保存到: {comparison_file}")
    return results

def run_transformer_norm_experiments():
    """Transformer: 归一化方法对比"""
    print("\n" + "="*60)
    print("Transformer模型 - 归一化方法对比实验")
    print("="*60)
    
    norm_types = ['layernorm', 'rmsnorm']
    results = []
    
    for norm_type in norm_types:
        exp_name = f"Transformer_norm_{norm_type}"
        config = {
            'train_file': 'data/train_100k.jsonl',
            'valid_file': 'data/valid.jsonl',
            'test_file': 'data/test.jsonl',
            'batch_size': 128,
            'epochs': 10,
            'lr': 0.0003,
            'warmup_steps': 4000,
            'max_length': 50,
            'd_model': 1024,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'pos_encoding': 'absolute',
            'norm_type': norm_type,
            'save_dir': f'checkpoints/transformer_norm_{norm_type}',
            'device': 'cuda'
        }
        
        config_file = f"{RESULTS_DIR}/transformer_norm_{norm_type}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        cmd = ['python', 'src/train_transformer.py', '--config', config_file]
        success, output = run_command(cmd, exp_name)
        
        if success:
            save_dir = f'checkpoints/transformer_norm_{norm_type}'
            result = collect_results('transformer', exp_name, save_dir=save_dir)
            if result:
                result['norm_type'] = norm_type
                results.append(result)
    
    comparison_file = f"{RESULTS_DIR}/transformer_norm_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n归一化方法对比结果已保存到: {comparison_file}")
    return results

def run_transformer_hyperparameter_experiments():
    """Transformer: 超参数敏感性分析"""
    print("\n" + "="*60)
    print("Transformer模型 - 超参数敏感性分析")
    print("="*60)
    
    # Batch Size 实验
    batch_sizes = [32, 64, 128, 256]
    batch_results = []
    
    for bs in batch_sizes:
        exp_name = f"Transformer_batch_{bs}"
        config = {
            'train_file': 'data/train_100k.jsonl',
            'valid_file': 'data/valid.jsonl',
            'test_file': 'data/test.jsonl',
            'batch_size': bs,
            'epochs': 5,  # 减少epoch以加快实验
            'lr': 0.0003,
            'warmup_steps': 4000,
            'max_length': 50,
            'd_model': 1024,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'pos_encoding': 'absolute',
            'norm_type': 'layernorm',
            'save_dir': f'checkpoints/transformer_bs_{bs}',
            'device': 'cuda'
        }
        
        config_file = f"{RESULTS_DIR}/transformer_bs_{bs}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        cmd = ['python', 'src/train_transformer.py', '--config', config_file]
        success, output = run_command(cmd, exp_name)
        
        if success:
            save_dir = f'checkpoints/transformer_bs_{bs}'
            result = collect_results('transformer', exp_name, save_dir=save_dir)
            if result:
                result['batch_size'] = bs
                batch_results.append(result)
    
    # Learning Rate 实验
    learning_rates = [0.0001, 0.0003, 0.0005, 0.001]
    lr_results = []
    
    for lr in learning_rates:
        exp_name = f"Transformer_lr_{lr}"
        config = {
            'train_file': 'data/train_100k.jsonl',
            'valid_file': 'data/valid.jsonl',
            'test_file': 'data/test.jsonl',
            'batch_size': 128,
            'epochs': 5,
            'lr': lr,
            'warmup_steps': 4000,
            'max_length': 50,
            'd_model': 1024,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'pos_encoding': 'absolute',
            'norm_type': 'layernorm',
            'save_dir': f'checkpoints/transformer_lr_{lr}',
            'device': 'cuda'
        }
        
        config_file = f"{RESULTS_DIR}/transformer_lr_{lr}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        cmd = ['python', 'src/train_transformer.py', '--config', config_file]
        success, output = run_command(cmd, exp_name)
        
        if success:
            save_dir = f'checkpoints/transformer_lr_{lr}'
            result = collect_results('transformer', exp_name, save_dir=save_dir)
            if result:
                result['learning_rate'] = lr
                lr_results.append(result)
    
    # Model Scale 实验
    model_scales = [
        {'d_model': 512, 'n_layers': 4},
        {'d_model': 768, 'n_layers': 6},
        {'d_model': 1024, 'n_layers': 6},
    ]
    scale_results = []
    
    for scale in model_scales:
        exp_name = f"Transformer_scale_{scale['d_model']}_{scale['n_layers']}"
        config = {
            'train_file': 'data/train_100k.jsonl',
            'valid_file': 'data/valid.jsonl',
            'test_file': 'data/test.jsonl',
            'batch_size': 128,
            'epochs': 5,
            'lr': 0.0003,
            'warmup_steps': 4000,
            'max_length': 50,
            'd_model': scale['d_model'],
            'n_heads': 8,
            'n_layers': scale['n_layers'],
            'd_ff': scale['d_model'] * 2,
            'pos_encoding': 'absolute',
            'norm_type': 'layernorm',
            'save_dir': f"checkpoints/transformer_scale_{scale['d_model']}_{scale['n_layers']}",
            'device': 'cuda'
        }
        
        config_file = f"{RESULTS_DIR}/transformer_scale_{scale['d_model']}_{scale['n_layers']}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        cmd = ['python', 'src/train_transformer.py', '--config', config_file]
        success, output = run_command(cmd, exp_name)
        
        if success:
            save_dir = f"checkpoints/transformer_scale_{scale['d_model']}_{scale['n_layers']}"
            result = collect_results('transformer', exp_name, save_dir=save_dir)
            if result:
                result['d_model'] = scale['d_model']
                result['n_layers'] = scale['n_layers']
                scale_results.append(result)
    
    # 保存所有超参数实验结果
    hyper_results = {
        'batch_size': batch_results,
        'learning_rate': lr_results,
        'model_scale': scale_results
    }
    
    comparison_file = f"{RESULTS_DIR}/transformer_hyperparameter_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(hyper_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n超参数敏感性分析结果已保存到: {comparison_file}")
    return hyper_results

def main():
    """运行所有对比实验"""
    print("="*60)
    print("开始运行完整的对比实验")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*60)
    print("注意: 所有实验函数调用默认被注释，以避免意外运行长时间实验")
    print("="*60)
    print("\n要运行实验，请:")
    print("1. 取消注释相应的函数调用")
    print("2. 或使用命令行参数运行特定实验:")
    print("   python run_experiments.py rnn_attention")
    print("   python run_experiments.py transformer_pos")
    print("\n可用实验类型:")
    print("  - rnn_attention: RNN注意力机制对比")
    print("  - rnn_training: RNN训练策略对比")
    print("  - rnn_decoding: RNN解码策略对比")
    print("  - transformer_pos: Transformer位置编码对比")
    print("  - transformer_norm: Transformer归一化对比")
    print("  - transformer_hyper: Transformer超参数分析")
    
    all_results = {}
    
    # RNN 实验（默认注释，需要时取消注释）
    print("\n" + "="*60)
    print("第一部分: RNN模型对比实验")
    print("="*60)
    print("(实验函数已注释，取消注释以运行)")
    
    # 取消下面的注释以运行实验
    # all_results['rnn_attention'] = run_rnn_attention_experiments()
    # all_results['rnn_training_policy'] = run_rnn_training_policy_experiments()
    # all_results['rnn_decoding'] = run_rnn_decoding_experiments()
    
    # Transformer 实验（默认注释，需要时取消注释）
    print("\n" + "="*60)
    print("第二部分: Transformer模型对比实验")
    print("="*60)
    print("(实验函数已注释，取消注释以运行)")
    
    # 取消下面的注释以运行实验
    # all_results['transformer_pos_encoding'] = run_transformer_pos_encoding_experiments()
    # all_results['transformer_norm'] = run_transformer_norm_experiments()
    # all_results['transformer_hyperparameter'] = run_transformer_hyperparameter_experiments()
    
    # 保存所有结果
    summary_file = f"{RESULTS_DIR}/all_experiments_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("脚本执行完成!")
    print("="*60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果文件: {summary_file}")
    print("\n注意: 由于实验时间较长，建议:")
    print("1. 使用GPU加速训练")
    print("2. 可以分批运行实验（使用命令行参数运行特定实验）")
    print("3. 根据计算资源调整epoch数")
    print("4. 先运行单个实验验证流程，再批量运行")

if __name__ == '__main__':
    # 可以选择运行特定实验
    import sys
    
    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
        if experiment_type == 'rnn_attention':
            run_rnn_attention_experiments()
        elif experiment_type == 'rnn_training':
            run_rnn_training_policy_experiments()
        elif experiment_type == 'rnn_decoding':
            run_rnn_decoding_experiments()
        elif experiment_type == 'transformer_pos':
            run_transformer_pos_encoding_experiments()
        elif experiment_type == 'transformer_norm':
            run_transformer_norm_experiments()
        elif experiment_type == 'transformer_hyper':
            run_transformer_hyperparameter_experiments()
        else:
            print(f"未知实验类型: {experiment_type}")
            print("可用类型: rnn_attention, rnn_training, rnn_decoding, transformer_pos, transformer_norm, transformer_hyper")
    else:
        main()

