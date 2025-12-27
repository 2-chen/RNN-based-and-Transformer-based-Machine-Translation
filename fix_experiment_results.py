#!/usr/bin/env python3
"""
修复脚本：收集已存在的实验结果并生成对比文件
"""

import json
import os
from pathlib import Path

def collect_existing_results():
    """手动收集已存在的实验结果"""
    
    os.makedirs('experiment_results', exist_ok=True)
    
    # RNN注意力机制对比
    print("收集RNN注意力机制实验结果...")
    rnn_attention_results = []
    for attn_type in ['dot', 'multiplicative', 'additive']:
        result_file = f"checkpoints/rnn_attn_{attn_type}/rnn_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['attention_type'] = attn_type
                rnn_attention_results.append(data)
                print(f"  ✓ 找到 {attn_type} 注意力实验结果")
        else:
            print(f"  ✗ 未找到 {attn_type} 注意力实验结果: {result_file}")
    
    if rnn_attention_results:
        with open('experiment_results/rnn_attention_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(rnn_attention_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存 {len(rnn_attention_results)} 个RNN注意力机制实验结果\n")
    else:
        print("✗ 未找到任何RNN注意力机制实验结果\n")
    
    # RNN训练策略对比
    print("收集RNN训练策略实验结果...")
    rnn_training_results = []
    for policy in ['teacher_forcing', 'free_running']:
        result_file = f"checkpoints/rnn_{policy}/rnn_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['training_policy'] = policy
                rnn_training_results.append(data)
                print(f"  ✓ 找到 {policy} 训练策略实验结果")
        else:
            print(f"  ✗ 未找到 {policy} 训练策略实验结果: {result_file}")
    
    if rnn_training_results:
        with open('experiment_results/rnn_training_policy_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(rnn_training_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存 {len(rnn_training_results)} 个RNN训练策略实验结果\n")
    else:
        print("✗ 未找到任何RNN训练策略实验结果\n")
    
    # Transformer位置编码对比
    print("收集Transformer位置编码实验结果...")
    transformer_pos_results = []
    for pos_type in ['absolute', 'relative']:
        result_file = f"checkpoints/transformer_pos_{pos_type}/transformer_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['pos_encoding'] = pos_type
                transformer_pos_results.append(data)
                print(f"  ✓ 找到 {pos_type} 位置编码实验结果")
        else:
            print(f"  ✗ 未找到 {pos_type} 位置编码实验结果: {result_file}")
    
    if transformer_pos_results:
        with open('experiment_results/transformer_pos_encoding_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(transformer_pos_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存 {len(transformer_pos_results)} 个Transformer位置编码实验结果\n")
    else:
        print("✗ 未找到任何Transformer位置编码实验结果\n")
    
    # Transformer归一化方法对比
    print("收集Transformer归一化方法实验结果...")
    transformer_norm_results = []
    for norm_type in ['layernorm', 'rmsnorm']:
        result_file = f"checkpoints/transformer_norm_{norm_type}/transformer_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['norm_type'] = norm_type
                transformer_norm_results.append(data)
                print(f"  ✓ 找到 {norm_type} 归一化实验结果")
        else:
            print(f"  ✗ 未找到 {norm_type} 归一化实验结果: {result_file}")
    
    if transformer_norm_results:
        with open('experiment_results/transformer_norm_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(transformer_norm_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存 {len(transformer_norm_results)} 个Transformer归一化实验结果\n")
    else:
        print("✗ 未找到任何Transformer归一化实验结果\n")
    
    print("="*60)
    print("结果收集完成！")
    print("="*60)
    print("\n已生成的对比文件：")
    if rnn_attention_results:
        print("  - experiment_results/rnn_attention_comparison.json")
    if rnn_training_results:
        print("  - experiment_results/rnn_training_policy_comparison.json")
    if transformer_pos_results:
        print("  - experiment_results/transformer_pos_encoding_comparison.json")
    if transformer_norm_results:
        print("  - experiment_results/transformer_norm_comparison.json")

if __name__ == '__main__':
    collect_existing_results()

