#!/usr/bin/env python3
"""
生成项目可视化图表
包括BLEU分数对比、模型性能对比、翻译样例等
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 设置字体 - 使用DejaVu Sans确保所有字符正常显示
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
# 使用英文标签避免中文字体问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 创建输出目录
os.makedirs('visualizations', exist_ok=True)

def load_json(filepath):
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_model_comparison():
    """绘制模型性能对比图"""
    # 基础模型性能
    models = ['RNN', 'Transformer', 'T5']
    valid_bleu = [30.73, 30.13, 14.17]
    test_bleu = [16.78, 8.66, 14.54]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, valid_bleu, width, label='Valid BLEU', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_bleu, width, label='Test BLEU', color='#2196F3', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已生成模型性能对比图: visualizations/model_comparison.png")

def plot_rnn_attention_comparison():
    """绘制RNN注意力机制对比图"""
    data = load_json('experiment_results/rnn_attention_comparison.json')
    
    attention_types = []
    valid_bleu = []
    test_bleu = []
    
    for item in data:
        attention_types.append(item['attention_type'].capitalize())
        valid_bleu.append(item['best_valid_bleu'])
        test_bleu.append(item['test_bleu'])
    
    x = np.arange(len(attention_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, valid_bleu, width, label='Valid BLEU', color='#FF9800', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_bleu, width, label='Test BLEU', color='#9C27B0', alpha=0.8)
    
    ax.set_xlabel('Attention Mechanism', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('RNN Model - Attention Mechanism Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attention_types)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/rnn_attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已生成RNN注意力机制对比图: visualizations/rnn_attention_comparison.png")

def plot_rnn_training_policy_comparison():
    """绘制RNN训练策略对比图"""
    data = load_json('experiment_results/rnn_training_policy_comparison.json')
    
    policies = []
    valid_bleu = []
    test_bleu = []
    
    for item in data:
        policy = item['training_policy']
        if policy == 'teacher_forcing':
            policies.append('Teacher\nForcing')
        else:
            policies.append('Free\nRunning')
        valid_bleu.append(item['best_valid_bleu'])
        test_bleu.append(item['test_bleu'])
    
    x = np.arange(len(policies))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, valid_bleu, width, label='Valid BLEU', color='#E91E63', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_bleu, width, label='Test BLEU', color='#00BCD4', alpha=0.8)
    
    ax.set_xlabel('Training Policy', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('RNN Model - Training Policy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/rnn_training_policy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已生成RNN训练策略对比图: visualizations/rnn_training_policy_comparison.png")

def plot_transformer_norm_comparison():
    """绘制Transformer归一化方法对比图"""
    data = load_json('experiment_results/transformer_norm_comparison.json')
    
    norm_types = []
    valid_bleu = []
    test_bleu = []
    
    for item in data:
        norm_type = item['norm_type']
        if norm_type == 'layernorm':
            norm_types.append('LayerNorm')
        else:
            norm_types.append('RMSNorm')
        valid_bleu.append(item['best_valid_bleu'])
        test_bleu.append(item['test_bleu'])
    
    x = np.arange(len(norm_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, valid_bleu, width, label='Valid BLEU', color='#3F51B5', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_bleu, width, label='Test BLEU', color='#F44336', alpha=0.8)
    
    ax.set_xlabel('Normalization Method', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('Transformer Model - Normalization Method Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(norm_types)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/transformer_norm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已生成Transformer归一化方法对比图: visualizations/transformer_norm_comparison.png")

def plot_generalization_gap():
    """绘制泛化能力对比图（Valid BLEU - Test BLEU）"""
    # 基础模型
    models = ['RNN', 'Transformer', 'T5']
    gaps = [30.73 - 16.78, 30.13 - 8.66, abs(14.17 - 14.54)]
    
    colors = ['#FF5722', '#F44336', '#4CAF50']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, gaps, color=colors, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('BLEU Gap (Valid - Test)', fontsize=12)
    ax.set_title('Model Generalization Comparison (Smaller is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/generalization_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已生成泛化能力对比图: visualizations/generalization_gap.png")

def plot_attention_comparison_detailed():
    """绘制详细的注意力机制对比（包含差距）"""
    data = load_json('experiment_results/rnn_attention_comparison.json')
    
    attention_types = []
    valid_bleu = []
    test_bleu = []
    gaps = []
    
    for item in data:
        attn_type = item['attention_type']
        if attn_type == 'dot':
            attention_types.append('Dot\nProduct')
        elif attn_type == 'multiplicative':
            attention_types.append('Multi-\nplicative')
        else:
            attention_types.append('Additive')
        valid = item['best_valid_bleu']
        test = item['test_bleu']
        valid_bleu.append(valid)
        test_bleu.append(test)
        gaps.append(valid - test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：BLEU分数对比
    x = np.arange(len(attention_types))
    width = 0.35
    bars1 = ax1.bar(x - width/2, valid_bleu, width, label='Valid BLEU', color='#4CAF50', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_bleu, width, label='Test BLEU', color='#2196F3', alpha=0.8)
    
    ax1.set_xlabel('Attention Mechanism', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('Attention Mechanism Performance Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(attention_types)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    # 右图：泛化差距
    colors = ['#FF5722' if g > 10 else '#4CAF50' for g in gaps]
    bars3 = ax2.bar(attention_types, gaps, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Attention Mechanism', fontsize=12)
    ax2.set_ylabel('BLEU Gap (Valid - Test)', fontsize=12)
    ax2.set_title('Attention Mechanism Generalization Comparison', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/attention_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已生成详细注意力机制对比图: visualizations/attention_comparison_detailed.png")

def generate_translation_examples():
    """生成翻译样例（从测试集中选择）"""
    # 读取测试集
    test_file = 'data/test.jsonl'
    examples = []
    
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # 只取前5个样例
                    break
                data = json.loads(line)
                examples.append({
                    'chinese': data['zh'],
                    'english': data['en']
                })
    
    # 生成Markdown格式的翻译样例表格
    md_content = "## 翻译样例分析\n\n"
    md_content += "### 测试集样例\n\n"
    md_content += "| 序号 | 中文原文 | 英文参考翻译 |\n"
    md_content += "|------|---------|-------------|\n"
    
    for i, ex in enumerate(examples, 1):
        md_content += f"| {i} | {ex['chinese']} | {ex['english']} |\n"
    
    md_content += "\n### 模型翻译对比\n\n"
    md_content += "**注意**：由于需要加载模型进行推理，翻译样例的生成需要运行推理脚本。\n"
    md_content += "可以使用以下命令生成翻译样例：\n\n"
    md_content += "```bash\n"
    md_content += "# RNN模型翻译\n"
    md_content += "python inference.py --model_type rnn --checkpoint checkpoints/rnn_best.pt --input \"记录指出 HMX-1 曾询问此次活动是否违反了该法案。\"\n\n"
    md_content += "# Transformer模型翻译\n"
    md_content += "python inference.py --model_type transformer --checkpoint checkpoints/transformer_best.pt --input \"白宫将此次"美国制造"活动定义为官方活动，因此不受《哈奇法案》管辖。\"\n\n"
    md_content += "# T5模型翻译\n"
    md_content += "python inference.py --model_type t5 --checkpoint checkpoints/t5_best --input \"但是即使是官方活动也带有政治色彩。\"\n"
    md_content += "```\n"
    
    with open('visualizations/translation_examples.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print("✓ 已生成翻译样例文档: visualizations/translation_examples.md")
    return examples

def plot_all_comparisons():
    """生成所有对比图表"""
    print("="*60)
    print("开始生成可视化图表...")
    print("="*60)
    
    try:
        plot_model_comparison()
        plot_rnn_attention_comparison()
        plot_rnn_training_policy_comparison()
        plot_transformer_norm_comparison()
        plot_generalization_gap()
        plot_attention_comparison_detailed()
        generate_translation_examples()
        
        print("\n" + "="*60)
        print("所有可视化图表生成完成！")
        print("="*60)
        print("\n生成的图表文件：")
        print("  - visualizations/model_comparison.png")
        print("  - visualizations/rnn_attention_comparison.png")
        print("  - visualizations/rnn_training_policy_comparison.png")
        print("  - visualizations/transformer_norm_comparison.png")
        print("  - visualizations/generalization_gap.png")
        print("  - visualizations/attention_comparison_detailed.png")
        print("  - visualizations/translation_examples.md")
        
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    plot_all_comparisons()

