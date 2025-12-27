#!/usr/bin/env python3
"""
收集所有实验结果并生成对比表格
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def collect_all_results():
    """收集所有实验结果"""
    results = defaultdict(dict)
    
    # 收集checkpoints目录下的所有结果文件
    checkpoint_dir = Path("checkpoints")
    
    for result_file in checkpoint_dir.rglob("*results.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取实验信息
            args = data.get('args', {})
            model_type = None
            experiment_name = None
            
            # 根据路径和配置判断实验类型
            path_str = str(result_file)
            
            if 'rnn' in path_str.lower():
                model_type = 'rnn'
                # 判断是哪种RNN实验
                if 'attn' in path_str or 'attention' in path_str:
                    experiment_name = f"attention_{args.get('attention_type', 'unknown')}"
                elif 'teacher' in path_str or 'free' in path_str:
                    if args.get('free_running', False):
                        experiment_name = 'free_running'
                    else:
                        experiment_name = 'teacher_forcing'
                elif 'test' in path_str:
                    experiment_name = 'test_baseline'
                else:
                    experiment_name = 'baseline'
                    
            elif 'transformer' in path_str.lower():
                model_type = 'transformer'
                if 'pos' in path_str:
                    experiment_name = f"pos_encoding_{args.get('pos_encoding', 'unknown')}"
                elif 'norm' in path_str:
                    experiment_name = f"norm_{args.get('norm_type', 'unknown')}"
                elif 'bs_' in path_str or 'batch' in path_str:
                    experiment_name = f"batch_size_{args.get('batch_size', 'unknown')}"
                elif 'lr_' in path_str or 'learning' in path_str:
                    experiment_name = f"learning_rate_{args.get('lr', 'unknown')}"
                elif 'scale' in path_str:
                    d_model = args.get('d_model', 'unknown')
                    n_layers = args.get('n_layers', 'unknown')
                    experiment_name = f"scale_{d_model}_{n_layers}"
                elif 'test' in path_str:
                    experiment_name = 'test_baseline'
                else:
                    experiment_name = 'baseline'
                    
            elif 't5' in path_str.lower():
                model_type = 't5'
                experiment_name = 'baseline'
            
            if model_type and experiment_name:
                key = f"{model_type}_{experiment_name}"
                results[key] = {
                    'model_type': model_type,
                    'experiment_name': experiment_name,
                    'valid_bleu': data.get('best_valid_bleu', 0),
                    'test_bleu': data.get('test_bleu', 0),
                    'args': args,
                    'file_path': str(result_file)
                }
        except Exception as e:
            print(f"读取 {result_file} 时出错: {e}")
    
    return results

def generate_comparison_tables(results):
    """生成对比表格"""
    output = []
    
    # RNN实验结果
    rnn_results = {k: v for k, v in results.items() if v['model_type'] == 'rnn'}
    if rnn_results:
        output.append("### RNN模型实验结果\n")
        
        # 注意力机制对比
        attention_results = {k: v for k, v in rnn_results.items() if 'attention' in k}
        if attention_results:
            output.append("#### 注意力机制对比\n")
            output.append("| 注意力机制 | Valid BLEU | Test BLEU | 训练轮数 |\n")
            output.append("|-----------|------------|-----------|----------|\n")
            for k, v in sorted(attention_results.items()):
                attn_type = v['experiment_name'].split('_')[-1]
                epochs = v['args'].get('epochs', 'N/A')
                output.append(f"| {attn_type} | {v['valid_bleu']:.2f} | {v['test_bleu']:.2f} | {epochs} |\n")
            output.append("\n")
        
        # 训练策略对比
        training_results = {k: v for k, v in rnn_results.items() if 'teacher' in k or 'free' in k}
        if training_results:
            output.append("#### 训练策略对比\n")
            output.append("| 训练策略 | Valid BLEU | Test BLEU | 训练轮数 |\n")
            output.append("|---------|------------|-----------|----------|\n")
            for k, v in sorted(training_results.items()):
                policy = v['experiment_name']
                epochs = v['args'].get('epochs', 'N/A')
                output.append(f"| {policy} | {v['valid_bleu']:.2f} | {v['test_bleu']:.2f} | {epochs} |\n")
            output.append("\n")
    
    # Transformer实验结果
    transformer_results = {k: v for k, v in results.items() if v['model_type'] == 'transformer'}
    if transformer_results:
        output.append("### Transformer模型实验结果\n")
        
        # 位置编码对比
        pos_results = {k: v for k, v in transformer_results.items() if 'pos_encoding' in k}
        if pos_results:
            output.append("#### 位置编码对比\n")
            output.append("| 位置编码 | Valid BLEU | Test BLEU | 训练轮数 |\n")
            output.append("|---------|------------|-----------|----------|\n")
            for k, v in sorted(pos_results.items()):
                pos_type = v['experiment_name'].split('_')[-1]
                epochs = v['args'].get('epochs', 'N/A')
                output.append(f"| {pos_type} | {v['valid_bleu']:.2f} | {v['test_bleu']:.2f} | {epochs} |\n")
            output.append("\n")
        
        # 归一化方法对比
        norm_results = {k: v for k, v in transformer_results.items() if 'norm' in k}
        if norm_results:
            output.append("#### 归一化方法对比\n")
            output.append("| 归一化方法 | Valid BLEU | Test BLEU | 训练轮数 |\n")
            output.append("|-----------|------------|-----------|----------|\n")
            for k, v in sorted(norm_results.items()):
                norm_type = v['experiment_name'].split('_')[-1]
                epochs = v['args'].get('epochs', 'N/A')
                output.append(f"| {norm_type} | {v['valid_bleu']:.2f} | {v['test_bleu']:.2f} | {epochs} |\n")
            output.append("\n")
        
        # 超参数分析
        batch_results = {k: v for k, v in transformer_results.items() if 'batch_size' in k}
        if batch_results:
            output.append("#### Batch Size敏感性分析\n")
            output.append("| Batch Size | Valid BLEU | Test BLEU | 训练轮数 |\n")
            output.append("|-----------|------------|-----------|----------|\n")
            for k, v in sorted(batch_results.items(), key=lambda x: int(x[1]['args'].get('batch_size', 0))):
                bs = v['args'].get('batch_size', 'N/A')
                epochs = v['args'].get('epochs', 'N/A')
                output.append(f"| {bs} | {v['valid_bleu']:.2f} | {v['test_bleu']:.2f} | {epochs} |\n")
            output.append("\n")
        
        lr_results = {k: v for k, v in transformer_results.items() if 'learning_rate' in k}
        if lr_results:
            output.append("#### Learning Rate敏感性分析\n")
            output.append("| Learning Rate | Valid BLEU | Test BLEU | 训练轮数 |\n")
            output.append("|-------------|------------|-----------|----------|\n")
            for k, v in sorted(lr_results.items(), key=lambda x: float(x[1]['args'].get('lr', 0))):
                lr = v['args'].get('lr', 'N/A')
                epochs = v['args'].get('epochs', 'N/A')
                output.append(f"| {lr} | {v['valid_bleu']:.2f} | {v['test_bleu']:.2f} | {epochs} |\n")
            output.append("\n")
    
    return ''.join(output)

def main():
    print("收集所有实验结果...")
    results = collect_all_results()
    
    print(f"\n找到 {len(results)} 个实验结果")
    
    # 保存原始结果
    output_file = "experiment_results/all_results_collected.json"
    os.makedirs("experiment_results", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"原始结果已保存到: {output_file}")
    
    # 生成对比表格
    comparison_text = generate_comparison_tables(results)
    
    if comparison_text:
        comparison_file = "experiment_results/comparison_tables.md"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("# 实验结果对比表格\n\n")
            f.write(comparison_text)
        print(f"\n对比表格已保存到: {comparison_file}")
        print("\n" + "="*60)
        print("对比表格预览:")
        print("="*60)
        print(comparison_text)
    else:
        print("\n未找到可对比的实验结果")
        print("请确保已运行对比实验并生成了结果文件")

if __name__ == '__main__':
    main()

