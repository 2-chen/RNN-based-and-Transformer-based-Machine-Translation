#!/usr/bin/env python3
"""
快速测试脚本 - 验证实验流程是否正常
运行一个非常短的实验（1个epoch）来测试
"""

import subprocess
import sys
import os

def test_rnn_experiment():
    """测试RNN实验流程"""
    print("="*60)
    print("测试: RNN模型训练流程")
    print("="*60)
    
    # 使用小数据集和少量epoch进行快速测试
    cmd = [
        'python', 'src/train_rnn.py',
        '--train_file', 'data/train_10k.jsonl',  # 使用小数据集
        '--valid_file', 'data/valid.jsonl',
        '--test_file', 'data/test.jsonl',
        '--batch_size', '32',
        '--epochs', '1',  # 只训练1个epoch
        '--attention_type', 'dot',
        '--rnn_type', 'gru',
        '--save_dir', 'checkpoints/test_rnn',
        '--device', 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print("\n开始测试...\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✓ 测试成功! RNN训练流程正常")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("✗ 测试失败，请检查错误信息")
            print("="*60)
            return False
    except Exception as e:
        print(f"\n错误: {e}")
        return False

def test_transformer_experiment():
    """测试Transformer实验流程"""
    print("\n" + "="*60)
    print("测试: Transformer模型训练流程")
    print("="*60)
    
    cmd = [
        'python', 'src/train_transformer.py',
        '--train_file', 'data/train_10k.jsonl',
        '--valid_file', 'data/valid.jsonl',
        '--test_file', 'data/test.jsonl',
        '--batch_size', '32',
        '--epochs', '1',
        '--d_model', '512',  # 使用较小的模型
        '--n_layers', '2',
        '--save_dir', 'checkpoints/test_transformer',
        '--device', 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print("\n开始测试...\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✓ 测试成功! Transformer训练流程正常")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("✗ 测试失败，请检查错误信息")
            print("="*60)
            return False
    except Exception as e:
        print(f"\n错误: {e}")
        return False

def main():
    print("="*60)
    print("实验流程测试脚本")
    print("="*60)
    print("\n这个脚本会运行一个非常短的实验（1个epoch）来验证流程")
    print("不会运行完整实验，只是测试代码是否正常工作\n")
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == 'rnn':
            test_rnn_experiment()
        elif test_type == 'transformer':
            test_transformer_experiment()
        else:
            print(f"未知测试类型: {test_type}")
            print("可用类型: rnn, transformer")
    else:
        print("选择要测试的模型:")
        print("1. RNN模型")
        print("2. Transformer模型")
        print("3. 两者都测试")
        print("\n或使用命令行参数:")
        print("  python test_experiment.py rnn")
        print("  python test_experiment.py transformer")
        
        # 默认测试RNN
        print("\n默认测试RNN模型...\n")
        test_rnn_experiment()

if __name__ == '__main__':
    main()

