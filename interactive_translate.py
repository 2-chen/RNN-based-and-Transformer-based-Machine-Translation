"""
交互式翻译脚本
支持持续输入中文文本进行翻译
"""

import argparse
import torch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inference import (
    load_rnn_model, load_transformer_model, load_t5_model,
    rnn_inference, transformer_inference, t5_inference
)


def main():
    parser = argparse.ArgumentParser(description='交互式机器翻译脚本')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rnn', 'transformer', 't5'],
                       help='模型类型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--max_length', type=int, default=50,
                       help='最大生成长度')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("=" * 60)
    print("交互式机器翻译系统")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"检查点: {args.checkpoint}")
    print(f"设备: {device}")
    print("=" * 60)
    print("提示: 输入中文文本进行翻译，输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()
    
    # 加载模型
    print("正在加载模型...")
    try:
        if args.model_type == 'rnn':
            model, src_vocab, tgt_vocab = load_rnn_model(args.checkpoint, device)
            inference_func = lambda text: rnn_inference(model, src_vocab, tgt_vocab, text, device, args.max_length)
        elif args.model_type == 'transformer':
            model, src_vocab, tgt_vocab = load_transformer_model(args.checkpoint, device)
            inference_func = lambda text: transformer_inference(model, src_vocab, tgt_vocab, text, device, args.max_length)
        elif args.model_type == 't5':
            model, tokenizer = load_t5_model(args.checkpoint, device)
            inference_func = lambda text: t5_inference(model, tokenizer, text, device)
        print("✓ 模型加载完成！")
        print()
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        sys.exit(1)
    
    # 交互式循环
    while True:
        try:
            # 获取用户输入
            user_input = input("请输入中文文本: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("感谢使用！再见！")
                break
            
            # 检查空输入
            if not user_input:
                print("输入不能为空，请重新输入。")
                continue
            
            # 执行翻译
            print("翻译中...")
            try:
                translation = inference_func(user_input)
                print(f"翻译结果: {translation}")
            except Exception as e:
                print(f"翻译失败: {e}")
            
            print()  # 空行分隔
            
        except KeyboardInterrupt:
            print("\n\n感谢使用！再见！")
            break
        except EOFError:
            print("\n\n感谢使用！再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            print("请重试或输入 'quit' 退出。")
            print()


if __name__ == '__main__':
    main()

