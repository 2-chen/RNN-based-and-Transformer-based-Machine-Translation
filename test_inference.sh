#!/bin/bash
# 一键测试inference.py脚本

echo "=========================================="
echo "测试 inference.py 推理脚本"
echo "=========================================="
echo ""

# 检查模型文件是否存在
check_model() {
    local model_type=$1
    local checkpoint=$2
    
    if [ ! -f "$checkpoint" ] && [ ! -d "$checkpoint" ]; then
        echo "⚠ 警告: 模型文件不存在: $checkpoint"
        echo "   请先训练模型或下载模型文件"
        return 1
    fi
    return 0
}

# 测试RNN模型
echo "1. 测试RNN模型 (jieba+NLTK)..."
if check_model "rnn" "checkpoints/rnn_jieba_nltk/rnn_best.pt"; then
    python inference.py \
        --model_type rnn \
        --checkpoint checkpoints/rnn_jieba_nltk/rnn_best.pt \
        --input "自然语言处理是人工智能的重要分支" \
        --device cpu
    echo ""
else
    echo "   跳过（模型文件不存在）"
    echo ""
fi

# 测试RNN模型 (HanLP+BPE)
echo "2. 测试RNN模型 (HanLP+BPE)..."
if check_model "rnn" "checkpoints/rnn_hanlp_bpe/rnn_best.pt"; then
    python inference.py \
        --model_type rnn \
        --checkpoint checkpoints/rnn_hanlp_bpe/rnn_best.pt \
        --input "自然语言处理是人工智能的重要分支" \
        --device cpu
    echo ""
else
    echo "   跳过（模型文件不存在）"
    echo ""
fi

# 测试Transformer模型 (jieba+NLTK)
echo "3. 测试Transformer模型 (jieba+NLTK)..."
if check_model "transformer" "checkpoints/transformer_jieba_nltk/transformer_best.pt"; then
    python inference.py \
        --model_type transformer \
        --checkpoint checkpoints/transformer_jieba_nltk/transformer_best.pt \
        --input "机器翻译可以帮助人们理解不同语言的内容" \
        --device cpu
    echo ""
else
    echo "   跳过（模型文件不存在）"
    echo ""
fi

# 测试Transformer模型 (HanLP+BPE)
echo "4. 测试Transformer模型 (HanLP+BPE)..."
if check_model "transformer" "checkpoints/transformer_hanlp_bpe/transformer_best.pt"; then
    python inference.py \
        --model_type transformer \
        --checkpoint checkpoints/transformer_hanlp_bpe/transformer_best.pt \
        --input "机器翻译可以帮助人们理解不同语言的内容" \
        --device cpu
    echo ""
else
    echo "   跳过（模型文件不存在）"
    echo ""
fi

# 测试T5模型
echo "5. 测试T5模型..."
if check_model "t5" "checkpoints/t5_best"; then
    python inference.py \
        --model_type t5 \
        --checkpoint checkpoints/t5_best \
        --input "深度学习在自然语言处理领域取得了重大突破" \
        --device cpu
    echo ""
else
    echo "   跳过（模型文件不存在）"
    echo ""
fi

echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "注意：如果某些模型文件不存在，请先训练模型或下载模型文件"
echo "模型文件路径："
echo "  - RNN: checkpoints/rnn_jieba_nltk/rnn_best.pt 或 checkpoints/rnn_hanlp_bpe/rnn_best.pt"
echo "  - Transformer: checkpoints/transformer_jieba_nltk/transformer_best.pt 或 checkpoints/transformer_hanlp_bpe/transformer_best.pt"
echo "  - T5: checkpoints/t5_best/"
echo ""

