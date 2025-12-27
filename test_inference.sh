#!/bin/bash
# 一键测试inference.py脚本（仅测试Demo模型）

echo "=========================================="
echo "测试 inference.py 推理脚本（Demo模型）"
echo "=========================================="
echo ""

# 检查模型文件是否存在
check_model() {
    local model_type=$1
    local checkpoint=$2
    
    if [ ! -f "$checkpoint" ] && [ ! -d "$checkpoint" ]; then
        echo "⚠ 警告: 模型文件不存在: $checkpoint"
        echo "   请先运行 'python train_demo.py --model_type $model_type' 生成模型"
        return 1
    fi
    return 0
}

# 测试Demo RNN模型
echo "1. 测试Demo RNN模型..."
if check_model "rnn" "checkpoints/demo_rnn/rnn_best.pt"; then
    python inference.py \
        --model_type rnn \
        --checkpoint checkpoints/demo_rnn/rnn_best.pt \
        --input "你好世界" \
        --device cpu
    echo ""
else
    echo "   跳过（模型文件不存在）"
    echo ""
fi

# 测试Demo Transformer模型
echo "2. 测试Demo Transformer模型..."
if check_model "transformer" "checkpoints/demo_transformer/transformer_best.pt"; then
    python inference.py \
        --model_type transformer \
        --checkpoint checkpoints/demo_transformer/transformer_best.pt \
        --input "你好世界" \
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
echo "说明："
echo "  - 本脚本仅测试Demo模型（文件<100MB，已上传到GitHub）"
echo "  - 如需测试其他模型，请先训练或下载模型文件"
echo ""
echo "快速生成Demo模型："
echo "  python train_demo.py --model_type rnn          # 训练RNN demo模型"
echo "  python train_demo.py --model_type transformer  # 训练Transformer demo模型"
echo "  python train_demo.py --model_type all          # 训练所有demo模型"
echo ""
echo "Demo模型文件路径："
echo "  - Demo RNN: checkpoints/demo_rnn/rnn_best.pt"
echo "  - Demo Transformer: checkpoints/demo_transformer/transformer_best.pt"
echo ""
