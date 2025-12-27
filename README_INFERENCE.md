# inference.py 一键测试指南

## 快速测试

### 方法1：使用测试脚本（推荐）

```bash
# 一键测试所有可用模型
./test_inference.sh
```

这个脚本会自动检测可用的模型文件，并依次测试：
- RNN模型（jieba+NLTK版本）
- RNN模型（HanLP+BPE版本）
- Transformer模型（jieba+NLTK版本）
- Transformer模型（HanLP+BPE版本）
- T5模型

### 方法2：手动测试单个模型

```bash
# RNN模型 (jieba+NLTK)
python inference.py \
    --model_type rnn \
    --checkpoint checkpoints/rnn_jieba_nltk/rnn_best.pt \
    --input "自然语言处理是人工智能的重要分支"

# RNN模型 (HanLP+BPE)
python inference.py \
    --model_type rnn \
    --checkpoint checkpoints/rnn_hanlp_bpe/rnn_best.pt \
    --input "自然语言处理是人工智能的重要分支"

# Transformer模型 (jieba+NLTK)
python inference.py \
    --model_type transformer \
    --checkpoint checkpoints/transformer_jieba_nltk/transformer_best.pt \
    --input "机器翻译可以帮助人们理解不同语言的内容"

# Transformer模型 (HanLP+BPE)
python inference.py \
    --model_type transformer \
    --checkpoint checkpoints/transformer_hanlp_bpe/transformer_best.pt \
    --input "机器翻译可以帮助人们理解不同语言的内容"

# T5模型
python inference.py \
    --model_type t5 \
    --checkpoint checkpoints/t5_best \
    --input "深度学习在自然语言处理领域取得了重大突破"
```

## 参数说明

- `--model_type`: 模型类型，可选 `rnn`, `transformer`, `t5`
- `--checkpoint`: 模型检查点路径
- `--input`: 输入的中文文本
- `--device`: 设备，可选 `cpu` 或 `cuda`（默认自动检测）
- `--max_length`: 最大生成长度（默认50）

## 注意事项

1. **模型文件**：确保模型文件存在，如果不存在会提示错误
2. **设备选择**：如果没有GPU，会自动使用CPU
3. **中文输入**：使用引号包裹中文文本，避免特殊字符问题

## 测试示例

```bash
# 测试简单句子
python inference.py --model_type rnn --checkpoint checkpoints/rnn_jieba_nltk/rnn_best.pt --input "你好"

# 测试复杂句子
python inference.py --model_type transformer --checkpoint checkpoints/transformer_hanlp_bpe/transformer_best.pt --input "人工智能正在改变我们的生活方式"

# 使用CPU
python inference.py --model_type rnn --checkpoint checkpoints/rnn_jieba_nltk/rnn_best.pt --input "测试" --device cpu
```

