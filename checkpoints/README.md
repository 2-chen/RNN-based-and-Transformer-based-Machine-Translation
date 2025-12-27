# 模型检查点目录说明

本目录包含所有训练好的模型检查点和结果文件，按分词方法分类组织。

## 目录结构

### RNN模型

- **`rnn_jieba_nltk/`** - 使用jieba（中文）+ NLTK（英文）分词的RNN模型
  - `rnn_best.pt` - 最佳模型权重
  - `rnn_results.json` - 训练结果（BLEU分数等）

- **`rnn_hanlp_bpe/`** - 使用HanLP（中文）+ BPE（英文）分词的RNN模型
  - `rnn_best.pt` - 最佳模型权重
  - `rnn_results.json` - 训练结果（BLEU分数等）

- **`rnn_test_hanlp_bpe/`** - 测试用的RNN模型（小数据集，HanLP+BPE）

### Transformer模型

- **`transformer_jieba_nltk/`** - 使用jieba（中文）+ NLTK（英文）分词的Transformer模型
  - `transformer_best.pt` - 最佳模型权重
  - `transformer_results.json` - 训练结果（BLEU分数等）

- **`transformer_hanlp_bpe/`** - 使用HanLP（中文）+ BPE（英文）分词的Transformer模型
  - `transformer_best.pt` - 最佳模型权重
  - `transformer_results.json` - 训练结果（BLEU分数等）

### T5模型

- **`t5_best/`** - T5预训练模型微调后的检查点
  - 使用SentencePiece进行分词（T5自带）
  - `t5_results.json` - 训练结果

### 其他

- **`test_rnn/`** - 测试用的RNN模型
- **`test_transformer/`** - 测试用的Transformer模型
- **`bpe_tokenizer.json`** - BPE分词器模型文件

## 使用示例

### 推理时使用不同模型

```bash
# 使用jieba+NLTK的RNN模型
python inference.py \
    --model_type rnn \
    --checkpoint checkpoints/rnn_jieba_nltk/rnn_best.pt \
    --input "自然语言处理是人工智能的重要分支"

# 使用HanLP+BPE的RNN模型
python inference.py \
    --model_type rnn \
    --checkpoint checkpoints/rnn_hanlp_bpe/rnn_best.pt \
    --input "自然语言处理是人工智能的重要分支"

# 使用jieba+NLTK的Transformer模型
python inference.py \
    --model_type transformer \
    --checkpoint checkpoints/transformer_jieba_nltk/transformer_best.pt \
    --input "机器翻译可以帮助人们理解不同语言的内容"

# 使用HanLP+BPE的Transformer模型
python inference.py \
    --model_type transformer \
    --checkpoint checkpoints/transformer_hanlp_bpe/transformer_best.pt \
    --input "机器翻译可以帮助人们理解不同语言的内容"

# 使用T5模型
python inference.py \
    --model_type t5 \
    --checkpoint checkpoints/t5_best \
    --input "深度学习在自然语言处理领域取得了重大突破"
```

## 性能对比

| 模型 | 分词方法 | 验证BLEU | 测试BLEU | 模型大小 |
|------|---------|---------|---------|---------|
| RNN | jieba+NLTK | 30.73 | 16.78 | 197MB |
| RNN | HanLP+BPE | - | - | 103MB |
| Transformer | jieba+NLTK | - | - | 833MB |
| Transformer | HanLP+BPE | - | - | 679MB |
| T5 | SentencePiece | - | - | - |

*注：具体BLEU分数请查看各目录下的results.json文件*

## 命名规则

- **`{model_type}_{chinese_tokenizer}_{english_tokenizer}/`**
  - `model_type`: rnn, transformer, t5
  - `chinese_tokenizer`: jieba, hanlp
  - `english_tokenizer`: nltk, bpe

例如：
- `rnn_jieba_nltk/` - RNN模型，使用jieba和NLTK
- `rnn_hanlp_bpe/` - RNN模型，使用HanLP和BPE
- `transformer_jieba_nltk/` - Transformer模型，使用jieba和NLTK
- `transformer_hanlp_bpe/` - Transformer模型，使用HanLP和BPE

