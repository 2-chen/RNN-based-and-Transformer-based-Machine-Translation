# 模型对比说明

## 文件组织

所有模型按以下规则命名和分类：

### 命名格式
`{model_type}_{chinese_tokenizer}_{english_tokenizer}/`

### 分类说明

#### 1. RNN模型
- **`rnn_jieba_nltk/`** - 原始版本，使用jieba+NLTK分词
- **`rnn_hanlp_bpe/`** - 改进版本，使用HanLP+BPE分词

#### 2. Transformer模型
- **`transformer_jieba_nltk/`** - 原始版本，使用jieba+NLTK分词
- **`transformer_hanlp_bpe/`** - 改进版本，使用HanLP+BPE分词

#### 3. T5模型
- **`t5_best/`** - T5模型（使用SentencePiece，T5自带）

## 快速查找

### 查找所有jieba+NLTK模型
```bash
find checkpoints -name "*jieba_nltk*" -type d
```

### 查找所有HanLP+BPE模型
```bash
find checkpoints -name "*hanlp_bpe*" -type d
```

### 查找所有RNN模型
```bash
find checkpoints -name "rnn_*" -type d
```

### 查找所有Transformer模型
```bash
find checkpoints -name "transformer_*" -type d
```

