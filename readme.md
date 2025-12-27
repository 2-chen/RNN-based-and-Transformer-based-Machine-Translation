# 中英文机器翻译项目

本项目实现了基于RNN、Transformer和T5的中英文机器翻译模型，并进行了全面的性能比较和分析。

**代码仓库**: https://github.com/2-chen/RNN-based-and-Transformer-based-Machine-Translation

## 项目结构

```
.
├── data/                    # 数据目录
│   ├── train_10k.jsonl      # 小训练集（10k样本）
│   ├── train_100k.jsonl     # 大训练集（100k样本）
│   ├── valid.jsonl          # 验证集（500样本）
│   └── test.jsonl           # 测试集（200样本）
├── src/                     # 源代码目录
│   ├── data_utils.py        # 数据预处理工具
│   ├── rnn_model.py         # RNN-based NMT模型
│   ├── transformer_model.py # Transformer-based NMT模型
│   ├── attention.py         # 注意力机制实现
│   ├── train_rnn.py         # RNN模型训练脚本
│   ├── train_transformer.py # Transformer模型训练脚本
│   ├── train_t5.py          # T5微调脚本
│   └── metrics.py           # 评估指标（BLEU等）
├── inference.py             # 一键推理脚本
├── configs/                 # 配置文件目录
│   ├── rnn_config.yaml      # RNN模型配置
│   └── transformer_config.yaml # Transformer模型配置
├── checkpoints/             # 模型检查点目录
└── requirements.txt         # 依赖包

```

## 安装依赖

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
```

注意：新版本的 NLTK 需要 `punkt_tab`，旧版本需要 `punkt`。代码会自动处理，但建议同时下载以确保兼容性。

## 使用方法

### 1. 数据预处理

数据预处理会自动在训练时进行，包括：
- 数据清洗（移除非法字符、过滤罕见词）
- 分词：
  - **中文**：使用HanLP（备选：Jieba）
  - **英文**：使用BPE子词分词（备选：NLTK）
- 词汇表构建
- 词向量初始化（可选预训练词向量）

### 2. 训练RNN模型

```bash
python src/train_rnn.py --config configs/rnn_config.yaml
```

### 3. 训练Transformer模型

```bash
python src/train_transformer.py --config configs/transformer_config.yaml
```

### 4. 微调T5模型

```bash
python src/train_t5.py --config configs/t5_config.yaml
```

### 5. 推理

#### 5.1 一键测试所有模型

```bash
# 运行测试脚本（会自动测试所有可用的模型）
./test_inference.sh
```

#### 5.2 单独测试特定模型

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

## 模型特性

### RNN-based NMT
- 支持GRU和LSTM编码器-解码器
- 实现多种注意力机制（点积、乘法、加法）
- 支持Teacher Forcing和Free Running训练策略
- 支持贪婪解码和束搜索解码

### Transformer-based NMT
- 从零开始实现Transformer架构
- 支持绝对和相对位置编码
- 支持LayerNorm和RMSNorm归一化
- 超参数敏感性分析
- T5预训练模型微调

## 实验结果

实验结果和详细分析请参考项目报告。

## 作者

陈嘉诚
250010008
