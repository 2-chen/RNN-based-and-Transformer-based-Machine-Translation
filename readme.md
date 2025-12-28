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

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/2-chen/RNN-based-and-Transformer-based-Machine-Translation.git

# 进入项目目录
cd RNN-based-and-Transformer-based-Machine-Translation

# 安装依赖
pip install -r requirements.txt

# 一键测试推理（使用预训练的Demo模型）
./test_inference.sh
```

### 使用Demo模型进行单次翻译

使用 `inference.py` 可以直接翻译指定的中文文本：

```bash
# 使用Demo RNN模型翻译
python inference.py \
    --model_type rnn \
    --checkpoint checkpoints/demo_rnn/rnn_best.pt \
    --input "全球经济缓慢增长又一年"

# 使用Demo Transformer模型翻译
python inference.py \
    --model_type transformer \
    --checkpoint checkpoints/demo_transformer/transformer_best.pt \
    --input "全球经济缓慢增长又一年"
```

## 使用方法

### 1. 数据预处理

数据预处理会自动在训练时进行，包括：
- 数据清洗（移除非法字符、过滤罕见词）
- 分词：
  - **中文**：使用HanLP（备选：Jieba）
  - **英文**：使用BPE子词分词（备选：NLTK）
- 词汇表构建
- 词向量初始化（可选预训练词向量）

### 2. 训练模型

#### 2.1 训练Demo模型（可选，用于快速测试）

Demo模型已经包含在仓库中，可以直接使用。如果需要重新训练：

```bash
# 训练RNN Demo模型（约30-50MB，2个epoch，5-10分钟）
python train_demo.py --model_type rnn

# 训练Transformer Demo模型（约50-80MB，2个epoch，5-10分钟）
python train_demo.py --model_type transformer

# 训练所有Demo模型
python train_demo.py --model_type all
```

**Demo模型特点**：
- 快速训练（2个epoch，约5-10分钟）
- 文件小（<100MB，已包含在仓库中）
- 适合测试和演示
- 性能不如完整训练的模型

#### 2.2 训练完整RNN模型

```bash
python src/train_rnn.py --config configs/rnn_config.yaml
```

#### 2.3 训练完整Transformer模型

```bash
python src/train_transformer.py --config configs/transformer_config.yaml
```

#### 2.4 微调T5模型

```bash
python src/train_t5.py --config configs/t5_config.yaml
```

### 3. 推理

#### 3.1 一键测试所有模型

```bash
# 运行测试脚本（会自动测试所有可用的Demo模型）
./test_inference.sh
```

#### 3.2 单独测试特定模型

```bash
# RNN模型 (jieba+NLTK)
python inference.py \
    --model_type rnn \
    --checkpoint checkpoints/rnn_jieba_nltk/rnn_best.pt \
    --input "记录指出 HMX-1 曾询问此次活动是否违反了该法案。"

# RNN模型 (HanLP+BPE)
python inference.py \
    --model_type rnn \
    --checkpoint checkpoints/rnn_hanlp_bpe/rnn_best.pt \
    --input "记录指出 HMX-1 曾询问此次活动是否违反了该法案。"

# Transformer模型 (jieba+NLTK)
python inference.py \
    --model_type transformer \
    --checkpoint checkpoints/transformer_jieba_nltk/transformer_best.pt \
    --input "白宫将此次"美国制造"活动定义为官方活动，因此不受《哈奇法案》管辖。"

# Transformer模型 (HanLP+BPE)
python inference.py \
    --model_type transformer \
    --checkpoint checkpoints/transformer_hanlp_bpe/transformer_best.pt \
    --input "白宫将此次"美国制造"活动定义为官方活动，因此不受《哈奇法案》管辖。"

# T5模型
python inference.py \
    --model_type t5 \
    --checkpoint checkpoints/t5_best \
    --input ""听起来你被锁住了啊，"副司令回复道。"

# Demo RNN模型（快速测试）
python inference.py \
    --model_type rnn \
    --checkpoint checkpoints/demo_rnn/rnn_best.pt \
    --input "全球经济缓慢增长又一年"

# Demo Transformer模型（快速测试）
python inference.py \
    --model_type transformer \
    --checkpoint checkpoints/demo_transformer/transformer_best.pt \
    --input "全球经济缓慢增长又一年"
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
