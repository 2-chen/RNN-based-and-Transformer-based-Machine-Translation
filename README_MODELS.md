# 模型文件说明

## 模型文件未上传的原因

由于GitHub对单个文件有**100MB的限制**，所有模型文件都无法直接上传：

- ❌ `checkpoints/rnn_jieba_nltk/rnn_best.pt` (197MB) - **超过限制**
- ❌ `checkpoints/rnn_hanlp_bpe/rnn_best.pt` (103MB) - **超过限制**

## 所有模型文件大小

- ❌ `checkpoints/rnn_jieba_nltk/rnn_best.pt` (197MB) - 超过100MB限制
- ❌ `checkpoints/rnn_hanlp_bpe/rnn_best.pt` (103MB) - 超过100MB限制
- ❌ `checkpoints/transformer_jieba_nltk/transformer_best.pt` (833MB) - 超过100MB限制
- ❌ `checkpoints/transformer_hanlp_bpe/transformer_best.pt` (679MB) - 超过100MB限制
- ❌ `checkpoints/t5_best/` (852MB) - 超过100MB限制

## 解决方案

### 方案1：使用Git LFS（推荐）

如果你有sudo权限，可以安装Git LFS：

```bash
# 安装Git LFS
sudo apt-get install git-lfs

# 初始化
git lfs install

# 跟踪大文件
git lfs track "checkpoints/transformer_jieba_nltk/transformer_best.pt"
git lfs track "checkpoints/transformer_hanlp_bpe/transformer_best.pt"
git lfs track "checkpoints/t5_best/**"

# 添加并提交
git add .gitattributes
git add checkpoints/transformer_jieba_nltk/transformer_best.pt
git add checkpoints/transformer_hanlp_bpe/transformer_best.pt
git add checkpoints/t5_best/
git commit -m "使用Git LFS添加大模型文件"
git push origin main
```

### 方案2：使用GitHub Releases

1. 将大文件打包：
```bash
tar -czf large_models.tar.gz \
    checkpoints/transformer_jieba_nltk/transformer_best.pt \
    checkpoints/transformer_hanlp_bpe/transformer_best.pt \
    checkpoints/t5_best/
```

2. 访问 https://github.com/2-chen/RNN-based-and-Transformer-based-Machine-Translation/releases
3. 创建新Release并上传 `large_models.tar.gz`

### 方案3：本地训练

如果只需要测试 `inference.py`，可以：
1. 使用已上传的RNN模型进行测试
2. 或者本地训练Transformer和T5模型

## 测试脚本行为

`test_inference.sh` 脚本会：
- ✅ 测试已上传的RNN模型（jieba+NLTK和HanLP+BPE版本）
- ⚠️ 跳过未上传的Transformer和T5模型（显示警告信息）

这是正常行为，不影响脚本运行。

