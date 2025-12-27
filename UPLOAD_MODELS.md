# 模型文件上传指南

## 问题说明

### 1. 测试脚本能否在没有pt文件时运行？

**答案：可以运行，但会跳过所有测试。**

`test_inference.sh` 脚本有检查机制，如果模型文件不存在，会显示：
```
⚠ 警告: 模型文件不存在: checkpoints/xxx/rnn_best.pt
   请先训练模型或下载模型文件
   跳过（模型文件不存在）
```

脚本会继续运行，只是不会执行实际的推理测试。

## 模型文件大小

根据 `test_inference.sh` 需要的必要模型文件：

| 模型文件 | 大小 | 说明 |
|---------|------|------|
| `checkpoints/rnn_jieba_nltk/rnn_best.pt` | 197MB | RNN模型（jieba+NLTK版本） |
| `checkpoints/rnn_hanlp_bpe/rnn_best.pt` | 103MB | RNN模型（HanLP+BPE版本） |
| `checkpoints/transformer_jieba_nltk/transformer_best.pt` | 833MB | Transformer模型（jieba+NLTK版本） |
| `checkpoints/transformer_hanlp_bpe/transformer_best.pt` | 679MB | Transformer模型（HanLP+BPE版本） |
| `checkpoints/t5_best/` | 852MB | T5模型目录 |
| **总计** | **约2.6GB** | - |

## 上传方案

### 方案1：使用Git LFS（推荐，适合大文件）

Git LFS（Large File Storage）是Git的扩展，专门用于处理大文件。

#### 安装Git LFS

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# 初始化Git LFS
git lfs install
```

#### 配置Git LFS跟踪模型文件

```bash
# 跟踪所有.pt文件
git lfs track "checkpoints/rnn_jieba_nltk/rnn_best.pt"
git lfs track "checkpoints/rnn_hanlp_bpe/rnn_best.pt"
git lfs track "checkpoints/transformer_jieba_nltk/transformer_best.pt"
git lfs track "checkpoints/transformer_hanlp_bpe/transformer_best.pt"
git lfs track "checkpoints/t5_best/**"

# 或者使用通配符（更简单）
git lfs track "checkpoints/*/rnn_best.pt"
git lfs track "checkpoints/*/transformer_best.pt"
git lfs track "checkpoints/t5_best/**"

# 提交.gitattributes文件
git add .gitattributes
git commit -m "配置Git LFS跟踪模型文件"
```

#### 添加并上传模型文件

```bash
# 添加模型文件（会自动使用LFS）
git add checkpoints/rnn_jieba_nltk/rnn_best.pt
git add checkpoints/rnn_hanlp_bpe/rnn_best.pt
git add checkpoints/transformer_jieba_nltk/transformer_best.pt
git add checkpoints/transformer_hanlp_bpe/transformer_best.pt
git add checkpoints/t5_best/

# 提交
git commit -m "添加必要的模型文件用于一键测试"

# 推送到GitHub
git push origin main
```

### 方案2：只上传较小的模型文件（如果Git LFS不可用）

如果Git LFS不可用，可以只上传较小的RNN模型文件：

```bash
# 只上传RNN模型（约300MB，在GitHub限制内）
git add checkpoints/rnn_jieba_nltk/rnn_best.pt
git add checkpoints/rnn_hanlp_bpe/rnn_best.pt
git commit -m "添加RNN模型文件用于测试"
git push origin main
```

**注意**：Transformer和T5模型文件太大（>800MB），GitHub单个文件限制是100MB，无法直接上传。

### 方案3：使用GitHub Releases（推荐用于大文件）

将模型文件打包并上传到GitHub Releases：

```bash
# 创建压缩包
tar -czf models.tar.gz \
    checkpoints/rnn_jieba_nltk/rnn_best.pt \
    checkpoints/rnn_hanlp_bpe/rnn_best.pt \
    checkpoints/transformer_jieba_nltk/transformer_best.pt \
    checkpoints/transformer_hanlp_bpe/transformer_best.pt \
    checkpoints/t5_best/

# 上传到GitHub Releases（通过网页界面）
# 1. 访问 https://github.com/2-chen/RNN-based-and-Transformer-based-Machine-Translation/releases
# 2. 点击 "Draft a new release"
# 3. 上传 models.tar.gz
```

然后在README中说明如何下载：

```markdown
## 下载模型文件

模型文件已上传到 [GitHub Releases](https://github.com/2-chen/RNN-based-and-Transformer-based-Machine-Translation/releases)。

下载并解压：
```bash
wget https://github.com/2-chen/RNN-based-and-Transformer-based-Machine-Translation/releases/download/v1.0/models.tar.gz
tar -xzf models.tar.gz
```
```

## 当前.gitignore配置

已修改 `.gitignore`，允许上传以下必要的模型文件：

- `checkpoints/rnn_jieba_nltk/rnn_best.pt`
- `checkpoints/rnn_hanlp_bpe/rnn_best.pt`
- `checkpoints/transformer_jieba_nltk/transformer_best.pt`
- `checkpoints/transformer_hanlp_bpe/transformer_best.pt`
- `checkpoints/t5_best/`（整个目录）

其他模型文件仍会被忽略。

## 推荐方案

**推荐使用方案1（Git LFS）**，因为：
1. 可以上传所有必要的模型文件
2. 保持仓库结构完整
3. 用户克隆后可以直接使用 `test_inference.sh`

如果Git LFS不可用，使用方案3（GitHub Releases）作为备选。

