# 推送到GitHub的步骤

代码已经准备好并提交到本地Git仓库，现在需要推送到GitHub。

## 当前状态

✅ 本地Git仓库已初始化
✅ 所有文件已添加到暂存区
✅ 已创建初始提交
✅ 已添加远程仓库：https://github.com/2-chen/RNN-based-and-Transformer-based-Machine-Translation.git

## 推送步骤

### 方法1：使用Personal Access Token（推荐）

1. **创建Personal Access Token**：
   - 访问：https://github.com/settings/tokens
   - 点击 "Generate new token (classic)"
   - 设置权限：至少勾选 `repo`
   - 生成后复制token（只显示一次，请保存好）

2. **推送代码**：
```bash
cd /data/nlpfinal/AP0004_Midterm&Final_translation_dataset_zh_en

# 推送（会提示输入用户名和密码）
git push -u origin main

# 用户名：2-chen
# 密码：使用刚才创建的Personal Access Token（不是GitHub密码）
```

### 方法2：使用SSH（如果已配置SSH密钥）

```bash
# 更改远程仓库URL为SSH
git remote set-url origin git@github.com:2-chen/RNN-based-and-Transformer-based-Machine-Translation.git

# 推送
git push -u origin main
```

### 方法3：使用GitHub CLI

```bash
# 安装GitHub CLI后
gh auth login
git push -u origin main
```

## 验证上传

推送成功后，访问以下URL验证：
https://github.com/2-chen/RNN-based-and-Transformer-based-Machine-Translation

## 测试inference.py

上传完成后，可以在README.md中看到测试说明，或直接运行：

```bash
# 一键测试所有模型
./test_inference.sh

# 或单独测试
python inference.py --model_type rnn --checkpoint checkpoints/rnn_jieba_nltk/rnn_best.pt --input "你好世界"
```

## 注意事项

1. **模型文件**：`.gitignore`已配置忽略`.pt`和`.pth`文件（模型文件太大），只上传代码和配置文件
2. **数据文件**：如果数据文件太大，可能也需要添加到`.gitignore`
3. **认证**：首次推送需要输入GitHub用户名和Personal Access Token

