# GitHub 上传指南

本指南将帮助你将项目上传到GitHub。

## 方法一：使用Git命令行（推荐）

### 1. 准备工作

#### 1.1 安装Git
如果还没有安装Git，请先安装：
- Windows: 下载 https://git-scm.com/download/win
- Linux: `sudo apt-get install git` 或 `sudo yum install git`
- Mac: `brew install git` 或从官网下载

#### 1.2 配置Git（首次使用）
```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

### 2. 在GitHub上创建仓库

1. 登录GitHub (https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - Repository name: `chinese-english-mt` (或你喜欢的名字)
   - Description: `中英文机器翻译项目 - RNN, Transformer, T5`
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为本地已有文件）
4. 点击 "Create repository"

### 3. 初始化本地Git仓库

在项目根目录下执行：

```bash
# 进入项目目录
cd /data/nlpfinal/AP0004_Midterm&Final_translation_dataset_zh_en

# 初始化Git仓库
git init

# 添加所有文件（除了.gitignore中指定的）
git add .

# 创建初始提交
git commit -m "Initial commit: 中英文机器翻译项目"
```

### 4. 创建 .gitignore 文件

在项目根目录创建 `.gitignore` 文件，内容如下：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# 模型检查点（如果太大可以不上传）
checkpoints/*.pt
checkpoints/*.pth
checkpoints/*.ckpt
# 但保留结果JSON文件
!checkpoints/*.json

# 预训练模型（通常太大，不上传）
models/
*.bin
*.safetensors

# 数据文件（如果太大）
# data/*.jsonl

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# 系统文件
.DS_Store
Thumbs.db

# 实验临时文件
experiment_results/
*.log

# Jupyter Notebook
.ipynb_checkpoints/

# 其他
*.pdf
!Midterm&Final\ project.pdf
```

### 5. 连接到GitHub仓库

```bash
# 添加远程仓库（替换为你的GitHub用户名和仓库名）
git remote add origin https://github.com/你的用户名/chinese-english-mt.git

# 或者使用SSH（如果配置了SSH密钥）
# git remote add origin git@github.com:你的用户名/chinese-english-mt.git

# 查看远程仓库
git remote -v
```

### 6. 上传代码

```bash
# 推送代码到GitHub
git branch -M main  # 将分支重命名为main（GitHub默认分支名）
git push -u origin main
```

如果提示输入用户名和密码：
- 用户名：你的GitHub用户名
- 密码：使用Personal Access Token（不是GitHub密码）

#### 6.1 创建Personal Access Token（如果需要）

1. GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
2. 点击 "Generate new token (classic)"
3. 设置权限：至少勾选 `repo`
4. 生成后复制token（只显示一次，请保存好）
5. 使用token作为密码

### 7. 后续更新

如果之后修改了代码，更新到GitHub：

```bash
# 查看修改的文件
git status

# 添加修改的文件
git add .

# 提交修改
git commit -m "描述你的修改"

# 推送到GitHub
git push
```

---

## 方法二：使用GitHub Desktop（图形界面）

### 1. 下载GitHub Desktop
- 访问 https://desktop.github.com/
- 下载并安装

### 2. 登录GitHub账户
- 打开GitHub Desktop
- 登录你的GitHub账户

### 3. 创建仓库
1. File -> New Repository
2. 填写信息：
   - Name: `chinese-english-mt`
   - Local path: 选择项目目录
   - 勾选 "Initialize this repository with a README"（可选）
3. 点击 "Create repository"

### 4. 提交和推送
1. 在GitHub Desktop中，你会看到所有修改的文件
2. 在左下角填写提交信息，如 "Initial commit"
3. 点击 "Commit to main"
4. 点击 "Publish repository" 或 "Push origin"

---

## 方法三：使用VS Code的Git功能

### 1. 安装Git扩展
- VS Code通常自带Git支持
- 如果没有，安装 "Git" 扩展

### 2. 初始化仓库
1. 在VS Code中打开项目文件夹
2. 点击左侧的源代码管理图标（或按 Ctrl+Shift+G）
3. 点击 "Initialize Repository"

### 3. 提交和推送
1. 在源代码管理面板中，你会看到所有修改的文件
2. 点击 "+" 号暂存所有文件（或选择特定文件）
3. 在上方输入提交信息
4. 点击 "✓" 提交
5. 点击 "..." 菜单，选择 "Push" -> "Push to..."

---

## 重要提示

### 1. 文件大小限制
- GitHub单个文件限制：100MB
- 仓库总大小建议：不超过1GB
- 如果模型文件太大，建议：
  - 使用Git LFS（Large File Storage）
  - 或不上传模型文件，只上传代码和配置

### 2. 敏感信息
**不要上传**：
- API密钥
- 密码
- 个人敏感数据
- 大型数据文件（如果涉及隐私）

### 3. 推荐的仓库结构

```
chinese-english-mt/
├── README.md              # 项目说明
├── requirements.txt       # 依赖包
├── .gitignore            # Git忽略文件
├── inference.py          # 推理脚本
├── run_experiments.py    # 实验脚本
├── configs/              # 配置文件
│   ├── rnn_config.yaml
│   ├── transformer_config.yaml
│   └── t5_config.yaml
├── src/                  # 源代码
│   ├── data_utils.py
│   ├── rnn_model.py
│   ├── transformer_model.py
│   ├── train_rnn.py
│   ├── train_transformer.py
│   ├── train_t5.py
│   └── metrics.py
├── data/                 # 数据（可选，如果不大）
│   └── README.md         # 说明数据来源
└── checkpoints/          # 模型检查点（可选）
    └── README.md         # 说明如何下载模型
```

### 4. 更新README.md

确保你的README.md包含：
- 项目简介
- 安装说明
- 使用方法
- 实验结果
- 代码仓库链接

---

## 快速命令参考

```bash
# 初始化仓库
git init

# 添加文件
git add .

# 提交
git commit -m "提交信息"

# 添加远程仓库
git remote add origin https://github.com/用户名/仓库名.git

# 推送
git push -u origin main

# 查看状态
git status

# 查看提交历史
git log

# 拉取更新
git pull

# 创建新分支
git checkout -b 分支名

# 切换分支
git checkout 分支名
```

---

## 常见问题

### Q1: 如何更新已存在的仓库？
```bash
git add .
git commit -m "更新说明"
git push
```

### Q2: 如何删除已上传的文件？
```bash
git rm 文件名
git commit -m "删除文件"
git push
```

### Q3: 如何忽略已跟踪的文件？
```bash
# 从Git中移除但保留本地文件
git rm --cached 文件名
# 添加到.gitignore
echo "文件名" >> .gitignore
git commit -m "忽略文件"
```

### Q4: 如何回退到之前的版本？
```bash
# 查看提交历史
git log

# 回退到指定提交
git reset --hard 提交ID

# 强制推送（谨慎使用）
git push -f
```

---

## 完成后的检查清单

- [ ] 创建了GitHub仓库
- [ ] 初始化了本地Git仓库
- [ ] 创建了.gitignore文件
- [ ] 添加了所有必要的文件
- [ ] 提交了初始版本
- [ ] 推送到GitHub成功
- [ ] 更新了README.md
- [ ] 在项目报告中添加了GitHub链接

---

**完成后，记得在项目报告的第一页更新代码仓库URL！**

