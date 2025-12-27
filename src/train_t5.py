"""
T5预训练模型微调脚本
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import json
import yaml
from data_utils import load_jsonl


class T5Dataset(Dataset):
    """T5数据集"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        source = f"translate Chinese to English: {item['zh']}"
        target = item['en']
        
        # 编码
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, 
                gradient_accumulation_steps=1, use_amp=False, scaler=None, max_grad_norm=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    accumulation_steps = 0
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 混合精度训练
        try:
            if use_amp:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    # 确保loss是标量（在多GPU环境下可能需要）
                    loss = outputs.loss
                    # 在多GPU环境下，DataParallel可能返回非标量
                    if isinstance(loss, torch.Tensor):
                        if loss.dim() > 0:
                            loss = loss.mean()
                    loss = loss / gradient_accumulation_steps
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # 确保loss是标量
                loss = outputs.loss
                # 在多GPU环境下，DataParallel可能返回非标量
                if isinstance(loss, torch.Tensor):
                    if loss.dim() > 0:
                        loss = loss.mean()
                loss = loss / gradient_accumulation_steps
            
            # 检查loss是否有效
            if torch.isnan(loss) or torch.isinf(loss) or not torch.isfinite(loss):
                print(f"警告: 检测到无效loss (step {step}), 跳过此batch")
                # 清零梯度以避免累积无效梯度
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                continue
        except Exception as e:
            print(f"错误: 前向传播失败 (step {step}): {e}")
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            continue
        
        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accumulation_steps += 1
        # 使用处理过的loss值（已经是标量）
        loss_value = loss.item() * gradient_accumulation_steps  # 恢复原始loss值用于统计
        if not (torch.isnan(loss) or torch.isinf(loss) or not torch.isfinite(loss)):
            total_loss += loss_value
        else:
            print(f"警告: 检测到无效的loss值 (step {step}), 跳过统计")
        
        # 梯度累积：每 accumulation_steps 步更新一次
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            if max_grad_norm > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
            else:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            
            scheduler.step()  # 在 optimizer.step() 之后调用
            optimizer.zero_grad()
            accumulation_steps = 0
    
    # 处理剩余的梯度
    if accumulation_steps > 0:
        if max_grad_norm > 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        else:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        scheduler.step()  # 在 optimizer.step() 之后调用
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, tokenizer, device):
    """评估模型"""
    # 如果是DataParallel，获取实际模型
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    actual_model.eval()
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 生成（使用实际模型）
            generated_ids = actual_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # 解码
            hypotheses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_hypotheses.extend([h.split() for h in hypotheses])
            all_references.extend([r.split() for r in references])
    
    # 计算BLEU
    from metrics import calculate_bleu
    bleu_score = calculate_bleu(all_references, all_hypotheses)
    return bleu_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='配置文件路径 (YAML格式)')
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--use_amp', action='store_true', default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', action='store_true', default=None)
    parser.add_argument('--use_multi_gpu', action='store_true', default=None)
    
    args = parser.parse_args()
    
    # 默认值
    defaults = {
        'train_file': 'data/train_10k.jsonl',
        'valid_file': 'data/valid.jsonl',
        'test_file': 'data/test.jsonl',
        'model_name': 't5-base',
        'batch_size': 8,
        'epochs': 5,
        'lr': 5e-5,
        'max_length': 128,
        'save_dir': 'checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gradient_accumulation_steps': 1,
        'warmup_steps': None,  # 如果为None，将使用总步数的10%
        'max_grad_norm': 1.0,
        'use_amp': False,
        'num_workers': 4,
        'pin_memory': True,
        'use_multi_gpu': False
    }
    
    # 如果提供了配置文件，从YAML加载配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 用配置文件中的值更新默认值
            for key, value in config.items():
                if key in defaults:
                    defaults[key] = value
    
    # 用命令行参数覆盖配置（如果提供了）
    for key in defaults:
        value = getattr(args, key)
        if value is not None:
            defaults[key] = value
    
    # 类型转换：确保数值类型正确
    type_conversions = {
        'batch_size': int,
        'epochs': int,
        'lr': float,
        'max_length': int,
        'gradient_accumulation_steps': int,
        'warmup_steps': int,
        'max_grad_norm': float,
        'num_workers': int
    }
    for key, convert_func in type_conversions.items():
        if key in defaults:
            try:
                value = defaults[key]
                # 如果是字符串，尝试转换
                if isinstance(value, str):
                    defaults[key] = convert_func(value)
                # 如果类型不匹配，强制转换
                elif convert_func == float and not isinstance(value, float):
                    defaults[key] = float(value)
                elif convert_func == int and not isinstance(value, int):
                    defaults[key] = int(value)
            except (ValueError, TypeError) as e:
                print(f"警告: 无法将 {key}={defaults[key]} 转换为 {convert_func.__name__}，使用默认值")
                # 使用原始默认值
                pass
    
    # 布尔值转换（YAML中的true/false会被解析为bool，但命令行参数需要特殊处理）
    bool_keys = ['use_amp', 'pin_memory', 'use_multi_gpu']
    for key in bool_keys:
        if key in defaults:
            value = defaults[key]
            if isinstance(value, str):
                defaults[key] = value.lower() in ('true', '1', 'yes', 'on')
            # YAML中的布尔值已经是bool类型，直接使用
    for key, convert_func in type_conversions.items():
        if key in defaults:
            try:
                value = defaults[key]
                # 如果是字符串，尝试转换
                if isinstance(value, str):
                    defaults[key] = convert_func(value)
                # 如果类型不匹配，强制转换
                elif convert_func == float and not isinstance(value, float):
                    defaults[key] = float(value)
                elif convert_func == int and not isinstance(value, int):
                    defaults[key] = int(value)
            except (ValueError, TypeError) as e:
                print(f"警告: 无法将 {key}={defaults[key]} 转换为 {convert_func.__name__}，使用默认值")
                # 使用原始默认值
                pass
    
    # 将配置赋值给args对象
    for key, value in defaults.items():
        setattr(args, key, value)
    
    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 张GPU")
    
    # 多GPU设置
    if args.use_multi_gpu and num_gpus > 1:
        print(f"使用 {num_gpus} 张GPU进行训练")
        device = torch.device('cuda:0')
    else:
        device = torch.device(args.device)
        if args.use_multi_gpu and num_gpus <= 1:
            print("警告: 请求使用多GPU但只有1张GPU可用，使用单GPU训练")
    
    # 加载tokenizer和模型
    print("加载T5模型...")
    
    # 检查是否是本地路径
    model_path = args.model_name
    local_files_only = False
    
    # 检查是否是本地文件路径（相对路径或绝对路径）
    if os.path.exists(model_path) and os.path.isdir(model_path):
        # 检查是否是有效的模型目录
        if os.path.exists(os.path.join(model_path, 'config.json')):
            local_files_only = True
            print(f"✓ 检测到本地模型路径: {os.path.abspath(model_path)}")
        else:
            print(f"警告: {model_path} 存在但不是有效的模型目录（缺少 config.json）")
    elif not os.path.exists(model_path):
        # 尝试相对路径（相对于脚本所在目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # src 的父目录
        relative_path = os.path.join(project_root, model_path)
        
        if os.path.exists(relative_path) and os.path.isdir(relative_path):
            if os.path.exists(os.path.join(relative_path, 'config.json')):
                model_path = relative_path
                local_files_only = True
                print(f"✓ 检测到本地模型路径: {os.path.abspath(model_path)}")
    
    if not local_files_only:
        print(f"尝试从 Hugging Face 下载模型: {model_path}")
        print("提示: 如果网络不可用，请确保模型路径正确")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=local_files_only
        ).to(device)
        
        # 多GPU支持
        if args.use_multi_gpu and num_gpus > 1:
            model = torch.nn.DataParallel(model)
            print(f"✓ 模型已部署到 {num_gpus} 张GPU")
    except Exception as e:
        error_msg = str(e)
        if "Network is unreachable" in error_msg or "Connection" in error_msg or "Failed to establish" in error_msg:
            print("\n❌ 错误: 无法连接到 Hugging Face，网络不可用")
            print(f"\n当前模型路径: {model_path}")
            print(f"路径是否存在: {os.path.exists(model_path)}")
            print("\n解决方案:")
            print("1. 检查配置文件中的 model_name 是否正确指向本地模型目录")
            print("2. 确保模型文件已完整上传到服务器")
            print("3. 检查模型目录是否包含 config.json 文件")
            raise RuntimeError("网络连接失败，请使用本地模型或检查网络设置") from e
        else:
            raise
    
    # 准备数据
    print("准备数据...")
    train_data = load_jsonl(args.train_file)
    valid_data = load_jsonl(args.valid_file)
    test_data = load_jsonl(args.test_file)
    
    train_dataset = T5Dataset(train_data, tokenizer, args.max_length)
    valid_dataset = T5Dataset(valid_data, tokenizer, args.max_length)
    test_dataset = T5Dataset(test_data, tokenizer, args.max_length)
    
    # 优化DataLoader性能
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # 优化器和调度器
    # 如果是多GPU，需要获取实际模型
    model_for_optimizer = model.module if isinstance(model, torch.nn.DataParallel) else model
    optimizer = AdamW(model_for_optimizer.parameters(), lr=args.lr)
    
    # 计算训练步数（考虑梯度累积）
    num_training_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else num_training_steps // 10
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 混合精度训练
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("✓ 启用混合精度训练 (AMP)")
    
    print(f"训练配置:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"  有效batch size: {args.batch_size * args.gradient_accumulation_steps * (num_gpus if args.use_multi_gpu else 1)}")
    print(f"  总训练步数: {num_training_steps}")
    print(f"  Warmup步数: {warmup_steps}")
    
    # 训练
    best_bleu = 0
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scheduler, 
            device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=args.use_amp,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        valid_bleu = evaluate(model, valid_loader, tokenizer, device)
        print(f"Valid BLEU: {valid_bleu:.4f}")
        
        # 保存最佳模型
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            save_path = os.path.abspath(os.path.join(args.save_dir, 't5_best'))
            # 如果是多GPU，需要获取实际模型
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"保存最佳模型，BLEU: {best_bleu:.4f}")
    
    # 测试
    print("\n在测试集上评估...")
    save_path = os.path.abspath(os.path.join(args.save_dir, 't5_best'))
    if os.path.exists(save_path):
        model = T5ForConditionalGeneration.from_pretrained(save_path).to(device)
        test_bleu = evaluate(model, test_loader, tokenizer, device)
        print(f"Test BLEU: {test_bleu:.4f}")
    else:
        print("未发现最佳模型文件，跳过测试。")
    
    # 保存结果
    results = {
        'best_valid_bleu': best_bleu,
        'test_bleu': test_bleu,
        'args': vars(args)
    }
    with open(os.path.join(args.save_dir, 't5_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
