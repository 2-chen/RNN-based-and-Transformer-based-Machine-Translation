"""
数据预处理工具模块
包括数据加载、清洗、分词、词汇表构建等功能
使用HanLP进行中文分词，BPE进行英文分词
"""

import json
import re
import os
from collections import Counter
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 导入HanLP（延迟检查，避免模块加载时的错误）
HANLP_AVAILABLE = None  # None表示未检查，True/False表示检查结果

# 导入BPE tokenizer
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import BertProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("警告: tokenizers库未安装，将回退到NLTK分词")

# 全局变量：HanLP模型和BPE tokenizer
_hanlp_model = None
_bpe_tokenizer = None

# 特殊标记
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    """词汇表类"""
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
    def build_vocab(self, sentences: List[List[str]]):
        """构建词汇表"""
        # 统计词频
        for sentence in sentences:
            self.word_count.update(sentence)
        
        # 添加特殊标记
        for token in SPECIAL_TOKENS:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
        
        # 添加满足最小频率的词
        for word, count in self.word_count.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
    
    def encode(self, sentence: List[str]) -> List[int]:
        """将句子编码为索引序列"""
        return [self.word2idx.get(word, self.word2idx[UNK_TOKEN]) 
                for word in sentence]
    
    def decode(self, indices: List[int]) -> List[str]:
        """将索引序列解码为词序列"""
        return [self.idx2word.get(idx, UNK_TOKEN) for idx in indices]
    
    def __len__(self):
        return len(self.word2idx)


def clean_text(text: str, lang: str = 'en') -> str:
    """清洗文本"""
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符（保留基本标点）
    if lang == 'en':
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    else:  # 中文
        text = re.sub(r'[^\u4e00-\u9fa5\w\s.,!?;:\'\"-]', '', text)
    return text.strip()


def _init_hanlp():
    """初始化HanLP模型（延迟加载）"""
    global _hanlp_model, HANLP_AVAILABLE
    
    # 延迟检查HanLP是否可用
    if HANLP_AVAILABLE is None:
        try:
            import hanlp
            HANLP_AVAILABLE = True
        except (ImportError, Exception):
            HANLP_AVAILABLE = False
    
    if _hanlp_model is None and HANLP_AVAILABLE:
        try:
            # 尝试加载HanLP的基础分词模型
            # HanLP 2.x版本使用不同的API
            try:
                # 方法1: 使用基础分词模型
                _hanlp_model = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            except:
                try:
                    # 方法2: 使用多任务模型（包含分词）
                    _hanlp_model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
                except:
                    # 方法3: 尝试直接使用字符串路径
                    _hanlp_model = hanlp.load('COARSE_ELECTRA_SMALL_ZH')
        except Exception as e:
            print(f"警告: 无法加载HanLP模型 ({e})，将使用jieba作为备选")
            import jieba
            _hanlp_model = jieba
    elif not HANLP_AVAILABLE:
        import jieba
        _hanlp_model = jieba
    return _hanlp_model


def tokenize_zh(text: str) -> List[str]:
    """中文分词 - 使用HanLP"""
    text = clean_text(text, lang='zh')
    if not text:
        return []
    
    model = _init_hanlp()
    
    if model is None:
        # 完全失败时使用简单空格分割
        return text.split()
    
    try:
        # 尝试不同的HanLP API
        if hasattr(model, 'tok'):
            # HanLP多任务模型的tok方法
            tokens = model.tok(text)
            if isinstance(tokens, list):
                return tokens
            elif hasattr(tokens, 'tolist'):
                return tokens.tolist()
            else:
                return list(tokens)
        elif hasattr(model, '__call__'):
            # 直接调用模型
            result = model(text)
            if isinstance(result, dict):
                # 多任务模型返回字典
                if 'tok' in result:
                    return result['tok'] if isinstance(result['tok'], list) else list(result['tok'])
                elif 'tok/fine' in result:
                    return result['tok/fine'] if isinstance(result['tok/fine'], list) else list(result['tok/fine'])
            elif isinstance(result, list):
                return result
            elif hasattr(result, 'tolist'):
                return result.tolist()
        elif hasattr(model, 'cut'):
            # jieba模型（备选）
            return list(model.cut(text))
        
        # 如果都不匹配，尝试简单分割
        return text.split()
    except Exception as e:
        # 出错时使用jieba作为备选
        try:
            import jieba
            return list(jieba.cut(text))
        except:
            return text.split()


def _init_bpe_tokenizer(vocab_size: int = 30000, min_frequency: int = 2):
    """初始化BPE tokenizer"""
    global _bpe_tokenizer
    if _bpe_tokenizer is None and TOKENIZERS_AVAILABLE:
        _bpe_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        _bpe_tokenizer.pre_tokenizer = Whitespace()
        _bpe_tokenizer.post_processor = BertProcessing(
            ("</s>", 2),  # EOS token
            ("<s>", 1)    # BOS token
        )
    return _bpe_tokenizer


def train_bpe_tokenizer(train_texts: List[str], vocab_size: int = 30000, 
                       min_frequency: int = 2, save_path: Optional[str] = None):
    """从训练文本训练BPE tokenizer"""
    if not TOKENIZERS_AVAILABLE:
        print("警告: tokenizers库不可用，无法训练BPE")
        return None
    
    print(f"训练BPE tokenizer (vocab_size={vocab_size}, min_frequency={min_frequency})...")
    
    tokenizer = _init_bpe_tokenizer(vocab_size, min_frequency)
    
    # 准备训练数据（每行一个句子）
    def get_training_corpus():
        for text in train_texts:
            yield text.lower()  # BPE通常使用小写
    
    # 训练BPE
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<unk>", "<sos>", "<eos>", "<s>", "</s>"]
    )
    
    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    
    # 保存tokenizer
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        tokenizer.save(save_path)
        print(f"BPE tokenizer已保存到: {save_path}")
    
    global _bpe_tokenizer
    _bpe_tokenizer = tokenizer
    return tokenizer


def load_bpe_tokenizer(load_path: str):
    """加载已训练的BPE tokenizer"""
    if not TOKENIZERS_AVAILABLE:
        print("警告: tokenizers库不可用")
        return None
    
    try:
        global _bpe_tokenizer
        _bpe_tokenizer = Tokenizer.from_file(load_path)
        print(f"BPE tokenizer已从 {load_path} 加载")
        return _bpe_tokenizer
    except Exception as e:
        print(f"加载BPE tokenizer失败: {e}")
        return None


def tokenize_en(text: str) -> List[str]:
    """英文分词 - 使用BPE"""
    text = clean_text(text, lang='en')
    
    if _bpe_tokenizer is None:
        # 如果BPE未初始化，回退到简单分词
        if not TOKENIZERS_AVAILABLE:
            # 完全回退：使用简单空格分割
            return text.lower().split()
        else:
            # 使用基础空格分割
            return text.lower().split()
    
    try:
        # 使用BPE tokenizer
        encoded = _bpe_tokenizer.encode(text.lower())
        tokens = encoded.tokens
        
        # 移除特殊标记（在后续处理中会重新添加）
        tokens = [t for t in tokens if t not in ["<s>", "</s>", "<sos>", "<eos>", "<pad>", "<unk>"]]
        return tokens
    except Exception as e:
        print(f"BPE分词错误: {e}，使用简单分割")
        return text.lower().split()


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def prepare_data(
    train_file: str,
    valid_file: str,
    test_file: str,
    max_length: int = 50,
    min_freq: int = 2,
    bpe_vocab_size: int = 30000,
    bpe_model_path: Optional[str] = None
) -> Tuple[Vocabulary, Vocabulary, List, List, List]:
    """
    准备数据并构建词汇表
    
    Args:
        train_file: 训练文件路径
        valid_file: 验证文件路径
        test_file: 测试文件路径
        max_length: 最大序列长度
        min_freq: 最小词频
        bpe_vocab_size: BPE词汇表大小
        bpe_model_path: BPE模型保存/加载路径（如果存在则加载，否则训练后保存）
    
    Returns:
        src_vocab: 源语言词汇表（中文）
        tgt_vocab: 目标语言词汇表（英文）
        train_data: 训练数据
        valid_data: 验证数据
        test_data: 测试数据
    """
    print("加载数据...")
    train_data = load_jsonl(train_file)
    valid_data = load_jsonl(valid_file)
    test_data = load_jsonl(test_file)
    
    # 训练或加载BPE tokenizer
    if TOKENIZERS_AVAILABLE:
        if bpe_model_path and os.path.exists(bpe_model_path):
            print(f"加载BPE模型: {bpe_model_path}")
            load_bpe_tokenizer(bpe_model_path)
        else:
            print("训练BPE tokenizer...")
            # 准备英文训练文本
            train_en_texts = [clean_text(item['en'], lang='en').lower() for item in train_data]
            # 训练BPE
            train_bpe_tokenizer(
                train_en_texts, 
                vocab_size=bpe_vocab_size, 
                min_frequency=min_freq,
                save_path=bpe_model_path
            )
    
    print("分词...")
    # 中文分词（使用HanLP）
    print("  中文分词（HanLP）...")
    train_zh = [tokenize_zh(item['zh']) for item in train_data]
    
    # 英文分词（使用BPE）
    print("  英文分词（BPE）...")
    train_en = [tokenize_en(item['en']) for item in train_data]
    
    # 过滤过长句子
    filtered_train = []
    filtered_zh = []
    filtered_en = []
    for zh, en, item in zip(train_zh, train_en, train_data):
        if len(zh) <= max_length and len(en) <= max_length:
            filtered_train.append(item)
            filtered_zh.append(zh)
            filtered_en.append(en)
    
    print(f"过滤后训练样本数: {len(filtered_train)}/{len(train_data)}")
    
    # 构建词汇表
    print("构建词汇表...")
    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)
    
    src_vocab.build_vocab(filtered_zh)
    tgt_vocab.build_vocab(filtered_en)
    
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 处理验证集和测试集
    valid_zh = [tokenize_zh(item['zh']) for item in valid_data]
    valid_en = [tokenize_en(item['en']) for item in valid_data]
    test_zh = [tokenize_zh(item['zh']) for item in test_data]
    test_en = [tokenize_en(item['en']) for item in test_data]
    
    # 过滤验证集和测试集
    filtered_valid = []
    filtered_test = []
    for zh, en, item in zip(valid_zh, valid_en, valid_data):
        if len(zh) <= max_length and len(en) <= max_length:
            filtered_valid.append(item)
    for zh, en, item in zip(test_zh, test_en, test_data):
        if len(zh) <= max_length and len(en) <= max_length:
            filtered_test.append(item)
    
    return src_vocab, tgt_vocab, filtered_train, filtered_valid, filtered_test


class TranslationDataset(Dataset):
    """翻译数据集类"""
    
    def __init__(
        self,
        data: List[Dict],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_length: int = 50
    ):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = tokenize_zh(item['zh'])
        tgt_text = tokenize_en(item['en'])
        
        # 编码
        src_ids = [self.src_vocab.word2idx.get(w, self.src_vocab.word2idx[UNK_TOKEN]) 
                   for w in src_text]
        tgt_ids = [self.tgt_vocab.word2idx.get(w, self.tgt_vocab.word2idx[UNK_TOKEN]) 
                   for w in tgt_text]
        
        # 添加SOS和EOS标记
        src_ids = [self.src_vocab.word2idx[SOS_TOKEN]] + src_ids + [self.src_vocab.word2idx[EOS_TOKEN]]
        tgt_ids = [self.tgt_vocab.word2idx[SOS_TOKEN]] + tgt_ids + [self.tgt_vocab.word2idx[EOS_TOKEN]]
        
        # 截断或填充
        src_ids = src_ids[:self.max_length]
        tgt_ids = tgt_ids[:self.max_length]
        
        src_ids = src_ids + [self.src_vocab.word2idx[PAD_TOKEN]] * (self.max_length - len(src_ids))
        tgt_ids = tgt_ids + [self.tgt_vocab.word2idx[PAD_TOKEN]] * (self.max_length - len(tgt_ids))
        
        return {
            'src': torch.LongTensor(src_ids),
            'tgt': torch.LongTensor(tgt_ids),
            'src_text': item['zh'],
            'tgt_text': item['en']
        }


def collate_fn(batch):
    """批处理函数"""
    src = torch.stack([item['src'] for item in batch])
    tgt = torch.stack([item['tgt'] for item in batch])
    src_text = [item['src_text'] for item in batch]
    tgt_text = [item['tgt_text'] for item in batch]
    
    return {
        'src': src,
        'tgt': tgt,
        'src_text': src_text,
        'tgt_text': tgt_text
    }
