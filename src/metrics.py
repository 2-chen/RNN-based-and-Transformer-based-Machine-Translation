"""
评估指标实现
包括BLEU分数计算
"""

from sacrebleu import BLEU
from typing import List


def calculate_bleu(references: List[List[str]], hypotheses: List[str]) -> float:
    """
    计算BLEU分数
    
    Args:
        references: 参考翻译列表，每个元素是词列表
        hypotheses: 生成的翻译列表，每个元素是词列表（需要转换为字符串）
    """
    # 将词列表转换为字符串
    refs = [[' '.join(ref)] for ref in references]
    hyps = [' '.join(hyp) for hyp in hypotheses]
    
    bleu = BLEU()
    score = bleu.corpus_score(hyps, refs)
    return score.score


def calculate_bleu_sentence(reference: List[str], hypothesis: List[str]) -> float:
    """计算单个句子的BLEU分数"""
    ref_str = ' '.join(reference)
    hyp_str = ' '.join(hypothesis)
    
    bleu = BLEU()
    score = bleu.sentence_score(hyp_str, [ref_str])
    return score.score
