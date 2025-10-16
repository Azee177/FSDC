# -*- coding: utf-8 -*-  # æŒ‡å®šæºæ–‡ä»¶ç¼–ç 
from __future__ import annotations  # å¯ç”¨æœªæ¥æ³¨è§£

from pathlib import Path  # å¯¼å…¥è·¯å¾„ç±»
from typing import Dict, Iterable, List, Optional, Tuple  # å¯¼å…¥ç±»å‹åˆ«å

import matplotlib.pyplot as plt  # å¯¼å…¥ç»˜å›¾åº“
import numpy as np  # å¯¼å…¥numpyåº“
import torch  # å¯¼å…¥torchåº“
from sklearn.metrics import confusion_matrix  # å¯¼å…¥æ··æ·†çŸ©é˜µå‡½æ•°

from .common import ensure_dir  # å¯¼å…¥ç›®å½•å·¥å…·


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: Tuple[int, ...] = (1, 5)) -> Dict[int, float]:  # å®šä¹‰TopKç²¾åº¦å‡½æ•°
    max_k = max(ks)  # è®¡ç®—æœ€å¤§Kå€¼
    _, pred = torch.topk(logits, max_k, dim=1)  # è·å–å‰Ké¢„æµ‹
    pred = pred.t()  # è½¬ç½®ä»¥æ–¹ä¾¿æ¯”è¾ƒ
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # æ„å»ºå‘½ä¸­çŸ©é˜µ
    results: Dict[int, float] = {}  # åˆå§‹åŒ–ç»“æœå­—å…¸
    total = targets.size(0)  # è·å–æ ·æœ¬æ•°é‡
    for k in ks:  # éå†å„ä¸ªK
        correct_k = correct[:k].reshape(-1).float().sum(0).item()  # ç»Ÿè®¡å‘½ä¸­æ•°é‡
        results[k] = float(correct_k / max(1, total))  # è®¡ç®—TopKç²¾åº¦
    return results  # è¿”å›ç²¾åº¦å­—å…¸


def build_confusion(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:  # å®šä¹‰æ··æ·†çŸ©é˜µæ„å»ºå‡½æ•°
    preds = logits.argmax(dim=1).cpu().numpy()  # è·å–é¢„æµ‹ç±»åˆ«
    labels = targets.cpu().numpy()  # è·å–çœŸå®ç±»åˆ«
    matrix = confusion_matrix(labels, preds, labels=list(range(num_classes)))  # è®¡ç®—æ··æ·†çŸ©é˜µ
    return matrix  # è¿”å›æ··æ·†çŸ©é˜µ


def plot_confusion(matrix: np.ndarray, class_names: List[str], save_path: Path) -> None:  # å®šä¹‰æ··æ·†çŸ©é˜µç»˜åˆ¶å‡½æ•°
    ensure_dir(save_path.parent)  # ç¡®ä¿è¾“å‡ºç›®å½•å­˜åœ¨
    fig, ax = plt.subplots(figsize=(8, 8))  # åˆ›å»ºç”»å¸ƒ
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')  # ç»˜åˆ¶çƒ­åŠ›å›¾
    ax.figure.colorbar(im, ax=ax)  # æ·»åŠ è‰²æ¡
    ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]))  # è®¾ç½®åæ ‡åˆ»åº¦
    ax.set_xticklabels(class_names, rotation=45, ha='right')  # è®¾ç½®æ¨ªè½´æ ‡ç­¾
    ax.set_yticklabels(class_names)  # è®¾ç½®çºµè½´æ ‡ç­¾
    ax.set_ylabel('True class')  # ä½¿ç”¨è‹±æ–‡çºµè½´æ ‡é¢˜
    ax.set_xlabel('Predicted class')  # ä½¿ç”¨è‹±æ–‡æ¨ªè½´æ ‡é¢˜
    plt.tight_layout()  # è°ƒæ•´å¸ƒå±€
    fig.savefig(save_path)  # ä¿å­˜å›¾ç‰‡
    plt.close(fig)  # å…³é—­ç”»å¸ƒ


def confusion_to_dict(matrix: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, int]]:  # å®šä¹‰æ··æ·†çŸ©é˜µåºåˆ—åŒ–å‡½æ•°
    result: Dict[str, Dict[str, int]] = {}  # åˆå§‹åŒ–ç»“æœå­—å…¸
    for i, real_name in enumerate(class_names):  # éå†çœŸå®ç±»åˆ«
        row_dict: Dict[str, int] = {}  # åˆå§‹åŒ–è¡Œå­—å…¸
        for j, pred_name in enumerate(class_names):  # éå†é¢„æµ‹ç±»åˆ«
            row_dict[pred_name] = int(matrix[i, j])  # å†™å…¥è®¡æ•°
        result[real_name] = row_dict  # ä¿å­˜ä¸€è¡Œç»“æœ
    return result  # è¿”å›åºåˆ—åŒ–ç»“æœ

def per_class_accuracy(matrix: np.ndarray, class_names: List[str]) -> Dict[str, float]:  # ¶¨Òå°´Àà±ğ¾«¶ÈÍ³¼Æº¯Êı
    totals = matrix.sum(axis=1)  # Í³¼ÆÃ¿ÀàÑù±¾×ÜÊı
    acc: Dict[str, float] = {}  # ³õÊ¼»¯½á¹û×Öµä
    for idx, name in enumerate(class_names):  # ±éÀúÀà±ğ
        total = float(totals[idx])  # ¶ÔÓ¦Àà±ğ×ÜÊı
        correct = float(matrix[idx, idx])  # ¶Ô½ÇÏßÃüÖĞ
        acc[name] = 0.0 if total == 0.0 else correct / total  # ¼ÆËã¾«¶È
    return acc  # ·µ»ØÃ¿Àà¾«¶È


def _sum_by_indices(matrix: np.ndarray, indices: Iterable[int]) -> Tuple[float, float]:  # ¶¨Òå¸¨ÖúÇóºÍº¯Êı
    idx_list = list(indices)  # ×ªÎªÁĞ±í±ÜÃâÖØ¸´µü´ú
    if not idx_list:  # ÈôË÷ÒıÎª¿Õ
        return 0.0, 0.0  # ·µ»ØÁã
    sub_matrix = matrix[idx_list, :]  # Ñ¡È¡×Ó¾ØÕó
    correct = float(sub_matrix[:, idx_list].diagonal().sum())  # Í³¼Æ¶Ô½ÇÏßÃüÖĞ
    total = float(sub_matrix.sum())  # Í³¼Æ×ÜÊı
    return correct, total  # ·µ»ØÃüÖĞÓë×ÜÊı


def group_accuracy(matrix: np.ndarray, class_names: List[str], group: Iterable[str]) -> float:  # ¶¨ÒåÈº×é¾«¶Èº¯Êı
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}  # ½¨Á¢Ãû³ÆË÷ÒıÓ³Éä
    indices = [name_to_idx[name] for name in group if name in name_to_idx]  # ¹ıÂË´æÔÚµÄÀà±ğ
    correct, total = _sum_by_indices(matrix, indices)  # Í³¼ÆÈº×éÃüÖĞ
    return 0.0 if total == 0.0 else correct / total  # ·µ»ØÈº×é¾«¶È


def summarize_incremental_metrics(
    matrix: np.ndarray,
    class_names: List[str],
    old_class_names: Iterable[str],
    baseline_old_acc: Optional[float] = None,
    baseline_new_acc: Optional[float] = None,
) -> Dict[str, float]:  # ¶¨ÒåÔöÁ¿Ö¸±ê»ã×Üº¯Êı
    overall_correct = float(matrix.trace())  # Í³¼Æ×ÜÃüÖĞ
    overall_total = float(matrix.sum())  # Í³¼Æ×ÜÑù±¾
    overall_acc = 0.0 if overall_total == 0.0 else overall_correct / overall_total  # ¼ÆËã×ÜÌå¾«¶È

    old_class_names = list(old_class_names)  # ¹Ì¶¨¾ÉÀàÁĞ±í
    new_class_names = [name for name in class_names if name not in old_class_names]  # ÅÉÉúĞÂÀàÁĞ±í

    old_acc = group_accuracy(matrix, class_names, old_class_names)  # ¼ÆËã¾ÉÀà¾«¶È
    new_acc = group_accuracy(matrix, class_names, new_class_names)  # ¼ÆËãĞÂÀà¾«¶È

    metrics: Dict[str, float] = {  # »ã×ÜÖ¸±ê
        'overall_acc': overall_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
    }

    if baseline_old_acc is not None:  # ÈôÌá¹©¾ÉÀà»ùÏß
        metrics['delta_old'] = old_acc - baseline_old_acc  # ¼ÇÂ¼¾ÉÀà¾«¶È±ä»¯
        if baseline_old_acc > 0:  # ¼ÆËãÒÅÍüÂÊ
            metrics['forget_rate'] = max(0.0, (baseline_old_acc - old_acc) / baseline_old_acc)
    if baseline_new_acc is not None:  # ÈôÌá¹©ĞÂÀà»ùÏß
        metrics['delta_new'] = new_acc - baseline_new_acc  # ¼ÇÂ¼ĞÂÀà¾«¶È±ä»¯

    return metrics  # ·µ»ØÖ¸±ê×Öµä
