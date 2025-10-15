# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

from pathlib import Path  # 导入路径类
from typing import Dict, List, Tuple  # 导入类型别名

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入numpy库
import torch  # 导入torch库
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数

from .common import ensure_dir  # 导入目录工具


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: Tuple[int, ...] = (1, 5)) -> Dict[int, float]:  # 定义TopK精度函数
    max_k = max(ks)  # 计算最大K值
    _, pred = torch.topk(logits, max_k, dim=1)  # 获取前K预测
    pred = pred.t()  # 转置以方便比较
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # 构建命中矩阵
    results: Dict[int, float] = {}  # 初始化结果字典
    total = targets.size(0)  # 获取样本数量
    for k in ks:  # 遍历各个K
        correct_k = correct[:k].reshape(-1).float().sum(0).item()  # 统计命中数量
        results[k] = float(correct_k / max(1, total))  # 计算TopK精度
    return results  # 返回精度字典


def build_confusion(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:  # 定义混淆矩阵构建函数
    preds = logits.argmax(dim=1).cpu().numpy()  # 获取预测类别
    labels = targets.cpu().numpy()  # 获取真实类别
    matrix = confusion_matrix(labels, preds, labels=list(range(num_classes)))  # 计算混淆矩阵
    return matrix  # 返回混淆矩阵


def plot_confusion(matrix: np.ndarray, class_names: List[str], save_path: Path) -> None:  # 定义混淆矩阵绘制函数
    ensure_dir(save_path.parent)  # 确保输出目录存在
    fig, ax = plt.subplots(figsize=(8, 8))  # 创建画布
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')  # 绘制热力图
    ax.figure.colorbar(im, ax=ax)  # 添加色条
    ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]))  # 设置坐标刻度
    ax.set_xticklabels(class_names, rotation=45, ha='right')  # 设置横轴标签
    ax.set_yticklabels(class_names)  # 设置纵轴标签
    ax.set_ylabel('True class')  # 使用英文纵轴标题
    ax.set_xlabel('Predicted class')  # 使用英文横轴标题
    plt.tight_layout()  # 调整布局
    fig.savefig(save_path)  # 保存图片
    plt.close(fig)  # 关闭画布


def confusion_to_dict(matrix: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, int]]:  # 定义混淆矩阵序列化函数
    result: Dict[str, Dict[str, int]] = {}  # 初始化结果字典
    for i, real_name in enumerate(class_names):  # 遍历真实类别
        row_dict: Dict[str, int] = {}  # 初始化行字典
        for j, pred_name in enumerate(class_names):  # 遍历预测类别
            row_dict[pred_name] = int(matrix[i, j])  # 写入计数
        result[real_name] = row_dict  # 保存一行结果
    return result  # 返回序列化结果
