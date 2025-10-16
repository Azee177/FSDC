# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

import argparse  # 导入参数解析模块
from pathlib import Path  # 导入路径模块
from typing import Dict  # 导入类型别名

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入numpy
import torch  # 导入torch
from sklearn.manifold import TSNE  # 导入t-SNE

try:  # 尝试导入UMAP
    import umap  # type: ignore
except ImportError:  # 若未安装
    umap = None  # 占位

from ..utils.common import ensure_dir, load_yaml_config  # 导入通用工具


def parse_args() -> argparse.Namespace:  # 定义命令行解析函数
    parser = argparse.ArgumentParser(description='特征可视化：t-SNE / UMAP')  # 创建解析器
    parser.add_argument('--paths', type=Path, required=True, help='指定路径配置文件')  # 添加路径参数
    parser.add_argument('--task_cfg', type=Path, required=True, help='指定任务配置文件')  # 添加任务参数
    parser.add_argument('--feature_cache', type=Path, required=True, help='指定特征缓存文件 (.pt)')  # 添加特征文件
    parser.add_argument('--max_samples', type=int, default=2000, help='限制采样数量以加速可视化')  # 添加样本上限
    parser.add_argument('--use_tsne', action='store_true', help='启用t-SNE可视化')  # 添加t-SNE开关
    parser.add_argument('--use_umap', action='store_true', help='启用UMAP可视化')  # 添加UMAP开关
    parser.add_argument('--tsne_perplexity', type=float, default=30.0, help='指定t-SNE困惑度')  # 添加困惑度
    parser.add_argument('--save_dir', type=Path, required=True, help='指定输出目录（默认outputs/viz）')  # 添加输出目录
    return parser.parse_args()  # 返回解析结果


def load_class_names(paths_cfg: Dict, task_cfg: Dict) -> Dict[int, str]:  # 定义类别名称加载函数
    class_names = task_cfg.get('class_names')  # 优先使用任务配置
    if class_names:  # 若存在
        return {idx: name for idx, name in enumerate(class_names)}  # 构建映射
    from ..data.coco_cls import read_class_map  # 延迟导入避免循环依赖

    class_map = read_class_map(Path(paths_cfg['class_map']))  # 读取映射
    sorted_items = sorted(class_map.items(), key=lambda kv: kv[1])  # 按索引排序
    return {idx: name for idx, (name, _) in enumerate(sorted_items)}  # 构建映射


def prepare_embedding_input(features: torch.Tensor, labels: torch.Tensor, max_samples: int) -> Dict[str, np.ndarray]:  # 定义数据准备函数
    total = features.size(0)  # 获取总数
    if total > max_samples:  # 若超出上限
        perm = torch.randperm(total)[:max_samples]  # 随机采样
        features = features[perm]  # 子集特征
        labels = labels[perm]  # 子集标签
    feats_np = features.numpy()  # 转换为numpy
    labels_np = labels.numpy()  # 转换为numpy
    return {'features': feats_np, 'labels': labels_np}  # 返回结果


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, class_map: Dict[int, str], save_path: Path, title: str) -> None:  # 定义绘图函数
    ensure_dir(save_path.parent)  # 确保输出目录存在
    plt.figure(figsize=(8, 8))  # 创建画布
    num_classes = len(class_map)  # 获取类别数
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))  # 生成配色
    for idx in range(num_classes):  # 遍历类别
        mask = labels == idx  # 构建掩码
        if np.any(mask):  # 若存在样本
            plt.scatter(embedding[mask, 0], embedding[mask, 1], s=8, color=colors[idx], label=class_map.get(idx, str(idx)), alpha=0.7)  # 绘制散点
    plt.legend(loc='best', fontsize='small', ncol=2)  # 添加图例
    plt.title(title)  # 设置标题
    plt.tight_layout()  # 调整布局
    plt.savefig(save_path)  # 保存图片
    plt.close()  # 关闭画布


def main() -> None:  # 定义主函数
    args = parse_args()  # 解析命令行参数
    paths_cfg = load_yaml_config(args.paths)  # 加载路径配置
    task_cfg = load_yaml_config(args.task_cfg)  # 加载任务配置
    ensure_dir(args.save_dir)  # 确保输出目录存在

    cache = torch.load(args.feature_cache, map_location='cpu')  # 读取特征缓存
    if isinstance(cache, dict) and 'features' in cache and 'labels' in cache:  # 判断缓存格式
        features = cache['features'].float()  # 获取特征
        labels = cache['labels'].long()  # 获取标签
    else:
        raise ValueError('feature_cache 必须包含 features 和 labels 字段')  # 抛出错误

    data = prepare_embedding_input(features, labels, args.max_samples)  # 准备数据
    class_map = load_class_names(paths_cfg, task_cfg)  # 加载类别映射

    if args.use_tsne:  # 若启用t-SNE
        tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, learning_rate='auto', init='pca', random_state=42)  # 创建t-SNE
        tsne_emb = tsne.fit_transform(data['features'])  # 计算嵌入
        plot_embedding(tsne_emb, data['labels'], class_map, args.save_dir / 'tsne.png', 't-SNE Embedding')  # 绘制t-SNE

    if args.use_umap:  # 若启用UMAP
        if umap is None:  # 判断是否安装
            raise RuntimeError('UMAP 未安装，请 `pip install umap-learn` 后再尝试')  # 抛出错误
        reducer = umap.UMAP(n_components=2, random_state=42)  # 创建UMAP
        umap_emb = reducer.fit_transform(data['features'])  # 计算嵌入
        plot_embedding(umap_emb, data['labels'], class_map, args.save_dir / 'umap.png', 'UMAP Embedding')  # 绘制UMAP


if __name__ == '__main__':  # 判断是否主入口
    main()  # 执行主函数
