# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

import argparse  # 导入参数解析模块
from pathlib import Path  # 导入路径类
from typing import Dict  # 导入类型别名

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入numpy库
import torch  # 导入torch库
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载器

from ..data.coco_cls import build_dataset, read_class_map  # 导入数据集函数
from ..data.transforms import build_transform  # 导入变换构建函数
from ..models.clip_wrapper import FrozenCLIPEncoder  # 导入CLIP封装
from ..models.compress_head import CompressionHead  # 导入压缩模块
from ..utils.common import load_yaml_config, ensure_dir, save_json, get_device  # 导入通用工具


def parse_args() -> argparse.Namespace:  # 定义参数解析函数
    parser = argparse.ArgumentParser(description='压缩通道响应可视化')  # 创建解析器
    parser.add_argument('--paths', type=Path, required=True, help='指定路径配置文件')  # 添加路径参数
    parser.add_argument('--task_cfg', type=Path, required=True, help='指定任务配置文件')  # 添加任务参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16', help='指定CLIP骨干名称')  # 添加骨干参数
    parser.add_argument('--pretrained', type=str, default='openai', help='指定预训练权重名称')  # 添加预训练参数
    parser.add_argument('--checkpoint', type=Path, required=True, help='指定压缩模型权重路径')  # 添加权重参数
    parser.add_argument('--batch_size', type=int, default=256, help='指定批次大小')  # 添加批次参数
    parser.add_argument('--num_workers', type=int, default=4, help='指定数据加载线程数')  # 添加线程数
    parser.add_argument('--use_patch', action='store_true', help='是否使用补丁特征输入')  # 添加补丁开关
    parser.add_argument('--save_dir', type=Path, required=True, help='指定输出目录')  # 添加输出目录
    return parser.parse_args()  # 返回解析结果


def main() -> None:  # 定义主函数
    args = parse_args()  # 解析命令行参数
    paths_cfg = load_yaml_config(args.paths)  # 加载路径配置
    task_cfg = load_yaml_config(args.task_cfg)  # 加载任务配置
    ensure_dir(args.save_dir)  # 确保输出目录存在
    device = get_device()  # 获取设备
    transform = build_transform(task_cfg.get('image_size', 224), False)  # 构建验证变换
    dataset = build_dataset(Path(paths_cfg['val_list']), transform)  # 构建验证数据集
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)  # 创建数据加载器
    class_map = read_class_map(Path(paths_cfg['class_map']))  # 读取类别映射
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]  # 构建类别名称
    encoder = FrozenCLIPEncoder(args.backbone, args.pretrained, device, Path(paths_cfg.get('cache_root', args.save_dir / 'cache')))  # 创建编码器
    cache = encoder.extract_and_cache(loader, 'viz_val', task_cfg.get('image_size', 224), args.use_patch, True)  # 获取缓存特征
    features = cache['features'].float()  # 获取特征张量
    labels = cache['labels'].long()  # 获取标签张量
    dataset_feat = TensorDataset(features, labels)  # 构建特征数据集
    loader_feat = DataLoader(dataset_feat, batch_size=args.batch_size, shuffle=False)  # 创建特征加载器
    checkpoint = torch.load(args.checkpoint, map_location='cpu')  # 加载权重
    input_dim = encoder.output_dim(args.use_patch)  # 获取输入维度
    out_dim = checkpoint['compressor']['mapper.weight'].shape[0]  # 读取输出通道
    compressor = CompressionHead(input_dim, out_dim, args.use_patch)  # 创建压缩模块
    compressor.load_state_dict(checkpoint['compressor'])  # 恢复参数
    compressor.to(device)  # 移动到设备
    compressor.eval()  # 切换到评估模式
    num_channels = compressor.output_dim()  # 获取通道数
    num_classes = len(class_names)  # 获取类别数
    activations = torch.zeros((num_channels, num_classes), dtype=torch.float32)  # 初始化激活矩阵
    counts = torch.zeros(num_classes, dtype=torch.float32)  # 初始化计数器
    with torch.no_grad():  # 关闭梯度
        for feats, target in loader_feat:  # 遍历批次
            feats = feats.to(device)  # 移动特征
            target = target.to(device)  # 移动标签
            compressed = compressor(feats)  # 计算压缩输出
            for class_idx in range(num_classes):  # 遍历类别
                mask = target == class_idx  # 构建掩码
                if mask.any():  # 判断是否存在样本
                    activations[:, class_idx] += compressed[mask].mean(dim=0).cpu()  # 累加平均激活
                    counts[class_idx] += 1.0  # 记录出现次数
    for class_idx in range(num_classes):  # 遍历类别
        if counts[class_idx] > 0:  # 判断是否有样本
            activations[:, class_idx] /= counts[class_idx]  # 计算平均激活
    heatmap = activations.numpy()  # 转换为numpy数组
    plt.figure(figsize=(10, 6))  # 创建画布
    plt.imshow(heatmap, aspect='auto', cmap='viridis')  # 绘制热力图
    plt.colorbar()  # 添加色条
    plt.xlabel('类别')  # 设置横轴标签
    plt.ylabel('通道')  # 设置纵轴标签
    plt.xticks(ticks=np.arange(num_classes), labels=class_names, rotation=45, ha='right')  # 设置类别标签
    plt.tight_layout()  # 调整布局
    plt.savefig(args.save_dir / 'channel_class_heatmap.png')  # 保存热力图
    plt.close()  # 关闭画布
    top_channels: Dict[str, Dict[str, float]] = {}  # 初始化输出字典
    for class_idx, class_name in enumerate(class_names):  # 遍历类别
        class_values = heatmap[:, class_idx]  # 获取该列数值
        order = np.argsort(-class_values)  # 计算降序索引
        top_indices = order[:5]  # 取前五通道
        top_channels[class_name] = {f'channel_{int(i)}': float(class_values[i]) for i in top_indices}  # 保存通道强度
    save_json(top_channels, args.save_dir / 'per_class_top_channels.json')  # 保存通道统计


if __name__ == '__main__':  # 判断是否主入口
    main()  # 执行主函数
