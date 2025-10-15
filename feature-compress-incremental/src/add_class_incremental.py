# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

import argparse  # 导入参数解析模块
from pathlib import Path  # 导入路径类
from typing import Dict, List  # 导入类型别名

import torch  # 导入torch库
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载器

from .data.coco_cls import build_dataset, read_class_map  # 导入数据集函数
from .data.transforms import build_transform  # 导入变换构建函数
from .models.clip_wrapper import FrozenCLIPEncoder  # 导入CLIP封装类
from .models.compress_head import CompressionHead  # 导入压缩模块
from .models.classifier import LinearClassifier  # 导入分类头
from .utils.common import ensure_dir, load_yaml_config, get_device, set_seed, save_json, log_experiment_info  # 导入通用工具
from .utils.metrics import topk_accuracy, build_confusion, plot_confusion, confusion_to_dict  # 导入指标工具


def parse_args() -> argparse.Namespace:  # 定义命令行解析函数
    parser = argparse.ArgumentParser(description='增量添加新类实验')  # 创建解析器
    parser.add_argument('--paths', type=Path, required=True, help='指定路径配置文件')  # 添加路径参数
    parser.add_argument('--task_cfg', type=Path, required=True, help='指定增量任务配置文件')  # 添加任务参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16', help='指定CLIP骨干名称')  # 添加骨干参数
    parser.add_argument('--pretrained', type=str, default='openai', help='指定预训练权重名称')  # 添加预训练参数
    parser.add_argument('--resume_ckpt', type=Path, required=True, help='指定初始训练权重路径')  # 添加权重参数
    parser.add_argument('--n_compress', type=int, default=32, help='指定压缩输出维度')  # 添加压缩维度
    parser.add_argument('--epochs', type=int, default=1, help='指定训练轮数')  # 添加训练轮数
    parser.add_argument('--batch_size', type=int, default=128, help='指定批次大小')  # 添加批次大小
    parser.add_argument('--lr', type=float, default=1e-3, help='指定学习率')  # 添加学习率
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='指定权重衰减')  # 添加权重衰减
    parser.add_argument('--num_workers', type=int, default=4, help='指定数据加载线程数')  # 添加线程数
    parser.add_argument('--use_cache', action='store_true', help='是否复用特征缓存')  # 添加缓存开关
    parser.add_argument('--save_dir', type=Path, required=True, help='指定输出目录')  # 添加输出目录
    parser.add_argument('--seed', type=int, default=123, help='指定随机种子')  # 添加随机种子
    parser.add_argument('--add_channels', type=int, default=0, help='指定新增压缩通道数')  # 添加通道参数
    parser.add_argument('--replay_per_class', type=int, default=5, help='指定每类回放样本数')  # 添加回放参数
    parser.add_argument('--use_patch', action='store_true', help='是否使用补丁特征输入')  # 添加补丁开关
    parser.add_argument('--dummy', action='store_true', help='是否使用假数据模式')  # 添加假数据开关
    return parser.parse_args()  # 返回解析结果


def load_configs(args: argparse.Namespace) -> Dict:  # 定义配置合并函数
    paths_cfg = load_yaml_config(args.paths)  # 加载路径配置
    task_cfg = load_yaml_config(args.task_cfg)  # 加载任务配置
    merged = {**paths_cfg, **task_cfg}  # 合并字典
    merged['backbone'] = args.backbone  # 写入骨干名称
    merged['pretrained'] = args.pretrained  # 写入预训练标识
    merged['n_compress'] = args.n_compress  # 写入压缩维度
    merged['epochs'] = args.epochs  # 写入训练轮数
    merged['batch_size'] = args.batch_size  # 写入批次大小
    merged['lr'] = args.lr  # 写入学习率
    merged['weight_decay'] = args.weight_decay  # 写入权重衰减
    merged['use_patch'] = args.use_patch  # 写入补丁标记
    merged['use_cache'] = args.use_cache  # 写入缓存标记
    merged['task_name'] = task_cfg.get('task_name', 'add_class')  # 写入任务名称
    return merged  # 返回合并配置


def sample_replay(features: torch.Tensor, labels: torch.Tensor, per_class: int, old_class_count: int) -> torch.Tensor:  # 定义回放索引采样函数
    indices: List[torch.Tensor] = []  # 初始化索引列表
    for class_id in range(old_class_count):  # 遍历旧类别
        mask = (labels == class_id).nonzero(as_tuple=False).view(-1)  # 找出对应索引
        if mask.numel() == 0:  # 判断该类是否存在样本
            continue  # 无样本则跳过
        choice = mask[torch.randperm(mask.numel())[:per_class]]  # 随机采样指定数量
        indices.append(choice)  # 追加到列表
    if not indices:  # 判断是否采样到数据
        return torch.empty(0, dtype=torch.long)  # 返回空索引
    return torch.cat(indices, dim=0)  # 拼接所有索引


def main() -> None:  # 定义主入口函数
    args = parse_args()  # 解析命令行参数
    cfg = load_configs(args)  # 加载配置
    set_seed(args.seed)  # 设置随机种子
    ensure_dir(args.save_dir)  # 确保输出目录存在
    device = get_device()  # 获取设备
    log_experiment_info(cfg)  # 打印配置
    if args.dummy:  # 判断是否假数据模式
        print('假数据模式暂未实现，直接退出。')  # 打印提示
        return  # 结束程序
    transform_train = build_transform(cfg.get('image_size', 224), True)  # 构建训练变换
    transform_val = build_transform(cfg.get('image_size', 224), False)  # 构建验证变换
    base_train_dataset = build_dataset(Path(cfg['train_list']), transform_train)  # 构建旧类训练集
    base_val_dataset = build_dataset(Path(cfg['val_list']), transform_val)  # 构建旧类验证集
    new_train_dataset = build_dataset(Path(cfg['new_train_list']), transform_train)  # 构建新类训练集
    new_val_dataset = build_dataset(Path(cfg['new_val_list']), transform_val)  # 构建新类验证集
    train_loader_base = DataLoader(base_train_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True)  # 创建旧类训练加载器
    val_loader_base = DataLoader(base_val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True)  # 创建旧类验证加载器
    train_loader_new = DataLoader(new_train_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True)  # 创建新类训练加载器
    val_loader_new = DataLoader(new_val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True)  # 创建新类验证加载器
    class_map = read_class_map(Path(cfg['class_map']))  # 读取类别映射
    sorted_items = sorted(class_map.items(), key=lambda kv: kv[1])  # 按索引排序映射
    new_class_names = list(cfg.get('new_class_names', ['new_cls']))  # 读取增量类别名称
    new_class_ids = cfg.get('new_class_ids', [])  # 读取增量类别ID
    if 'class_names' in cfg:
        old_class_names = list(cfg['class_names'])  # 使用配置中的旧类别名称
        old_class_count = len(old_class_names)  # 统计旧类别数量
    else:
        estimated_old = len(class_map) - (len(new_class_ids) if new_class_ids else len(new_class_names))
        old_class_count = max(0, estimated_old)  # 根据映射减去新增类别推断旧类别数量
        old_class_names = [name for name, idx in sorted_items if idx < old_class_count]  # 退化为映射中的名称
    if len(old_class_names) < old_class_count:
        old_class_names.extend([f'class_{i}' for i in range(len(old_class_names), old_class_count)])
    else:
        old_class_names = old_class_names[:old_class_count]
    expected_new = len(new_class_ids) if new_class_ids else len(new_class_names)
    if len(new_class_names) < expected_new:
        new_class_names.extend([f'new_cls_{i}' for i in range(len(new_class_names), expected_new)])
    else:
        new_class_names = new_class_names[:expected_new]
    class_names = old_class_names + new_class_names  # 合并全部类别名称
    total_classes = len(class_names)  # 计算类别总数
    cache_root = Path(cfg.get('cache_root', args.save_dir / 'cache'))  # 获取缓存目录
    encoder = FrozenCLIPEncoder(args.backbone, args.pretrained, device, cache_root)  # 创建编码器
    return_patch = cfg.get('use_patch', False)  # 获取补丁标记
    base_cache = encoder.extract_and_cache(train_loader_base, 'train', cfg.get('image_size', 224), return_patch, args.use_cache)  # 提取旧类训练特征
    new_cache = encoder.extract_and_cache(train_loader_new, 'train_new', cfg.get('image_size', 224), return_patch, False)  # 提取新类训练特征
    val_base_cache = encoder.extract_and_cache(val_loader_base, 'val', cfg.get('image_size', 224), return_patch, args.use_cache)  # 提取旧类验证特征
    val_new_cache = encoder.extract_and_cache(val_loader_new, 'val_new', cfg.get('image_size', 224), return_patch, False)  # 提取新类验证特征
    replay_indices = sample_replay(base_cache['features'], base_cache['labels'], args.replay_per_class, old_class_count)  # 采样回放索引
    replay_features = base_cache['features'][replay_indices] if replay_indices.numel() > 0 else torch.empty(0)  # 获取回放特征
    replay_labels = base_cache['labels'][replay_indices] if replay_indices.numel() > 0 else torch.empty(0, dtype=torch.long)  # 获取回放标签
    new_labels = torch.full((new_cache['features'].shape[0],), old_class_count, dtype=torch.long)  # 构建新类标签
    if replay_features.numel() > 0:  # 判断是否存在回放
        combined_features = torch.cat([new_cache['features'], replay_features], dim=0)  # 合并特征
        combined_labels = torch.cat([new_labels, replay_labels], dim=0)  # 合并标签
    else:  # 否则仅使用新类
        combined_features = new_cache['features']  # 直接使用新类特征
        combined_labels = new_labels  # 直接使用新类标签
    train_dataset = TensorDataset(combined_features.float(), combined_labels.long())  # 构建增量训练数据集
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)  # 创建增量训练加载器
    val_features = torch.cat([val_base_cache['features'], val_new_cache['features']], dim=0)  # 合并验证特征
    val_labels = torch.cat([val_base_cache['labels'], torch.full((val_new_cache['features'].shape[0],), old_class_count, dtype=torch.long)], dim=0)  # 合并验证标签
    val_dataset = TensorDataset(val_features.float(), val_labels.long())  # 构建验证数据集
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)  # 创建验证加载器
    checkpoint = torch.load(args.resume_ckpt, map_location='cpu')  # 加载初始权重
    input_dim = encoder.output_dim(return_patch)  # 获取输入维度
    compressor = CompressionHead(input_dim, cfg['n_compress'], return_patch)  # 创建压缩模块
    classifier = LinearClassifier(compressor.output_dim(), old_class_count)  # 创建分类头
    compressor.load_state_dict(checkpoint['compressor'])  # 恢复压缩模块参数
    classifier.load_state_dict(checkpoint['classifier'])  # 恢复分类头参数
    compressor.add_channels(args.add_channels)  # 按需扩展通道
    classifier.expand_classes(total_classes - old_class_count)  # 扩展类别维度
    compressor.to(device)  # 移动压缩模块
    classifier.to(device)  # 移动分类头
    optimizer = torch.optim.AdamW(list(compressor.parameters()) + list(classifier.parameters()), lr=cfg['lr'], weight_decay=cfg['weight_decay'])  # 创建优化器
    criterion = torch.nn.CrossEntropyLoss()  # 创建损失函数
    best_top1 = 0.0  # 初始化最佳Top1
    for epoch in range(1, cfg['epochs'] + 1):  # 遍历训练轮次
        compressor.train()  # 切换压缩模块到训练模式
        classifier.train()  # 切换分类头到训练模式
        for feats, labels in train_loader:  # 遍历训练批次
            feats = feats.to(device)  # 移动特征
            labels = labels.to(device)  # 移动标签
            optimizer.zero_grad(set_to_none=True)  # 清空梯度
            logits = classifier(compressor(feats))  # 前向传播
            loss = criterion(logits, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        compressor.eval()  # 切换压缩模块到评估模式
        classifier.eval()  # 切换分类头到评估模式
        logits_list = []  # 初始化输出列表
        labels_list = []  # 初始化标签列表
        with torch.no_grad():  # 关闭梯度
            for feats, labels in val_loader:  # 遍历验证批次
                feats = feats.to(device)  # 移动特征
                logits = classifier(compressor(feats))  # 计算预测
                logits_list.append(logits.cpu())  # 保存输出
                labels_list.append(labels)  # 保存标签
        logits_tensor = torch.cat(logits_list, dim=0)  # 拼接输出
        labels_tensor = torch.cat(labels_list, dim=0)  # 拼接标签
        acc = topk_accuracy(logits_tensor, labels_tensor, ks=(1, 5))  # 计算精度
        print(f'Epoch {epoch}: top1={acc[1]:.4f} top5={acc[5]:.4f}')  # 打印日志
        if acc[1] > best_top1:  # 判断是否刷新最佳
            best_top1 = acc[1]  # 更新最佳Top1
            torch.save({
                'compressor': compressor.state_dict(),  # 保存压缩模块
                'classifier': classifier.state_dict(),  # 保存分类头
                'config': cfg,  # 保存配置
            }, args.save_dir / 'best_incremental.ckpt')  # 保存最佳权重
    torch.save({
        'compressor': compressor.state_dict(),  # 保存压缩模块
        'classifier': classifier.state_dict(),  # 保存分类头
        'config': cfg,  # 保存配置
    }, args.save_dir / 'last_incremental.ckpt')  # 保存最终权重
    confusion = build_confusion(logits_tensor, labels_tensor, total_classes)  # 计算混淆矩阵
    plot_confusion(confusion, class_names, args.save_dir / 'confusion_incremental.png')  # 绘制混淆矩阵
    save_json(confusion_to_dict(confusion, class_names), args.save_dir / 'confusion_incremental.json')  # 保存混淆矩阵
    old_mask = labels_tensor < old_class_count  # 构建旧类掩码
    new_mask = labels_tensor >= old_class_count  # 构建新类掩码
    old_acc = float((logits_tensor[old_mask].argmax(dim=1) == labels_tensor[old_mask]).float().mean().item()) if old_mask.any() else 0.0  # 计算旧类精度
    new_acc = float((logits_tensor[new_mask].argmax(dim=1) == labels_tensor[new_mask]).float().mean().item()) if new_mask.any() else 0.0  # 计算新类精度
    save_json({
        'best_top1': best_top1,  # 保存最佳Top1
        'old_acc': old_acc,  # 保存旧类精度
        'new_acc': new_acc,  # 保存新类精度
        'replay_per_class': args.replay_per_class,  # 保存回放数量
    }, args.save_dir / 'metrics_incremental.json')  # 写入指标文件


if __name__ == '__main__':  # 判断是否主入口
    main()  # 执行主函数
