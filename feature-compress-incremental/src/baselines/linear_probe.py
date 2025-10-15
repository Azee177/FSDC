# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

import argparse  # 导入参数解析模块
from pathlib import Path  # 导入路径类
from typing import Dict  # 导入类型别名

import torch  # 导入torch库
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载器

from ..data.coco_cls import build_dataset, build_fake_dataset, read_class_map  # 导入数据集函数
from ..data.transforms import build_transform  # 导入变换构建函数
from ..models.clip_wrapper import FrozenCLIPEncoder  # 导入CLIP封装
from ..models.classifier import LinearClassifier  # 导入线性分类器
from ..utils.common import ensure_dir, load_yaml_config, set_seed, get_device, save_json, log_experiment_info, count_parameters  # 导入通用工具
from ..utils.metrics import topk_accuracy, build_confusion, plot_confusion, confusion_to_dict  # 导入指标工具


def parse_args() -> argparse.Namespace:  # 定义命令行解析函数
    parser = argparse.ArgumentParser(description='线性探针基线')  # 创建解析器
    parser.add_argument('--paths', type=Path, required=True, help='指定路径配置文件')  # 添加路径参数
    parser.add_argument('--task_cfg', type=Path, required=True, help='指定任务配置文件')  # 添加任务参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16', help='指定CLIP骨干名称')  # 添加骨干参数
    parser.add_argument('--pretrained', type=str, default='openai', help='指定预训练权重名称')  # 添加预训练参数
    parser.add_argument('--epochs', type=int, default=5, help='指定训练轮数')  # 添加训练轮数
    parser.add_argument('--batch_size', type=int, default=128, help='指定批次大小')  # 添加批次大小
    parser.add_argument('--lr', type=float, default=1e-3, help='指定学习率')  # 添加学习率
    parser.add_argument('--weight_decay', type=float, default=0.0, help='指定权重衰减')  # 添加权重衰减
    parser.add_argument('--num_workers', type=int, default=4, help='指定数据加载线程数')  # 添加线程参数
    parser.add_argument('--use_cache', action='store_true', help='是否复用特征缓存')  # 添加缓存开关
    parser.add_argument('--save_dir', type=Path, required=True, help='指定输出目录')  # 添加输出目录
    parser.add_argument('--seed', type=int, default=7, help='指定随机种子')  # 添加随机种子
    parser.add_argument('--dummy', action='store_true', help='是否使用假数据模式')  # 添加假数据开关
    return parser.parse_args()  # 返回解析结果


def load_configs(args: argparse.Namespace) -> Dict:  # 定义配置合并函数
    paths_cfg = load_yaml_config(args.paths)  # 加载路径配置
    task_cfg = load_yaml_config(args.task_cfg)  # 加载任务配置
    merged = {**paths_cfg, **task_cfg}  # 合并字典
    merged['epochs'] = args.epochs  # 写入训练轮数
    merged['batch_size'] = args.batch_size  # 写入批次大小
    merged['lr'] = args.lr  # 写入学习率
    merged['weight_decay'] = args.weight_decay  # 写入权重衰减
    merged['backbone'] = args.backbone  # 写入骨干名称
    merged['pretrained'] = args.pretrained  # 写入预训练标识
    merged['use_cache'] = args.use_cache  # 写入缓存标记
    merged['image_size'] = task_cfg.get('image_size', 224)  # 写入图像尺寸
    merged['class_names'] = task_cfg.get('class_names', [])  # 写入类别名称
    return merged  # 返回合并配置


def prepare_dataloaders(cfg: Dict, args: argparse.Namespace):  # 定义数据加载器准备函数
    if args.dummy:  # 判断是否假数据模式
        num_classes = int(cfg.get('num_classes', 10))  # 读取类别数量
        fake_train = build_fake_dataset(num_classes, cfg['image_size'], 512, args.seed)  # 构建训练假数据
        fake_val = build_fake_dataset(num_classes, cfg['image_size'], 128, args.seed + 1)  # 构建验证假数据
        train_loader = DataLoader(fake_train, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)  # 创建训练加载器
        val_loader = DataLoader(fake_val, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)  # 创建验证加载器
        class_map = {str(i): i for i in range(num_classes)}  # 构建类别映射
        cfg['class_names'] = [f'cls_{i}' for i in range(num_classes)]  # 更新类别名称
        return train_loader, val_loader, class_map  # 返回加载器与映射
    transform_train = build_transform(cfg['image_size'], True)  # 构建训练变换
    transform_val = build_transform(cfg['image_size'], False)  # 构建验证变换
    train_dataset = build_dataset(Path(cfg['train_list']), transform_train)  # 构建训练数据集
    val_dataset = build_dataset(Path(cfg['val_list']), transform_val)  # 构建验证数据集
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True)  # 创建训练加载器
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True)  # 创建验证加载器
    class_map = read_class_map(Path(cfg['class_map']))  # 读取类别映射
    if not cfg.get('class_names'):  # 判断是否缺失名称
        sorted_items = sorted(class_map.items(), key=lambda kv: kv[1])  # 按索引排序
        cfg['class_names'] = [item[0] for item in sorted_items]  # 填写名称
    return train_loader, val_loader, class_map  # 返回加载器与映射


def main() -> None:  # 定义主函数
    args = parse_args()  # 解析命令行参数
    cfg = load_configs(args)  # 加载配置
    set_seed(args.seed)  # 设置随机种子
    ensure_dir(args.save_dir)  # 确保输出目录存在
    device = get_device()  # 获取设备
    log_experiment_info(cfg)  # 打印配置
    train_loader_raw, val_loader_raw, class_map = prepare_dataloaders(cfg, args)  # 准备数据加载器
    num_classes = len(cfg['class_names'])  # 使用配置中类别数量
    encoder = FrozenCLIPEncoder(args.backbone, args.pretrained, device, Path(cfg.get('cache_root', args.save_dir / 'cache')))  # 创建编码器
    train_cache = encoder.extract_and_cache(train_loader_raw, 'train', cfg['image_size'], False, args.use_cache)  # 提取训练特征
    val_cache = encoder.extract_and_cache(val_loader_raw, 'val', cfg['image_size'], False, args.use_cache)  # 提取验证特征
    train_dataset = TensorDataset(train_cache['features'].float(), train_cache['labels'].long())  # 构建训练特征数据集
    val_dataset = TensorDataset(val_cache['features'].float(), val_cache['labels'].long())  # 构建验证特征数据集
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)  # 创建特征训练加载器
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)  # 创建特征验证加载器
    classifier = LinearClassifier(encoder.output_dim(False), num_classes)  # 创建线性分类器
    classifier.to(device)  # 移动到设备
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])  # 创建优化器
    criterion = torch.nn.CrossEntropyLoss()  # 创建损失函数
    best_top1 = 0.0  # 初始化最佳精度
    history = []  # 初始化历史记录
    for epoch in range(1, cfg['epochs'] + 1):  # 遍历训练轮次
        classifier.train()  # 切换到训练模式
        for feats, labels in train_loader:  # 遍历训练批次
            feats = feats.to(device)  # 移动特征
            labels = labels.to(device)  # 移动标签
            optimizer.zero_grad(set_to_none=True)  # 清空梯度
            logits = classifier(feats)  # 计算预测
            loss = criterion(logits, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        classifier.eval()  # 切换到评估模式
        logits_list = []  # 初始化输出列表
        labels_list = []  # 初始化标签列表
        with torch.no_grad():  # 关闭梯度
            for feats, labels in val_loader:  # 遍历验证批次
                feats = feats.to(device)  # 移动特征
                logits = classifier(feats)  # 计算预测
                logits_list.append(logits.cpu())  # 保存输出
                labels_list.append(labels)  # 保存标签
        logits_tensor = torch.cat(logits_list, dim=0)  # 拼接输出
        labels_tensor = torch.cat(labels_list, dim=0)  # 拼接标签
        acc = topk_accuracy(logits_tensor, labels_tensor, ks=(1, 5))  # 计算精度
        val_loss = criterion(logits_tensor, labels_tensor).item()  # 计算验证损失
        record = {'epoch': epoch, 'val_loss': val_loss, 'top1': acc[1], 'top5': acc[5]}  # 构建记录
        history.append(record)  # 添加记录
        print(f'Epoch {epoch}: val_loss={val_loss:.4f} top1={acc[1]:.4f} top5={acc[5]:.4f}')  # 打印日志
        if acc[1] > best_top1:  # 判断是否刷新最佳
            best_top1 = acc[1]  # 更新最佳Top1
            torch.save({'classifier': classifier.state_dict(), 'config': cfg}, args.save_dir / 'best.ckpt')  # 保存最佳权重
    torch.save({'classifier': classifier.state_dict(), 'config': cfg}, args.save_dir / 'last.ckpt')  # 保存最终权重
    save_json({'history': history, 'best_top1': best_top1, 'params': count_parameters(classifier)}, args.save_dir / 'metrics.json')  # 保存指标
    class_names = cfg.get('class_names', [str(i) for i in range(num_classes)])  # 获取类别名称
    confusion = build_confusion(logits_tensor, labels_tensor, num_classes)  # 构建混淆矩阵
    plot_confusion(confusion, class_names, args.save_dir / 'confusion_linear.png')  # 绘制混淆矩阵
    save_json(confusion_to_dict(confusion, class_names), args.save_dir / 'confusion_linear.json')  # 保存混淆矩阵


if __name__ == '__main__':  # 判断是否主入口
    main()  # 执行主函数
