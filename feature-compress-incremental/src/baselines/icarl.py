# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

import argparse  # 导入参数解析模块
from pathlib import Path  # 导入路径模块
from typing import Dict  # 导入类型别名

import torch  # 导入torch
import torch.nn.functional as F  # 导入函数式API
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载模块

from ..data.coco_cls import build_dataset, read_class_map  # 导入数据集工具
from ..data.transforms import build_transform  # 导入变换构建函数
from ..models.clip_wrapper import FrozenCLIPEncoder  # 导入CLIP封装
from ..models.classifier import LinearClassifier  # 导入线性分类器
from ..utils.common import ensure_dir, load_yaml_config, set_seed, get_device, save_json, log_experiment_info, setup_logging  # 导入通用工具
from loguru import logger  # 导入日志工具
from ..utils.metrics import (  # 导入指标工具
    topk_accuracy,
    build_confusion,
    plot_confusion,
    confusion_to_dict,
    per_class_accuracy,
    summarize_incremental_metrics,
)  # 导入指标工具


def parse_args() -> argparse.Namespace:  # 定义命令行解析函数
    parser = argparse.ArgumentParser(description='增量基线：iCaRL')  # 创建解析器
    parser.add_argument('--paths', type=Path, required=True, help='指定路径配置文件')  # 添加路径参数
    parser.add_argument('--task_cfg', type=Path, required=True, help='指定任务配置文件')  # 添加任务参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16', help='指定CLIP骨干名称')  # 添加骨干参数
    parser.add_argument('--pretrained', type=str, default='openai', help='指定预训练权重标识或路径')  # 添加预训练参数
    parser.add_argument('--resume_ckpt', type=Path, required=True, help='指定旧类模型权重路径')  # 添加权重参数
    parser.add_argument('--epochs', type=int, default=1, help='指定训练轮数')  # 添加训练轮数
    parser.add_argument('--batch_size', type=int, default=128, help='指定批次大小')  # 添加批次大小
    parser.add_argument('--lr', type=float, default=1e-3, help='指定学习率')  # 添加学习率
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='指定权重衰减')  # 添加权重衰减
    parser.add_argument('--num_workers', type=int, default=4, help='指定数据加载线程数')  # 添加线程参数
    parser.add_argument('--use_cache', action='store_true', help='是否复用特征缓存')  # 添加缓存开关
    parser.add_argument('--save_dir', type=Path, required=True, help='指定输出目录')  # 添加输出目录
    parser.add_argument('--seed', type=int, default=99, help='指定随机种子')  # 添加随机种子
    parser.add_argument('--memory_per_class', type=int, default=20, help='指定每类记忆样本数量')  # 添加记忆参数
    parser.add_argument('--temperature', type=float, default=2.0, help='指定蒸馏温度T')  # 添加蒸馏温度
    parser.add_argument('--alpha', type=float, default=0.5, help='指定蒸馏损失权重')  # 添加蒸馏权重
    return parser.parse_args()  # 返回解析结果


def load_configs(args: argparse.Namespace) -> Dict:  # 定义配置合并函数
    paths_cfg = load_yaml_config(args.paths)  # 加载路径配置
    task_cfg = load_yaml_config(args.task_cfg)  # 加载任务配置
    merged = {**paths_cfg, **task_cfg}  # 合并字典
    merged['backbone'] = args.backbone  # 写入骨干名称
    merged['pretrained'] = args.pretrained  # 写入预训练标识
    merged['epochs'] = args.epochs  # 写入训练轮数
    merged['batch_size'] = args.batch_size  # 写入批次大小
    merged['lr'] = args.lr  # 写入学习率
    merged['weight_decay'] = args.weight_decay  # 写入权重衰减
    merged['memory_per_class'] = args.memory_per_class  # 写入记忆大小
    merged['temperature'] = args.temperature  # 写入温度
    merged['alpha'] = args.alpha  # 写入蒸馏权重
    merged['use_cache'] = args.use_cache  # 写入缓存标记
    return merged  # 返回合并配置


def select_exemplars(features: torch.Tensor, labels: torch.Tensor, per_class: int, num_classes: int) -> Dict[int, torch.Tensor]:  # 定义记忆选择函数
    exemplars: Dict[int, torch.Tensor] = {}  # 初始化记忆字典
    for cid in range(num_classes):  # 遍历旧类
        mask = (labels == cid).nonzero(as_tuple=False).view(-1)  # 找到该类索引
        if mask.numel() == 0:  # 若无样本
            continue  # 跳过
        count = min(per_class, mask.numel())  # 计算采样数量
        perm = torch.randperm(mask.numel())[:count]  # 随机采样
        exemplars[cid] = features[mask[perm]].clone()  # 保存样本
    return exemplars  # 返回记忆


def compute_class_means(exemplars: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:  # 定义类别均值函数
    means: Dict[int, torch.Tensor] = {}  # 初始化均值字典
    for cid, feats in exemplars.items():  # 遍历记忆
        if feats.numel() == 0:  # 若为空
            continue  # 跳过
        means[cid] = F.normalize(feats.mean(dim=0), dim=0)  # 计算并归一化均值
    return means  # 返回均值


def main() -> None:  # 定义主函数
    args = parse_args()  # 解析命令行参数
    cfg = load_configs(args)  # 合并配置
    ensure_dir(args.save_dir)  # 确保输出目录存在
    log_path = setup_logging(args.save_dir, 'icarl.log')  # 配置日志文件
    logger.info(f'Log file: {log_path}')  # 输出日志路径
    set_seed(args.seed)  # 设置随机种子
    device = get_device()  # 获取设备
    log_experiment_info(cfg)  # 打印实验配置

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
    sorted_items = sorted(class_map.items(), key=lambda kv: kv[1])  # 按索引排序
    new_class_names = list(cfg.get('new_class_names', ['new_cls']))  # 获取新类名称
    if 'class_names' in cfg:  # 判断是否提供旧类名称
        old_class_names = list(cfg['class_names'])  # 使用配置中的旧类名称
        old_class_count = len(old_class_names)  # 统计旧类数量
    else:  # 否则根据映射推断
        old_class_count = len(class_map) - len(new_class_names)  # 推断旧类数量
        old_class_names = [name for name, idx in sorted_items if idx < old_class_count]  # 使用映射名称
    if len(old_class_names) < old_class_count:  # 若旧类名称不足
        old_class_names.extend([f'class_{i}' for i in range(len(old_class_names), old_class_count)])  # 补全名称
    old_class_names = old_class_names[:old_class_count]  # 截断名称

    cache_root = Path(cfg.get('cache_root', args.save_dir / 'cache'))  # 获取缓存目录
    encoder = FrozenCLIPEncoder(args.backbone, args.pretrained, device, cache_root)  # 创建编码器
    return_patch = cfg.get('use_patch', False)  # 获取补丁标记

    if args.use_cache:  # 若使用缓存
        base_cache = encoder.extract_and_cache(train_loader_base, 'train', cfg.get('image_size', 224), return_patch, True)  # 加载旧类训练特征
        new_cache = encoder.extract_and_cache(train_loader_new, 'train_new', cfg.get('image_size', 224), return_patch, True)  # 加载新类训练特征
        val_base_cache = encoder.extract_and_cache(val_loader_base, 'val', cfg.get('image_size', 224), return_patch, True)  # 加载旧类验证特征
        val_new_cache = encoder.extract_and_cache(val_loader_new, 'val_new', cfg.get('image_size', 224), return_patch, True)  # 加载新类验证特征
    else:  # 否则重新提取
        base_cache = encoder.extract_and_cache(train_loader_base, 'train_icarl', cfg.get('image_size', 224), return_patch, False)  # 生成旧类训练特征
        new_cache = encoder.extract_and_cache(train_loader_new, 'train_new_icarl', cfg.get('image_size', 224), return_patch, False)  # 生成新类训练特征
        val_base_cache = encoder.extract_and_cache(val_loader_base, 'val_icarl', cfg.get('image_size', 224), return_patch, False)  # 生成旧类验证特征
        val_new_cache = encoder.extract_and_cache(val_loader_new, 'val_new_icarl', cfg.get('image_size', 224), return_patch, False)  # 生成新类验证特征

    raw_new_ids = {int(v) for v in new_cache['labels'].tolist()}  # 收集新类标签
    raw_new_ids.update(int(v) for v in val_new_cache['labels'].tolist())  # 合并验证标签
    sorted_new_ids = sorted(raw_new_ids)  # 排序标签
    if not sorted_new_ids:  # 若集合为空
        sorted_new_ids = [old_class_count]  # 使用默认标签
    if len(new_class_names) < len(sorted_new_ids):  # 若新类名称不足
        new_class_names.extend([f'new_cls_{i}' for i in range(len(new_class_names), len(sorted_new_ids))])  # 补全名称
    elif len(new_class_names) > len(sorted_new_ids):  # 若名称多余
        new_class_names = new_class_names[:len(sorted_new_ids)]  # 截断
    total_classes = old_class_count + len(new_class_names)  # 计算总类别数
    class_names = old_class_names + new_class_names  # 合并名称
    id_to_idx = {cid: old_class_count + idx for idx, cid in enumerate(sorted_new_ids)}  # 构建标签映射

    train_new_labels = torch.tensor([id_to_idx[int(cid)] for cid in new_cache['labels'].tolist()], dtype=torch.long)  # 映射训练新类标签
    val_new_labels = torch.tensor([id_to_idx[int(cid)] for cid in val_new_cache['labels'].tolist()], dtype=torch.long)  # 映射验证新类标签

    exemplars = select_exemplars(base_cache['features'], base_cache['labels'], cfg['memory_per_class'], old_class_count)  # 选择旧类记忆
    exemplar_feats = []
    exemplar_labels = []
    for cid, feats in exemplars.items():  # 遍历记忆
        exemplar_feats.append(feats)  # 保存特征
        exemplar_labels.append(torch.full((feats.size(0),), cid, dtype=torch.long))  # 保存标签
    if exemplar_feats:  # 若存在记忆
        exemplar_features = torch.cat(exemplar_feats, dim=0)  # 合并特征
        exemplar_labels = torch.cat(exemplar_labels, dim=0)  # 合并标签
    else:  # 若记忆为空
        exemplar_features = torch.empty(0, encoder.output_dim(return_patch))  # 创建空特征
        exemplar_labels = torch.empty(0, dtype=torch.long)  # 创建空标签
    torch.save({'features': exemplar_features.cpu(), 'labels': exemplar_labels.cpu()}, args.save_dir / 'exemplars.pt')  # 保存记忆

    train_features = torch.cat([new_cache['features'], exemplar_features], dim=0).float()  # 合并训练特征
    train_labels = torch.cat([train_new_labels, exemplar_labels], dim=0).long()  # 合并训练标签
    train_dataset = TensorDataset(train_features, train_labels)  # 构建训练数据集
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)  # 创建训练加载器

    teacher = LinearClassifier(encoder.output_dim(return_patch), old_class_count)  # 创建教师分类器
    classifier = LinearClassifier(encoder.output_dim(return_patch), old_class_count)  # 创建学生分类器
    checkpoint = torch.load(args.resume_ckpt, map_location='cpu')  # 加载旧类权重
    teacher.load_state_dict(checkpoint['classifier'])  # 加载教师权重
    classifier.load_state_dict(checkpoint['classifier'])  # 加载学生初始权重
    classifier.expand_classes(len(new_class_names))  # 扩展新增类别

    teacher.to(device)  # 将教师移动到设备
    classifier.to(device)  # 将学生移动到设备
    teacher.eval()  # 冻结教师

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])  # 创建优化器
    criterion = torch.nn.CrossEntropyLoss()  # 创建交叉熵损失
    temperature = cfg['temperature']  # 读取温度
    alpha = cfg['alpha']  # 读取蒸馏权重
    best_top1 = 0.0  # 初始化最佳精度

    val_features = torch.cat([val_base_cache['features'], val_new_cache['features']], dim=0).float()  # 合并验证特征
    val_labels = torch.cat([val_base_cache['labels'], val_new_labels], dim=0).long()  # 合并验证标签
    val_loader = DataLoader(TensorDataset(val_features, val_labels), batch_size=cfg['batch_size'], shuffle=False)  # 创建验证加载器

    for epoch in range(1, cfg['epochs'] + 1):  # 遍历训练轮次
        classifier.train()  # 切换到训练模式
        for feats, labels in train_loader:  # 遍历训练批次
            feats = feats.to(device)  # 移动特征
            labels = labels.to(device)  # 移动标签
            optimizer.zero_grad(set_to_none=True)  # 清空梯度
            logits = classifier(feats)  # 计算学生输出
            with torch.no_grad():  # 关闭梯度
                teacher_logits = teacher(feats)  # 计算教师输出
            loss_ce = criterion(logits, labels)  # 交叉熵损失
            distill_student = F.log_softmax(logits[:, :old_class_count] / temperature, dim=1)  # 学生旧类分布
            distill_teacher = F.softmax(teacher_logits / temperature, dim=1)  # 教师旧类分布
            loss_kd = F.kl_div(distill_student, distill_teacher, reduction='batchmean') * (temperature * temperature)  # 蒸馏损失
            loss = (1.0 - alpha) * loss_ce + alpha * loss_kd  # 总损失
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
        logger.info(f'[iCaRL] Epoch {epoch}: top1={acc[1]:.4f} top5={acc[5]:.4f}')  # 打印日志
        if acc[1] > best_top1:  # 判断是否刷新
            best_top1 = acc[1]  # 更新最佳Top1
            torch.save({'classifier': classifier.state_dict(), 'config': cfg}, args.save_dir / 'best_icarl.ckpt')  # 保存最佳权重

    torch.save({'classifier': classifier.state_dict(), 'config': cfg}, args.save_dir / 'last_icarl.ckpt')  # 保存最终权重

    prototype_dict = compute_class_means(exemplars)  # 初始化原型字典
    for mapped_cid in torch.unique(train_new_labels).tolist():  # 遍历新类
        feats = new_cache['features'][train_new_labels == mapped_cid]  # 取出特征
        if feats.numel() == 0:  # 若为空
            continue  # 跳过
        prototype_dict[mapped_cid] = F.normalize(feats.mean(dim=0), dim=0)  # 更新原型

    if prototype_dict:  # 若存在原型
        proto_ids = sorted(prototype_dict.keys())  # 排序类别
        proto_matrix = torch.stack([prototype_dict[cid] for cid in proto_ids], dim=0)  # 构建原型矩阵
        proto_matrix = F.normalize(proto_matrix, dim=1)  # 归一化原型
        proto_id_tensor = torch.tensor(proto_ids, dtype=torch.long)  # 转为张量
        norm_val_feats = F.normalize(val_features, dim=1)  # 归一化验证特征
        sims = torch.matmul(norm_val_feats, proto_matrix.t())  # 计算相似度
        indices = sims.argmax(dim=1)  # 找到最近原型
        proto_predictions = proto_id_tensor[indices]  # 获得预测
    else:  # 若无原型
        proto_predictions = logits_tensor.argmax(dim=1)  # 回退到分类器预测

    confusion = build_confusion(logits_tensor, labels_tensor, total_classes)  # 基于分类器输出构建混淆矩阵
    plot_confusion(confusion, class_names, args.save_dir / 'confusion_icarl.png')  # 绘制混淆矩阵
    save_json(confusion_to_dict(confusion, class_names), args.save_dir / 'confusion_icarl.json')  # 保存混淆矩阵
    per_class_acc = per_class_accuracy(confusion, class_names)  # 计算逐类精度
    save_json(per_class_acc, args.save_dir / 'per_class_accuracy_classifier.json')  # 保存分类器逐类精度

    proto_logits = F.one_hot(proto_predictions, total_classes).float()  # 构建原型预测logits
    proto_confusion = build_confusion(proto_logits, labels_tensor, total_classes)  # 构建原型混淆矩阵
    save_json(confusion_to_dict(proto_confusion, class_names), args.save_dir / 'confusion_icarl_proto.json')  # 保存原型混淆矩阵
    save_json(per_class_accuracy(proto_confusion, class_names), args.save_dir / 'per_class_accuracy_proto.json')  # 保存原型逐类精度

    proto_top1 = float((proto_predictions == labels_tensor).float().mean().item())  # 计算原型Top1

    summary_metrics = summarize_incremental_metrics(
        confusion,
        class_names,
        old_class_names,
        baseline_old_acc=cfg.get('baseline_old_acc'),
        baseline_new_acc=cfg.get('baseline_new_acc'),
    )  # 汇总指标
    summary_metrics['best_top1'] = best_top1  # 写入最佳Top1
    summary_metrics['memory_per_class'] = cfg['memory_per_class']  # 写入记忆大小
    summary_metrics['temperature'] = temperature  # 写入温度
    summary_metrics['alpha'] = alpha  # 写入蒸馏权重
    summary_metrics['proto_top1'] = proto_top1  # 写入原型精度
    save_json(summary_metrics, args.save_dir / 'metrics_icarl.json')  # 保存指标


if __name__ == '__main__':  # 判断是否主入口
    main()  # 执行主函数
