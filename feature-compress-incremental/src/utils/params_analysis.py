# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

from pathlib import Path  # 导入路径模块
from typing import Dict  # 导入类型别名

import torch  # 导入torch

from ..models.clip_wrapper import FrozenCLIPEncoder  # 导入CLIP封装
from ..models.compress_head import CompressionHead  # 导入压缩模块
from ..models.classifier import LinearClassifier  # 导入分类器
from ..utils.common import load_yaml_config, ensure_dir  # 导入通用工具


def count_parameters(model: torch.nn.Module) -> int:  # 定义参数统计函数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计可训练参数总数


def analyze_model(  # 定义分析函数
    backbone: str,
    pretrained: str,
    n_compress: int,
    class_count: int,
    device: str,
    cache_root: Path,
) -> Dict[str, int]:
    encoder = FrozenCLIPEncoder(backbone, pretrained, torch.device(device), cache_root)  # 创建编码器
    return_patch = False  # 只分析全局特征
    input_dim = encoder.output_dim(return_patch)  # 获取输出维度
    compressor = CompressionHead(input_dim, n_compress, return_patch)  # 创建压缩模块
    classifier = LinearClassifier(compressor.output_dim(), class_count)  # 创建分类器
    params = {
        'clip_backbone': count_parameters(encoder.model),  # 主干参数
        'compress_head': count_parameters(compressor),  # 压缩参数
        'classifier': count_parameters(classifier),  # 分类器参数
    }
    params['compress_pipeline'] = params['compress_head'] + params['classifier']  # 总压缩参数
    return params  # 返回统计结果


def load_split_sizes(train_list: Path, val_list: Path) -> Dict[str, int]:  # 定义数据大小统计
    def count_lines(path: Path) -> int:  # 定义行计数函数
        if not path.exists():  # 若文件不存在
            return 0  # 返回0
        return sum(1 for _ in path.open('r', encoding='utf-8') if _.strip())  # 统计非空行

    return {'train_samples': count_lines(train_list), 'val_samples': count_lines(val_list)}  # 返回统计


def load_replay_buffer(path: Path) -> Dict[str, int]:  # 定义回放缓存统计
    if not path.exists():  # 若文件不存在
        return {'exemplar_count': 0, 'feature_dim': 0}  # 返回默认值
    cache = torch.load(path, map_location='cpu')  # 加载缓存
    feats = cache.get('features')  # 获取特征
    if feats is None:  # 若无特征
        return {'exemplar_count': 0, 'feature_dim': 0}  # 返回默认值
    return {'exemplar_count': feats.size(0), 'feature_dim': feats.size(1)}  # 返回数量和维度


def main() -> None:  # 定义主函数（命令行入口预留）
    import argparse  # 延迟导入参数解析

    parser = argparse.ArgumentParser(description='模型与存储成本分析')  # 创建解析器
    parser.add_argument('--paths', type=Path, required=True, help='指定路径配置文件')  # 添加路径参数
    parser.add_argument('--task_cfg', type=Path, required=True, help='指定任务配置文件')  # 添加任务参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16', help='指定CLIP骨干名称')  # 添加骨干参数
    parser.add_argument('--pretrained', type=str, default='openai', help='指定预训练权重标识或路径')  # 添加预训练参数
    parser.add_argument('--n_compress', type=int, default=32, help='指定压缩维度')  # 添加压缩维度
    parser.add_argument('--class_count', type=int, default=10, help='指定类别数量')  # 添加类别数量
    parser.add_argument('--device', type=str, default='cuda', help='指定设备')  # 添加设备
    parser.add_argument('--exemplar_cache', type=Path, help='指定回放缓存文件（可选）')  # 添加回放文件
    parser.add_argument('--save_path', type=Path, required=True, help='指定输出JSON路径')  # 添加输出路径
    args = parser.parse_args()  # 解析命令行

    paths_cfg = load_yaml_config(args.paths)  # 加载路径配置
    task_cfg = load_yaml_config(args.task_cfg)  # 加载任务配置
    cache_root = Path(paths_cfg.get('cache_root', 'outputs/cache'))  # 获取缓存目录

    params = analyze_model(args.backbone, args.pretrained, args.n_compress, args.class_count, args.device, cache_root)  # 统计参数
    data_sizes = load_split_sizes(Path(paths_cfg['train_list']), Path(paths_cfg['val_list']))  # 统计样本

    analysis = {'params': params, 'data': data_sizes}  # 汇总结果

    if args.exemplar_cache:  # 若提供回放缓存
        analysis['replay_buffer'] = load_replay_buffer(args.exemplar_cache)  # 统计回放

    ensure_dir(args.save_path.parent)  # 确保输出目录存在
    from ..utils.common import save_json  # 延迟导入保存函数

    save_json(analysis, args.save_path)  # 写入JSON


if __name__ == '__main__':  # 判断是否主入口
    main()  # 执行主函数
