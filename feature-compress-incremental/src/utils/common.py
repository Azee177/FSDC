# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解功能

import json  # 导入JSON库
import os  # 导入操作系统库
import random  # 导入随机库
import time  # 导入时间库
from dataclasses import dataclass  # 导入数据类装饰器
from pathlib import Path  # 导入路径类
from typing import Any, Dict, Optional  # 导入类型别名

import numpy as np  # 导入numpy库
import torch  # 导入torch库
import yaml  # 导入yaml库
from loguru import logger  # 导入loguru日志库


def ensure_dir(path: Path) -> None:  # 定义目录创建函数
    path.mkdir(parents=True, exist_ok=True)  # 创建目录并忽略已存在情况


def load_yaml_config(path: Path) -> Dict[str, Any]:  # 定义YAML读取函数
    with path.open('r', encoding='utf-8') as f:  # 打开配置文件
        data = yaml.safe_load(f)  # 解析YAML内容
    return data  # 返回变量字典


def save_json(data: Dict[str, Any], path: Path) -> None:  # 定义JSON保存函数
    ensure_dir(path.parent)  # 确保父目录存在
    with path.open('w', encoding='utf-8') as f:  # 打开目标文件
        json.dump(data, f, ensure_ascii=False, indent=2)  # 写入JSON数据


def set_seed(seed: int) -> None:  # 定义随机种子设置函数
    random.seed(seed)  # 设置Python随机种子
    np.random.seed(seed)  # 设置Numpy随机种子
    torch.manual_seed(seed)  # 设置CPU随机种子
    if torch.cuda.is_available():  # 判断GPU是否可用
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机种子


def count_parameters(model: torch.nn.Module) -> int:  # 定义参数统计函数
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计可训练参数数量
    return int(total)  # 返回整数参数量


def get_device(prefer: Optional[str] = None) -> torch.device:  # 定义设备选择函数
    if prefer is not None:  # 判断是否指定设备
        return torch.device(prefer)  # 直接返回指定设备
    if torch.cuda.is_available():  # 判断是否存在GPU
        return torch.device('cuda')  # 返回CUDA设备
    return torch.device('cpu')  # 默认返回CPU设备


def cache_file_name(split: str, backbone: str, image_size: int) -> str:  # 定义缓存文件名函数
    safe_backbone = backbone.replace('/', '-')  # 替换非法字符
    return f'{safe_backbone}_size{image_size}_{split}.pt'  # 返回拼接文件名


def cache_feature_path(cache_root: Path, split: str, backbone: str, image_size: int) -> Path:  # 定义缓存路径函数
    ensure_dir(cache_root)  # 确保缓存目录存在
    return cache_root / cache_file_name(split, backbone, image_size)  # 拼接得到缓存路径


def load_feature_cache(cache_path: Path) -> Optional[Dict[str, Any]]:  # 定义缓存加载函数
    if not cache_path.exists():  # 判断缓存是否存在
        return None  # 若不存在返回空
    logger.info(f'加载缓存特征: {cache_path}')  # 打印日志
    return torch.load(cache_path, map_location='cpu')  # 加载缓存并返回


def save_feature_cache(data: Dict[str, Any], cache_path: Path) -> None:  # 定义缓存保存函数
    ensure_dir(cache_path.parent)  # 确保目录存在
    torch.save(data, cache_path)  # 保存张量到磁盘


@dataclass  # 声明数据类装饰器
class Timer:  # 定义计时器类
    name: str  # 保存计时名称

    def __post_init__(self) -> None:  # 定义后初始化函数
        self.start_time: float = 0.0  # 初始化开始时间
        self.elapsed: float = 0.0  # 初始化耗时变量

    def __enter__(self) -> 'Timer':  # 定义上下文进入函数
        self.start_time = time.time()  # 记录开始时间
        logger.info(f'开始计时: {self.name}')  # 输出开始日志
        return self  # 返回自身实例

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # 定义上下文退出函数
        self.elapsed = time.time() - self.start_time  # 计算耗时
        logger.info(f'结束计时: {self.name}, 耗时 {self.elapsed:.2f} 秒')  # 输出耗时日志


def format_seconds(seconds: float) -> str:  # 定义时间格式化函数
    minutes = int(seconds // 60)  # 计算分钟数
    remain = seconds - minutes * 60  # 计算剩余秒数
    return f'{minutes}分{remain:.1f}秒'  # 返回格式化结果



_LOG_SINKS: set[str] = set()  # 记录已注册的日志文件路径

def setup_logging(log_dir: Path, filename: str = 'run.log', level: str = 'INFO') -> Path:  # 定义日志保存函数
    ensure_dir(log_dir)  # 确保日志目录存在
    log_path = log_dir / filename  # 构建日志文件路径
    key = str(log_path.resolve())  # 记录绝对路径避免重复添加
    if key not in _LOG_SINKS:  # 判断是否已注册
        logger.add(log_path, level=level, enqueue=True)  # 添加文件日志
        _LOG_SINKS.add(key)  # 记录已注册
    return log_path  # 返回日志路径

def log_experiment_info(cfg: Dict[str, Any]) -> None:  # 定义实验信息打印函数
    logger.info('===== 实验配置 =====')  # 输出分隔线
    for key, value in cfg.items():  # 遍历配置项
        logger.info(f'{key}: {value}')  # 打印键值对
