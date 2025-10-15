# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

import json  # 导入JSON库
from dataclasses import dataclass  # 导入数据类装饰器
from pathlib import Path  # 导入路径类
from typing import Callable, Dict, List, Optional  # 导入类型别名

from PIL import Image  # 导入图像库
from torch.utils.data import Dataset  # 导入数据集基类
from torchvision.datasets import FakeData  # 导入假数据集

from .transforms import build_transform  # 导入变换构建函数


@dataclass  # 声明数据类装饰器
class SampleRecord:  # 定义样本记录结构
    image_path: Path  # 存储图像路径
    label: int  # 存储类别编号


class CocoClassificationDataset(Dataset):  # 定义COCO分类数据集
    def __init__(self, records: List[SampleRecord], transform: Callable) -> None:  # 定义初始化函数
        self.records = records  # 保存样本列表
        self.transform = transform  # 保存变换函数

    def __len__(self) -> int:  # 定义长度函数
        return len(self.records)  # 返回样本数量

    def __getitem__(self, index: int):  # 定义取样函数
        record = self.records[index]  # 获取指定记录
        image = Image.open(record.image_path).convert('RGB')  # 读取并转换图像
        tensor = self.transform(image)  # 应用图像变换
        return tensor, record.label  # 返回张量与标签


def read_class_map(path: Path) -> Dict[str, int]:  # 定义类别映射读取函数
    with path.open('r', encoding='utf-8') as f:  # 打开映射文件
        mapping = json.load(f)  # 解析JSON内容
    return {str(k): int(v) for k, v in mapping.items()}  # 返回标准化映射


def read_split_list(path: Path) -> List[SampleRecord]:  # 定义列表文件读取函数
    records: List[SampleRecord] = []  # 初始化记录列表
    with path.open('r', encoding='utf-8') as f:  # 打开列表文件
        for line in f:  # 遍历每行
            line = line.strip()  # 去除空白字符
            if not line:  # 判断是否为空行
                continue  # 跳过空行
            parts = line.split()  # 拆分路径与标签
            img_path = Path(parts[0])  # 解析图像路径
            label = int(parts[1])  # 解析标签编号
            records.append(SampleRecord(image_path=img_path, label=label))  # 保存记录
    return records  # 返回记录列表


def build_dataset(list_path: Path, transform: Callable) -> CocoClassificationDataset:  # 定义数据集构建函数
    records = read_split_list(list_path)  # 读取列表文件
    return CocoClassificationDataset(records, transform)  # 返回数据集实例


def build_fake_dataset(num_classes: int, image_size: int, length: int, seed: int) -> FakeData:  # 定义假数据集构建函数
    transform = build_transform(image_size, True)  # 构建训练变换
    return FakeData(size=length, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform, target_transform=None, random_offset=seed)  # 返回假数据集


def describe_dataset(records: List[SampleRecord]) -> Dict[int, int]:  # 定义数据集统计函数
    stats: Dict[int, int] = {}  # 初始化统计字典
    for rec in records:  # 遍历记录
        stats[rec.label] = stats.get(rec.label, 0) + 1  # 累加对应类别计数
    return stats  # 返回统计结果
