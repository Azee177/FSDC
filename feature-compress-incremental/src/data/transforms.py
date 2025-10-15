# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

from typing import Callable  # 导入类型别名

from torchvision import transforms  # 导入变换模块
from torchvision.transforms.functional import InterpolationMode  # 导入插值模式


def build_transform(image_size: int, train: bool) -> Callable:  # 定义变换构建函数
    resize_op = transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC)  # 定义缩放操作
    random_crop = transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC)  # 定义随机裁剪
    center_crop = transforms.CenterCrop(image_size)  # 定义中心裁剪
    to_tensor = transforms.ToTensor()  # 定义张量转换
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))  # 定义归一化
    if train:  # 判断是否训练模式
        ops = [random_crop, to_tensor, normalize]  # 组合训练操作
    else:  # 否则为验证模式
        ops = [resize_op, center_crop, to_tensor, normalize]  # 组合验证操作
    return transforms.Compose(ops)  # 返回复合变换
