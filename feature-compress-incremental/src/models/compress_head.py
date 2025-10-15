# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

import torch  # 导入torch库
import torch.nn as nn  # 导入神经网络模块


class CompressionHead(nn.Module):  # 定义压缩映射模块
    def __init__(self, input_dim: int, output_dim: int, use_patch: bool) -> None:  # 定义初始化函数
        super().__init__()  # 调用父类初始化
        self.use_patch = use_patch  # 记录输入类型
        if use_patch:  # 判断是否使用补丁输入
            self.mapper = nn.Conv2d(input_dim, output_dim, kernel_size=1)  # 创建1x1卷积
        else:  # 否则使用向量输入
            self.mapper = nn.Linear(input_dim, output_dim)  # 创建线性映射

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # 定义前向函数
        if self.use_patch:  # 判断输入是否为特征图
            mapped = self.mapper(features)  # 进行卷积映射
            pooled = mapped.mean(dim=(2, 3))  # 对空间维度平均
            return pooled  # 返回压缩向量
        return self.mapper(features)  # 返回线性映射结果

    def add_channels(self, extra_dim: int) -> None:  # 定义新增通道函数
        if extra_dim <= 0 or not self.use_patch:  # 判断是否需要扩展
            return  # 不需要则返回
        old_conv: nn.Conv2d = self.mapper  # 获取旧卷积
        new_out = old_conv.out_channels + extra_dim  # 计算新通道数
        new_conv = nn.Conv2d(old_conv.in_channels, new_out, kernel_size=1)  # 创建新卷积
        with torch.no_grad():  # 关闭梯度复制参数
            new_conv.weight[: old_conv.out_channels] = old_conv.weight  # 复制旧权重
            new_conv.bias[: old_conv.out_channels] = old_conv.bias  # 复制旧偏置
        self.mapper = new_conv  # 替换卷积层

    def output_dim(self) -> int:  # 定义输出维度函数
        if isinstance(self.mapper, nn.Conv2d):  # 判断映射类型
            return int(self.mapper.out_channels)  # 返回卷积输出通道
        return int(self.mapper.out_features)  # 返回线性输出维度

    def input_dim(self) -> int:  # 定义输入维度函数
        if isinstance(self.mapper, nn.Conv2d):  # 判断映射类型
            return int(self.mapper.in_channels)  # 返回卷积输入通道
        return int(self.mapper.in_features)  # 返回线性输入维度
