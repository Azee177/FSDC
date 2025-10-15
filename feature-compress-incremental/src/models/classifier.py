# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

import torch  # 导入torch库
import torch.nn as nn  # 导入神经网络模块


class LinearClassifier(nn.Module):  # 定义线性分类器
    def __init__(self, input_dim: int, num_classes: int) -> None:  # 定义初始化函数
        super().__init__()  # 调用父类初始化
        self.head = nn.Linear(input_dim, num_classes)  # 创建线性层

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # 定义前向函数
        return self.head(features)  # 返回分类结果

    def expand_classes(self, num_new: int) -> None:  # 定义扩展类别函数
        if num_new <= 0:  # 判断是否需要扩展
            return  # 不需要则返回
        old_head = self.head  # 保存旧线性层
        in_dim = old_head.in_features  # 获取输入维度
        out_dim = old_head.out_features  # 获取输出维度
        new_head = nn.Linear(in_dim, out_dim + num_new)  # 创建新线性层
        with torch.no_grad():  # 关闭梯度复制参数
            new_head.weight[:out_dim] = old_head.weight  # 复制旧权重
            new_head.bias[:out_dim] = old_head.bias  # 复制旧偏置
        self.head = new_head  # 替换线性层

    def num_classes(self) -> int:  # 定义查询函数
        return int(self.head.out_features)  # 返回类别数量
