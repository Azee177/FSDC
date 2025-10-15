# -*- coding: utf-8 -*-  # 指定源文件编码
from __future__ import annotations  # 启用未来注解

from pathlib import Path  # 导入路径类
from typing import Dict  # 导入类型别名

import torch  # 导入torch库
from torch.utils.data import DataLoader  # 导入数据加载器
from transformers import CLIPVisionModelWithProjection  # 导入视觉模型

from ..utils.common import cache_feature_path, load_feature_cache, save_feature_cache, Timer  # 导入工具


class FrozenCLIPEncoder:  # 定义冻结的CLIP编码器
    def __init__(self, backbone: str, pretrained: str, device: torch.device, cache_root: Path) -> None:  # 定义初始化函数
        self.backbone = backbone  # 保存骨干名称
        self.pretrained = pretrained  # 保存预训练权重路径
        self.device = device  # 保存设备
        self.cache_root = cache_root  # 保存缓存目录
        self.model = CLIPVisionModelWithProjection.from_pretrained(pretrained, local_files_only=True)  # 加载视觉模型
        self.model.to(self.device)  # 将模型移动到设备
        for param in self.model.parameters():  # 遍历模型参数
            param.requires_grad = False  # 冻结梯度
        self.model.eval()  # 设置为评估模式
        self.vision = self.model.vision_model  # 保存视觉子模块
        self.embed_dim = int(self.model.config.projection_dim)  # 记录全局嵌入维度
        self.patch_dim = int(self.vision.config.hidden_size)  # 记录补丁隐藏维度

    def _forward_global(self, images: torch.Tensor) -> torch.Tensor:  # 定义全局特征函数
        with torch.no_grad():  # 关闭梯度
            outputs = self.model(pixel_values=images)  # 前向计算
            embeds = outputs.image_embeds  # 获取图像嵌入
        return embeds  # 返回全局特征

    def _forward_patch(self, images: torch.Tensor) -> torch.Tensor:  # 定义补丁特征函数
        with torch.no_grad():  # 关闭梯度
            outputs = self.vision(pixel_values=images, output_hidden_states=False, return_dict=True)  # 通过视觉模块
            tokens = outputs.last_hidden_state  # 获取序列特征
        if tokens.size(1) > 1:  # 判断是否包含CLS
            tokens = tokens[:, 1:, :]  # 去除CLS token
        grid = int(tokens.size(1) ** 0.5)  # 计算网格边长
        patch_map = tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), grid, grid)  # 重塑为特征图
        return patch_map  # 返回补丁特征

    def encode_batch(self, images: torch.Tensor, return_patch: bool) -> torch.Tensor:  # 定义批量编码函数
        images = images.to(self.device, non_blocking=True)  # 将图像移动到设备
        if return_patch:  # 判断输出类型
            return self._forward_patch(images)  # 返回补丁特征
        return self._forward_global(images)  # 返回全局特征

    def extract_and_cache(self, loader: DataLoader, split: str, image_size: int, return_patch: bool, use_cache: bool) -> Dict[str, torch.Tensor]:  # 定义提取缓存函数
        cache_path = cache_feature_path(self.cache_root, split, self.backbone, image_size)  # 计算缓存路径
        if use_cache:  # 判断是否使用缓存
            cached = load_feature_cache(cache_path)  # 读取缓存
            if cached is not None:  # 判断缓存是否存在
                return cached  # 返回缓存内容
        storage: Dict[str, torch.Tensor] = {'features': [], 'labels': []}  # 初始化存储字典
        with Timer(f'提取{split}特征'):  # 启动计时器
            for images, labels in loader:  # 遍历数据
                feats = self.encode_batch(images, return_patch=return_patch)  # 计算特征
                storage['features'].append(feats.cpu())  # 保存特征
                storage['labels'].append(labels.cpu())  # 保存标签
        storage['features'] = torch.cat(storage['features'], dim=0)  # 拼接特征
        storage['labels'] = torch.cat(storage['labels'], dim=0)  # 拼接标签
        save_feature_cache(storage, cache_path)  # 写入缓存
        return storage  # 返回结果

    def output_dim(self, return_patch: bool) -> int:  # 定义输出维度接口
        if return_patch:  # 判断输出类型
            return int(self.patch_dim)  # 返回补丁维度
        return int(self.embed_dim)  # 返回全局嵌入维度
