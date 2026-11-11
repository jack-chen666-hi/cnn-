# -*- coding: utf-8 -*-
"""
模型 1（基线）：VGG 风格卷积神经网络
=====================================
借鉴 VGG-16 的设计思想：
  - 全部使用 3x3 小卷积核，两个卷积堆叠的感受野等价于一个 5x5 卷积；
  - 卷积层之间用 2x2 最大池化把空间尺寸减半；
  - 通道数随空间尺寸缩小而翻倍（64 -> 128 -> 256 -> 512）；
  - 每个卷积层后接 BatchNorm + ReLU；
  - 最后用全局平均池化替代 VGG-16 中庞大的全连接层，再接一个线性分类头。
针对 CIFAR-10 的 32x32 输入做了精简，共 8 个卷积层。
"""
import torch.nn as nn

from .modules import ConvBlock

# 网络配置：数字代表该卷积层输出通道数，"M" 代表 2x2 最大池化
VGG_CONFIG = [64, 64, "M",
              128, 128, "M",
              256, 256, "M",
              512, 512, "M"]


class VGGBaseline(nn.Module):
    """VGG 风格基线网络（标准卷积 + BatchNorm + ReLU）。"""

    model_name = "baseline"
    model_desc = "VGG基线（标准卷积+BN+ReLU）"

    def __init__(self, num_classes=10):
        super().__init__()
        layers = []
        in_channels = 3
        for item in VGG_CONFIG:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(ConvBlock(in_channels, item, norm="bn", act="relu"))
                in_channels = item
        self.features = nn.Sequential(*layers)

        # 全局平均池化：512 张 2x2 特征图 -> 512 维向量
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

    def _initialize_weights(self):
        """权重初始化：卷积层 Kaiming 初始化，BN 权重为 1、偏置为 0。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
