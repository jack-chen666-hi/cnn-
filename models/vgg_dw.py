# -*- coding: utf-8 -*-
"""
模型 2（结构创新一）：深度可分离卷积版 VGG
==========================================
与基线网络的唯一区别：除第一层（3 通道输入，通道数太少，按 MobileNet
惯例保留标准卷积）外，其余 7 个卷积层全部替换为深度可分离卷积
（DepthwiseSeparableConv）。

深度可分离卷积把标准卷积的“空间滤波 + 通道融合”拆成两步，
参数量和计算量可以下降到原来的几分之一，是移动端 CNN 的经典设计。
网络的整体层数、通道数、池化位置、分类头与基线完全一致。
"""
import torch.nn as nn

from .modules import ConvBlock, DepthwiseSeparableConv


class VGGDepthwise(nn.Module):
    """深度可分离卷积版 VGG（MobileNet 思想）。"""

    model_name = "dw"
    model_desc = "深度可分离卷积版（DWConv）"

    def __init__(self, num_classes=10):
        super().__init__()
        # 显式按 4 个 stage 构建，结构与基线一一对应
        self.stage1 = nn.Sequential(
            ConvBlock(3, 64),                 # 首层保留标准卷积
            DepthwiseSeparableConv(64, 64),
            nn.MaxPool2d(2, 2),               # 32 -> 16
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            DepthwiseSeparableConv(128, 128),
            nn.MaxPool2d(2, 2),               # 16 -> 8
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            DepthwiseSeparableConv(256, 256),
            nn.MaxPool2d(2, 2),               # 8 -> 4
        )
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
            DepthwiseSeparableConv(512, 512),
            nn.MaxPool2d(2, 2),               # 4 -> 2
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

    def _initialize_weights(self):
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
