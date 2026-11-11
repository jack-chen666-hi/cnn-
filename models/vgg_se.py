# -*- coding: utf-8 -*-
"""
模型 3（结构创新二）：引入 SE 通道注意力的 VGG
==============================================
在基线网络的每个 stage 末尾、最大池化之前插入一个 SEBlock，
共 4 个 SE 模块。网络主体（卷积、BN、ReLU、分类头）与基线完全相同，
新增的参数仅来自 SE 模块内部的瓶颈全连接层。

SE 模块让网络根据全局信息自动学习“每个通道该被关注多少”，
对特征图进行通道级的重新加权，从而提升判别能力。
"""
import torch.nn as nn

from .modules import ConvBlock, SEBlock


class VGGSE(nn.Module):
    """带 Squeeze-and-Excitation 通道注意力的 VGG。"""

    model_name = "se"
    model_desc = "SE注意力版（SE-Net）"

    def __init__(self, num_classes=10):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            SEBlock(64),
            nn.MaxPool2d(2, 2),               # 32 -> 16
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            SEBlock(128),
            nn.MaxPool2d(2, 2),               # 16 -> 8
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            SEBlock(256),
            nn.MaxPool2d(2, 2),               # 8 -> 4
        )
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            SEBlock(512),
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
