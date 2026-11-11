# -*- coding: utf-8 -*-
"""
模型 4（结构创新三）：GELU 激活 + Group Normalization 版 VGG
============================================================
与基线网络的卷积层数、通道数、池化位置完全相同，只替换两类组件：
  1. 激活函数 ReLU -> GELU：
     ReLU 在负半轴直接置零，GELU 则依据输入的概率分布平滑地门控，
     处处可导，在很多视觉任务上比 ReLU 更稳定；
  2. 归一化层 BatchNorm -> GroupNorm：
     BatchNorm 的统计量依赖 batch 内样本，batch 较小时效果波动明显；
     GroupNorm 把每个样本的通道分组后在组内归一化，统计量与 batch 无关。
"""
import torch.nn as nn

from .modules import ConvBlock


class VGGGELUGN(nn.Module):
    """GELU 激活 + GroupNorm 归一化版 VGG。"""

    model_name = "gelu_gn"
    model_desc = "GELU激活+GroupNorm版"

    def __init__(self, num_classes=10):
        super().__init__()
        layers = []
        in_channels = 3
        # 与基线相同的配置，只是每个卷积块指定 norm="gn"、act="gelu"
        for item in [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"]:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(ConvBlock(in_channels, item,
                                        norm="gn", act="gelu"))
                in_channels = item
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

    def _initialize_weights(self):
        # GELU 配合 Kaiming 初始化（按 ReLU 口径的近似，工程上常用）
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
