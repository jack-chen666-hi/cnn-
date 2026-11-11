# -*- coding: utf-8 -*-
"""
CNN 通用组件：
  1. ConvBlock              —— 标准卷积块（卷积 + 归一化 + 激活）
  2. DepthwiseSeparableConv —— 深度可分离卷积（深度卷积 + 逐点卷积）
  3. SEBlock                —— Squeeze-and-Excitation 通道注意力模块

四个模型通过组合这些组件得到，组件之间的差异在模型文件中一目了然。
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 激活函数 / 归一化层工厂
# ---------------------------------------------------------------------------
def get_activation(name):
    """按名称返回激活函数层。"""
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        # GELU：高斯误差线性单元，Transformer 中广泛使用的平滑激活
        return nn.GELU()
    if name == "mish":
        # Mish：x * tanh(softplus(x))，自门控平滑激活
        return nn.Mish(inplace=True)
    raise ValueError(f"不支持的激活函数: {name}")


def get_norm(name, num_channels):
    """按名称返回归一化层。"""
    if name == "bn":
        # Batch Normalization：对一个 batch 内的特征做归一化
        return nn.BatchNorm2d(num_channels)
    if name == "gn":
        # Group Normalization：把通道分成 8 组，在每组内做归一化，
        # 不依赖 batch 大小。本网络通道数均为 8 的整数倍。
        return nn.GroupNorm(num_groups=8, num_channels=num_channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"不支持的归一化层: {name}")


# ---------------------------------------------------------------------------
# 1. 标准卷积块：3x3 卷积 + 归一化 + 激活
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """VGG 风格的基础卷积块，默认 3x3 卷积、padding=1，保持特征图尺寸不变。"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 norm="bn", act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# 2. 深度可分离卷积：Depthwise（空间滤波）+ Pointwise（通道融合）
# ---------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积（MobileNet 核心组件），把标准卷积拆成两步：
      depthwise：每个输入通道各自用一个 3x3 卷积核滤波（groups=in_channels）；
      pointwise：用 1x1 卷积完成通道间的信息融合。
    在相近的表达能力下，参数量和计算量远低于标准卷积。
    """

    def __init__(self, in_channels, out_channels, norm="bn", act="relu"):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            padding=1, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False,
        )
        self.norm1 = get_norm(norm, in_channels)
        self.norm2 = get_norm(norm, out_channels)
        self.act1 = get_activation(act)
        self.act2 = get_activation(act)

    def forward(self, x):
        x = self.act1(self.norm1(self.depthwise(x)))
        x = self.act2(self.norm2(self.pointwise(x)))
        return x


# ---------------------------------------------------------------------------
# 3. SE 通道注意力模块（Squeeze-and-Excitation）
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """
    SE 模块（Squeeze-and-Excitation Network, CVPR 2018）：
      squeeze：全局平均池化，把每个通道的整张特征图压缩成一个数；
      excitation：两层瓶颈全连接学习每个通道的重要性权重（0~1）；
      scale：用学到的权重对原特征图逐通道加权，突出有用特征、抑制无用特征。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)  # 瓶颈层通道数，下限为 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.avg_pool(x).view(b, c)   # squeeze
        weights = self.fc(weights).view(b, c, 1, 1)  # excitation
        return x * weights                      # scale
