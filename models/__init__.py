# -*- coding: utf-8 -*-
"""
模型注册表：通过字符串名称即可构建对应模型，
训练脚本按名称依次训练，新增模型时只需在此注册。
"""
from .vgg_baseline import VGGBaseline
from .vgg_dw import VGGDepthwise
from .vgg_se import VGGSE
from .vgg_gelu_gn import VGGGELUGN

MODEL_REGISTRY = {
    "baseline": VGGBaseline,
    "dw": VGGDepthwise,
    "se": VGGSE,
    "gelu_gn": VGGGELUGN,
}

# 精简实验默认训练顺序
MODEL_ORDER = ["baseline", "dw", "se", "gelu_gn"]
