# -*- coding: utf-8 -*-
"""
模型规模度量工具（零第三方依赖）：
  - count_parameters：统计可训练参数量；
  - count_macs      ：通过 forward hook 统计单张图片的 MACs
                       （乘加操作次数，衡量计算量的常用口径）。
注意：1 次 MAC（一次乘加）按 2 次 FLOP 计，报告中会注明口径。
"""
import torch
import torch.nn as nn


def count_parameters(model):
    """返回模型可训练参数总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model, input_size=(1, 3, 32, 32)):
    """
    统计一次前向传播的 MACs（Multiply-Accumulate Operations）。
    仅统计 Conv2d 和 Linear 两类含矩阵乘加的层，归一化/激活/池化忽略。
    """
    macs_total = [0]
    hooks = []

    def conv_hook(module, inputs, output):
        batch, out_c, out_h, out_w = output.shape
        kh, kw = module.kernel_size
        # 分组卷积（含深度卷积）：每组输入通道数 = in_channels / groups
        in_c_per_group = module.in_channels // module.groups
        macs = out_h * out_w * out_c * in_c_per_group * kh * kw
        if module.bias is not None:
            macs += out_h * out_w * out_c
        macs_total[0] += macs

    def linear_hook(module, inputs, output):
        macs = module.in_features * module.out_features
        if module.bias is not None:
            macs += module.out_features
        macs_total[0] += macs

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_size, device=next(model.parameters()).device))

    for hook in hooks:
        hook.remove()
    return macs_total[0]
