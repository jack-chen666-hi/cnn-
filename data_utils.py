# -*- coding: utf-8 -*-
"""
数据加载与预处理
==============
  - set_seed：固定所有随机种子，保证实验可复现；
  - LocalCIFAR10：从本地 npz 读取的 CIFAR-10 数据集，
    接口与 torchvision 标准数据集一致（返回 PIL 图像 + 标签）；
  - get_dataloaders：返回训练/测试 DataLoader，
    训练集带数据增强（随机裁剪 + 水平翻转），测试集仅做归一化。

说明：本机网络无法稳定访问 CIFAR-10 官方下载源，
数据通过 HuggingFace 镜像下载 parquet 后由 prepare_data.py 转换为 npz，
数据内容与官方 CIFAR-10 完全相同（5 万训练 + 1 万测试，类别均衡）。
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (BATCH_SIZE, CIFAR10_MEAN, CIFAR10_STD, DATA_DIR,
                    NUM_WORKERS)


def set_seed(seed=42):
    """固定 Python / NumPy / PyTorch 的随机种子，并让 CUDA 结果确定。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LocalCIFAR10(Dataset):
    """读取 data/cifar10_prepared 下 npz 的 CIFAR-10 数据集。"""

    def __init__(self, train=True, transform=None):
        split = "train" if train else "test"
        path = os.path.join(DATA_DIR, "cifar10_prepared", f"{split}.npz")
        data = np.load(path)
        self.images = data["images"]   # uint8 N×32×32×3
        self.labels = data["labels"]   # uint8 N
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


def _build_transforms():
    """构建训练集和测试集的预处理流水线。"""
    train_transform = transforms.Compose([
        # 四周填充 4 个像素后随机裁回 32x32，制造位置扰动
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ToTensor(),                      # PIL -> [0,1] 张量
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, test_transform


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """返回 (train_loader, test_loader)。"""
    train_transform, test_transform = _build_transforms()

    train_set = LocalCIFAR10(train=True, transform=train_transform)
    test_set = LocalCIFAR10(train=False, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0))
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0))
    return train_loader, test_loader
