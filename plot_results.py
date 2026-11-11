# -*- coding: utf-8 -*-
"""
结果可视化脚本
==============
读取 results/logs 下的真实训练日志与 checkpoints 权重，生成：
  1. training_curves.png  —— 4 模型训练损失 / 测试准确率对比曲线；
  2. confusion_matrix.png —— 最佳模型在 1 万张测试图上的混淆矩阵；
  3. dataset_samples.png  —— CIFAR-10 训练集样本展示。
所有图表均由真实数据生成，不使用任何手工编造的数字。
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (CIFAR10_CLASSES_CN, CIFAR10_MEAN, CIFAR10_STD,
                    CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR, SUMMARY_PATH)
from data_utils import LocalCIFAR10, set_seed
from models import MODEL_REGISTRY

# Windows 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"baseline": "#1f77b4", "dw": "#ff7f0e",
          "se": "#2ca02c", "gelu_gn": "#d62728"}


def read_log(model_name):
    """读取某个模型的逐轮 CSV 日志。"""
    path = os.path.join(LOG_DIR, f"{model_name}_log.csv")
    rows = {"epoch": [], "train_loss": [], "test_acc": []}
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows["epoch"].append(int(row["epoch"]))
            rows["train_loss"].append(float(row["train_loss"]))
            rows["test_acc"].append(float(row["test_acc"]))
    return rows


def plot_training_curves(model_names):
    """画训练损失和测试准确率对比曲线（1x2 子图）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in model_names:
        log = read_log(name)
        desc = MODEL_REGISTRY[name].model_desc
        axes[0].plot(log["epoch"], log["train_loss"],
                     color=COLORS[name], label=desc, linewidth=1.6)
        axes[1].plot(log["epoch"], log["test_acc"],
                     color=COLORS[name], label=desc, linewidth=1.6)

    axes[0].set_xlabel("训练轮次 Epoch")
    axes[0].set_ylabel("训练损失 CrossEntropy Loss")
    axes[0].set_title("训练损失对比曲线")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("训练轮次 Epoch")
    axes[1].set_ylabel("测试准确率 (%)")
    axes[1].set_title("测试准确率对比曲线")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "training_curves.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"已保存: {out}")


@torch.no_grad()
def plot_confusion_matrix(best_name, device):
    """用最佳模型对整个测试集预测，画 10x10 混淆矩阵。"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = LocalCIFAR10(train=False, transform=test_transform)
    loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = MODEL_REGISTRY[best_name](num_classes=10).to(device)
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, f"{best_name}.pth"),
        map_location=device, weights_only=True))
    model.eval()

    matrix = np.zeros((10, 10), dtype=np.int64)
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        for true, pred in zip(labels.numpy(), preds):
            matrix[true, pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(10), CIFAR10_CLASSES_CN, rotation=45, ha="right")
    ax.set_yticks(range(10), CIFAR10_CLASSES_CN)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"混淆矩阵（{MODEL_REGISTRY[best_name].model_desc}，"
                 f"测试集共 {matrix.sum()} 张）")
    threshold = matrix.max() * 0.5
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black",
                    fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "confusion_matrix.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"已保存: {out}")
    return matrix


def plot_dataset_samples():
    """展示训练集 5x5 样本（直接读取原始 PIL 图像，无需归一化）。"""
    set_seed(42)
    train_set = LocalCIFAR10(train=True)
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        image, label = train_set[idx]
        ax.imshow(image)
        ax.set_title(CIFAR10_CLASSES_CN[label], fontsize=9)
        ax.axis("off")
    fig.suptitle("CIFAR-10 训练集样本示例", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "dataset_samples.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"已保存: {out}")


def main():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    names = [n for n in ("baseline", "dw", "se", "gelu_gn") if n in summary]

    plot_dataset_samples()
    plot_training_curves(names)
    best = max(names, key=lambda n: summary[n]["best_test_acc"])
    print(f"最佳模型为: {MODEL_REGISTRY[best].model_desc}，"
          f"准确率 {summary[best]['best_test_acc']:.2f}%")
    plot_confusion_matrix(best, torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
