# -*- coding: utf-8 -*-
"""
全局配置：路径、超参数、数据集常量。
所有脚本统一从这里读取配置，避免参数散落在各处。
"""
import os

# ---------------------------------------------------------------------------
# 路径配置（基于本文件所在目录，整体移动到别的电脑上也能直接运行）
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")            # CIFAR-10 数据集缓存目录
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")  # 模型权重
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "logs")     # 每个 epoch 的训练日志
FIGURE_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "results", "sample_images")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "summary.json")

for _d in (DATA_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR, SAMPLE_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# 训练超参数（4 个模型使用完全相同的设置，保证对比公平）
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4

# ---------------------------------------------------------------------------
# CIFAR-10 数据集常量
# ---------------------------------------------------------------------------
# 10 个类别（英文标签用于程序，中文释义用于报告/演示输出）
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_CLASSES_CN = [
    "飞机", "汽车", "鸟", "猫", "鹿",
    "狗", "青蛙", "马", "船", "卡车",
]

# CIFAR-10 训练集统计量（RGB 三通道均值/标准差），用于归一化
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2446)
