# -*- coding: utf-8 -*-
"""
数据准备脚本
============
把 data/ 下从 HuggingFace 镜像下载的 CIFAR-10 parquet 文件解码，
转换成训练脚本直接读取的 npz 格式（data/cifar10_prepared/）。

只需运行一次：
  python prepare_data.py

npz 中保存：
  images：uint8 数组，形状 N×32×32×3（RGB）；
  labels：uint8 数组，形状 N，取值 0~9。
"""
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

from config import CIFAR10_CLASSES, DATA_DIR


def decode_parquet(parquet_path):
    """读取 parquet 并把图像列解码成 uint8 数组。"""
    table = pd.read_parquet(parquet_path)
    images = np.zeros((len(table), 32, 32, 3), dtype=np.uint8)
    for idx, item in enumerate(table["img"]):
        # HF 图像列结构：{'bytes': 图片二进制, 'path': 文件名}
        image = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        if array.shape != (32, 32, 3):
            raise ValueError(f"第 {idx} 张图片尺寸异常: {array.shape}")
        images[idx] = array
    labels = table["label"].to_numpy(dtype=np.uint8)
    return images, labels


def main():
    out_dir = os.path.join(DATA_DIR, "cifar10_prepared")
    os.makedirs(out_dir, exist_ok=True)

    for split, file in (("train", "train.parquet"), ("test", "test.parquet")):
        parquet_path = os.path.join(DATA_DIR, file)
        images, labels = decode_parquet(parquet_path)
        out_path = os.path.join(out_dir, f"{split}.npz")
        np.savez(out_path, images=images, labels=labels)

        print(f"{split:5s}：{len(images):,} 张，形状 {images.shape}，"
              f"已保存 {out_path}")
        unique, counts = np.unique(labels, return_counts=True)
        dist = "，".join(
            f"{CIFAR10_CLASSES[i]} {counts[list(unique).index(i)]}"
            for i in unique)
        print(f"       类别分布：{dist}")


if __name__ == "__main__":
    main()
