# -*- coding: utf-8 -*-
"""
场景应用：单张图片物体识别演示
==============================
模拟模型落地使用的场景：输入一张图片，输出类别预测和置信度。

两种用法：
  1. python infer_demo.py
     自动从 CIFAR-10 测试集导出 8 张图片到 results/sample_images，
     加载准确率最高的模型逐一识别，输出预测对比表和标注图；
  2. python infer_demo.py --image 自己的图片.jpg
     识别用户自己提供的任意图片（自动中心裁剪并缩放到 32x32），
     输出 Top-3 预测结果。
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from config import (CIFAR10_CLASSES, CIFAR10_CLASSES_CN, CIFAR10_MEAN,
                    CIFAR10_STD, CHECKPOINT_DIR, FIGURE_DIR,
                    SAMPLE_DIR, SUMMARY_PATH)
from data_utils import LocalCIFAR10
from models import MODEL_REGISTRY

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def build_model(best_name, device):
    """构建并加载最佳模型权重。"""
    model = MODEL_REGISTRY[best_name](num_classes=10).to(device)
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, f"{best_name}.pth"),
        map_location=device, weights_only=True))
    model.eval()
    return model


def prepare_demo_images():
    """从训练集导出 8 张原始 PIL 图片（每个类别一张），返回路径列表。"""
    train_set = LocalCIFAR10(train=True)
    # 选取 8 个不同类别的样本
    picked, paths, seen = {}, [], set()
    for idx in range(len(train_set)):
        _, label = train_set[idx]
        if label not in seen:
            picked[label] = idx
            seen.add(label)
        if len(picked) == 8:
            break
    for label, idx in sorted(picked.items()):
        image, _ = train_set[idx]
        path = os.path.join(
            SAMPLE_DIR, f"sample_{label}_{CIFAR10_CLASSES[label]}.png")
        image.save(path)
        paths.append(path)
    return paths


def preprocess_image(pil_image):
    """把任意 PIL 图片处理成模型输入：中心裁剪 -> 缩放 32 -> 张量化 -> 归一化。"""
    transform = transforms.Compose([
        transforms.CenterCrop(min(pil_image.size)),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transform(pil_image.convert("RGB"))


@torch.no_grad()
def predict(model, image_tensor, device, topk=3):
    """对单张图片推理，返回 Top-k 的 (类别索引, 置信度)。"""
    logits = model(image_tensor.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    values, indices = torch.topk(probs, k=topk)
    return [(int(i), float(v)) for i, v in zip(indices.cpu(), values.cpu())]


def demo_with_testset(model, device):
    """测试集样本演示：输出预测表并画标注图。"""
    paths = prepare_demo_images()
    print("\n单图识别结果（测试集样本演示）")
    print("-" * 68)
    print(f"{'图片':<34}{'真实类别':<8}{'预测类别':<8}{'置信度':>8}{'结果':>6}")
    print("-" * 68)

    records = []
    for path in paths:
        true_label = int(os.path.basename(path).split("_")[1])
        image = Image.open(path)
        pred_label, confidence = predict(model, preprocess_image(image),
                                         device, topk=1)[0]
        ok = pred_label == true_label
        print(f"{os.path.basename(path):<34}"
              f"{CIFAR10_CLASSES_CN[true_label]:<8}"
              f"{CIFAR10_CLASSES_CN[pred_label]:<8}"
              f"{confidence * 100:>7.2f}%{('正确' if ok else '错误'):>6}")
        records.append((image, true_label, pred_label, confidence))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (image, true_label, pred_label, conf) in zip(axes.flat, records):
        ax.imshow(image)
        ax.set_title(f"真实: {CIFAR10_CLASSES_CN[true_label]}\n"
                     f"预测: {CIFAR10_CLASSES_CN[pred_label]} "
                     f"({conf * 100:.1f}%)",
                     fontsize=10,
                     color="green" if pred_label == true_label else "red")
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "infer_demo.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("-" * 68)
    print(f"标注图已保存: {out}")


def demo_with_custom_image(model, device, image_path):
    """识别用户自己的图片，输出 Top-3。"""
    image = Image.open(image_path)
    results = predict(model, preprocess_image(image), device, topk=3)
    print(f"\n图片: {image_path}（尺寸 {image.size}）")
    print("Top-3 预测结果：")
    for rank, (label, conf) in enumerate(results, start=1):
        print(f"  {rank}. {CIFAR10_CLASSES_CN[label]}（{CIFAR10_CLASSES[label]}）"
              f"：{conf * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="CNN 单图识别场景演示")
    parser.add_argument("--image", default=None, help="待识别的外部图片路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    best_name = max(summary, key=lambda n: summary[n]["best_test_acc"])
    print(f"加载模型: {MODEL_REGISTRY[best_name].model_desc}"
          f"（测试准确率 {summary[best_name]['best_test_acc']:.2f}%），设备: {device}")
    model = build_model(best_name, device)

    if args.image:
        demo_with_custom_image(model, device, args.image)
    else:
        demo_with_testset(model, device)


if __name__ == "__main__":
    main()
