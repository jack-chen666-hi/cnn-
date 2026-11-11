# -*- coding: utf-8 -*-
"""
训练入口脚本
============
用法：
  python train.py --model all       # 依次训练全部 4 个模型
  python train.py --model baseline  # 只训练指定模型（baseline/dw/se/gelu_gn）
  python train.py --model all --epochs 30

每个模型使用完全相同的超参数、数据划分和随机种子，保证对比公平。
运行后产出：
  checkpoints/{model}.pth        —— 测试集准确率最高的权重；
  results/logs/{model}_log.csv   —— 每轮损失/准确率/耗时；
  results/summary.json           —— 参数量、MACs、最佳准确率等汇总。
"""
import argparse
import csv
import json
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (BATCH_SIZE, CHECKPOINT_DIR, EPOCHS, LEARNING_RATE,
                    LOG_DIR, MOMENTUM, SEED, SUMMARY_PATH, WEIGHT_DECAY)
from data_utils import get_dataloaders, set_seed
from metrics import count_macs, count_parameters
from models import MODEL_ORDER, MODEL_REGISTRY


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回 (平均损失, 训练准确率%)。"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在测试集上评估，返回 (平均损失, 测试准确率%)。"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, 100.0 * correct / total


def load_summary():
    """读取已有汇总（支持分模型多次运行后追加）。"""
    try:
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_summary(summary):
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def train_model(model_name, epochs, batch_size, lr, device):
    """完整训练一个模型并返回结果字典。"""
    print("\n" + "=" * 72)
    print(f"开始训练模型: {MODEL_REGISTRY[model_name].model_desc}（标识: {model_name}）")
    print("=" * 72)

    # 每个模型训练前都重置随机种子，保证初始化和数据打乱顺序一致
    set_seed(SEED)
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = MODEL_REGISTRY[model_name](num_classes=10).to(device)
    params = count_parameters(model)
    macs = count_macs(model, input_size=(1, 3, 32, 32))
    print(f"可训练参数量: {params:,}（{params / 1e6:.3f} M）")
    print(f"单图 MACs: {macs:,}（{macs / 1e6:.3f} M）")

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = f"{CHECKPOINT_DIR}/{model_name}.pth"
    log_path = f"{LOG_DIR}/{model_name}_log.csv"
    best_acc, best_epoch = 0.0, 0
    log_rows = []
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:2d}/{epochs} | 训练损失 {train_loss:.4f} | "
              f"训练准确率 {train_acc:6.2f}% | 测试准确率 {test_acc:6.2f}% | "
              f"lr {current_lr:.5f} | 耗时 {epoch_time:5.1f}s")

        log_rows.append([epoch, round(train_loss, 6), round(train_acc, 4),
                         round(test_loss, 6), round(test_acc, 4),
                         round(current_lr, 6), round(epoch_time, 2)])

        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            torch.save(model.state_dict(), ckpt_path)
        scheduler.step()

    total_time = time.time() - total_start

    # 载入最佳权重重新评估，确认最终数字与保存的权重严格对应
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)

    # 写逐轮日志 CSV
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "test_loss", "test_acc", "lr", "epoch_time_s"])
        writer.writerows(log_rows)

    result = {
        "model_name": model_name,
        "model_desc": model.model_desc,
        "params": params,
        "macs": macs,
        "best_test_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "final_test_acc": round(final_acc, 4),
        "final_test_loss": round(final_loss, 6),
        "total_train_time_s": round(total_time, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
    }
    print("-" * 72)
    print(f"训练完成 | 最佳测试准确率 {best_acc:.2f}%（第 {best_epoch} 轮）| "
          f"最佳权重复评 {final_acc:.2f}% | 总耗时 {total_time:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="VGG 系列模型训练（CIFAR-10）")
    parser.add_argument("--model", default="all",
                        choices=["all"] + MODEL_ORDER, help="训练哪个模型")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("运行环境信息")
    print(f"  PyTorch 版本 : {torch.__version__}")
    print(f"  CUDA 可用    : {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"  GPU 设备     : {torch.cuda.get_device_name(0)}")
    print(f"  训练设备     : {device}")
    print("=" * 72)

    targets = MODEL_ORDER if args.model == "all" else [args.model]
    summary = load_summary()
    for name in targets:
        summary[name] = train_model(
            name, args.epochs, args.batch_size, args.lr, device)
        save_summary(summary)

    print("\n" + "=" * 72)
    print("全部实验汇总")
    print("=" * 72)
    header = f"{'模型':<28}{'参数量':>12}{'MACs(M)':>12}{'最佳准确率':>12}"
    print(header)
    for name in targets:
        r = summary[name]
        print(f"{r['model_desc']:<28}{r['params']:>12,}"
              f"{r['macs'] / 1e6:>12.3f}{r['best_test_acc']:>11.2f}%")
    print(f"\n汇总文件: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
