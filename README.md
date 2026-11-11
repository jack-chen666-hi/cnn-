# 基于 CIFAR-10 的 VGG 风格 CNN 设计与结构改进实验

在 CIFAR-10 图像分类数据集上实现一个 VGG 风格的经典卷积神经网络基线，并完成三个结构创新变体：
**深度可分离卷积、SE 通道注意力、GELU + GroupNorm**。四个模型在完全相同的超参数下各训练 30 轮，
公平比较参数量、计算量与分类精度。本项目所有结果均由代码真实运行得到。

- 作者：陈渤
- 学号：2403010321
- 班级：智科 1 班

## 一、项目结构

```
CNN大作业/
├── config.py            # 全局配置：路径、超参数、类别、归一化常量
├── data_utils.py        # 随机种子、本地数据集 LocalCIFAR10、DataLoader
├── metrics.py           # 参数量与 MACs 统计
├── prepare_data.py      # parquet -> npz 数据转换（只需运行一次）
├── train.py             # 训练入口
├── plot_results.py      # 训练曲线、混淆矩阵、数据集样本图
├── infer_demo.py        # 单图推理场景演示（支持外部图片）
├── requirements.txt     # 依赖清单
├── .gitignore
├── models/              # 四个模型与通用组件
│   ├── modules.py       # ConvBlock / 深度可分离卷积 / SEBlock / 激活归一化工厂
│   ├── vgg_baseline.py  # 基线：标准卷积 + BN + ReLU
│   ├── vgg_dw.py        # 创新一：深度可分离卷积
│   ├── vgg_se.py        # 创新二：SE 注意力
│   └── vgg_gelu_gn.py   # 创新三：GELU + GroupNorm
├── data/                # 数据集（不入库，按下方步骤准备）
├── checkpoints/         # 模型权重（不入库，训练后生成）
└── results/             # 日志、图表、汇总（不入库的部分见 .gitignore）
    ├── summary.json
    ├── logs/            # 每个模型的逐轮 CSV
    ├── figures/         # 生成的图片
    └── sample_images/   # 推理演示导出的样本图
```

## 二、运行环境

- 操作系统：Windows 10/11（本项目在 Windows + PyCharm 下验证）
- Python：3.10 及以上（开发使用 3.13）
- 硬件：建议带 NVIDIA 显卡（开发环境为 RTX 4060 Laptop，8 GB 显存）；无独显时可使用 CPU，速度较慢
- 关键依赖：PyTorch 2.11（CUDA 12.8 版）、torchvision、numpy、pandas、pyarrow、matplotlib、pillow、python-docx

## 三、环境搭建

1. 进入项目目录，创建并激活虚拟环境（PyCharm 中也可直接在 Settings 里创建解释器）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 安装依赖（使用国内镜像，PyTorch 的 CUDA 版本会自动从阿里云镜像获取）：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 验证 GPU 是否可用，应输出 `True` 与显卡名称：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

> 若使用纯 CPU 环境，可把 requirements.txt 中的 `torch`、`torchvision` 改为官方普通版本，
> 代码会自动选择 CPU 设备，无需修改任何源码。

## 四、数据准备

CIFAR-10 共 50000 张训练图、10000 张测试图，全部为 32×32 彩色图像。由于官方源在国内访问不稳定，
本项目从 HuggingFace 镜像下载 parquet 文件后在本地解码。

1. 在项目目录下创建 `data` 文件夹，下载以下两个文件并放入其中（整个数据集约 140 MB）：

```text
https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
```

2. 把两个文件分别重命名为 `train.parquet` 和 `test.parquet`，最终路径为：

```text
data/train.parquet
data/test.parquet
```

3. 运行数据准备脚本，把 parquet 解码为训练脚本直接读取的 npz：

```powershell
python prepare_data.py
```

成功后会在 `data/cifar10_prepared/` 下生成 `train.npz`、`test.npz`，并打印出
训练集 50000 张、测试集 10000 张及每类 5000/1000 的均衡分布。

## 五、模型训练

训练全部四个模型（每个 30 轮，使用 GPU 时整体约 25 分钟）：

```powershell
python train.py --model all --epochs 30
```

也可只训练某一个模型，可选标识为 `baseline`、`dw`、`se`、`gelu_gn`：

```powershell
python train.py --model baseline --epochs 30
```

其他可选参数：`--batch-size`（默认 128）、`--lr`（默认 0.05）。
训练完成后产出：

- `checkpoints/{model}.pth`：测试准确率最高的权重；
- `results/logs/{model}_log.csv`：每轮损失、准确率、学习率、耗时；
- `results/summary.json`：参数量、MACs、最佳准确率等汇总。

## 六、结果可视化

```powershell
python plot_results.py
```

生成三张图（保存在 `results/figures/`）：

- `training_curves.png`：四个模型的训练损失与测试准确率对比曲线；
- `confusion_matrix.png`：最佳模型在 10000 张测试图上的混淆矩阵；
- `dataset_samples.png`：CIFAR-10 训练集样本示例。

## 七、单图推理演示（场景应用）

自动从训练集导出 8 张不同类别图片，加载最佳模型逐一识别并生成标注图：

```powershell
python infer_demo.py
```

识别自己的任意外部图片（自动中心裁剪并缩放到 32×32，输出 Top-3 预测）：

```powershell
python infer_demo.py --image 你的图片.jpg
```

## 八、实验结果

四个模型在统一设置（seed=42，30 epochs，batch=128，SGD lr=0.05，余弦退火）下的真实结果：

| 模型 | 参数量 | MACs(M) | 最佳准确率 | 最佳轮次 | 训练时间(s) |
|---|---:|---:|---:|---:|---:|
| VGG 基线（标准卷积+BN+ReLU） | 4,692,426 | 209.393 | 92.23% | 30 | 369.17 |
| 深度可分离卷积版（DWConv） | 546,378 | 26.207 | 90.19% | 30 | 286.06 |
| SE 注意力版（SE-Net） | 4,735,946 | 209.436 | 92.63% | 27 | 407.89 |
| GELU + GroupNorm 版 | 4,692,426 | 209.393 | 92.38% | 30 | 388.93 |

主要结论：

- **深度可分离卷积**：参数量减少 88.36%、MACs 减少 87.48%，准确率仅下降 2.04 个百分点，轻量化收益极高，适合边缘部署；
- **SE 注意力**：参数仅增加 0.93%、MACs 几乎不变，准确率提升 0.40 个百分点，取得最高的 92.63%；
- **GELU + GroupNorm**：模型规模不变，准确率提升 0.15 个百分点，训练稳定。

## 九、把代码上传到 GitHub / Gitee 仓库

### 1. 安装 Git

本机若尚未安装 Git，先到官网 <https://git-scm.com/download/win> 下载安装（一路默认下一步即可），
或在 PowerShell 中用 winget 安装：

```powershell
winget install --id Git.Git -e --source winget
```

安装后重新打开终端，运行 `git --version` 能看到版本号即成功。

### 2. 在网页上创建空仓库

- Gitee（国内访问快，推荐）：登录 <https://gitee.com>，右上角“+”→“新建仓库”，填写仓库名（如 `cnn-cifar10`），
  **不要**勾选“初始化仓库”，创建后复制仓库地址（形如 `https://gitee.com/用户名/cnn-cifar10.git`）。
- GitHub：登录 <https://github.com>，右上角“+”→“New repository”，填写仓库名，不勾选 README，
  创建后复制地址（形如 `https://github.com/用户名/cnn-cifar10.git`）。

### 3. 本地提交并推送

在项目根目录（`CNN大作业`）下依次执行：

```powershell
git init
git add .
git commit -m "CNN大作业：VGG基线与三个结构改进变体"
git branch -M main
git remote add origin 你复制的仓库地址
git push -u origin main
```

首次推送会弹出登录窗口，按提示登录 Gitee/GitHub 即可。`.gitignore` 已配置忽略
`data/`、`checkpoints/`、`.venv/`、`.idea/` 等大体积或本地文件，仓库中只保留源码与说明，
数据和权重按 README 步骤重新生成。

### 4. 后续修改的更新方式

以后代码有改动，只需在项目目录执行：

```powershell
git add .
git commit -m "说明本次修改内容"
git push
```

## 十、生成 Word 报告（可选复现）

报告 `CNN大作业报告_陈渤_2403010321.docx` 已随项目提供。如需重新生成，确保已安装 python-docx 后运行
报告生成脚本即可，脚本会直接读取 `summary.json` 与真实日志，保证报告数字与代码输出一致。
