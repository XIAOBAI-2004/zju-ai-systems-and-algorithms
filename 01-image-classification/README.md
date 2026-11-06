# 26类垃圾分类

这个包对应 `course_all_aug.ckpt`。原作业的 Mo 系统测试两次得到 96.15 分（125/130）。在课程公开 `val` 上用同一套评分预处理得到 124/130。两组数字来自不同的测试。

## 从课程原题开始

这道作业包含两个视觉实验。第一次做，建议先读题面、跑通手写数字识别，再看垃圾分类的原始模板，最后对照本目录中的参考实现。

| 内容 | 入口 |
| --- | --- |
| 课程题目、环境与数据准备 | [原题导读](course_original/README.md) · [完整题面](course_original/assignment.txt) · [Mo 题面](https://mo.zju.edu.cn/competition/detail?type=practice&paperId=6ae994799de7db84111b29e1&courseId=6aa55443ace337cd9b662556&sectionId=6ae00c84b9499f8f910a8042) |
| 实验一 · 手写数字识别 | [原版 LeNet5 Notebook](course_original/lenet_mnist.ipynb) |
| 实验二 · 26 类垃圾分类 | [原版 MobileNetV2 Notebook](course_original/main.ipynb) · [26 类标签对照](course_original/labels.md) |
| 评分与参考实现 | [原版平台测试指南](course_original/testing_guide.ipynb) · [最终评分入口](main.py) · [最终分类头训练](train_final.py) |

`course_original/` 收录课程发放的初始模板，同级的 `main.py` 和 `train_final.py` 是垃圾分类的参考实现。只体验最终模型时可以直接从下面的预测示例开始。

## 运行一次预测

在仓库根目录打开终端。本机验证环境为 Python 3.11、MindSpore 2.7.1、Apple Silicon CPU。

```bash
cd 01-image-classification
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'from PIL import Image; import numpy as np; from main import predict; print(predict(np.array(Image.open("example/eggshell.jpg").convert("RGB"))))'
```

示例预期输出 `Eggshell`。可将命令中的图片路径换成自己的照片。`predict(image)` 接收一张形状为 `(H, W, 3)` 的 RGB 数组，返回英文类别名。

## Mo 评分使用的文件

| 路径 | 用途 |
| --- | --- |
| `main.py` | 加载权重，直接缩放到 224 × 224、按 ImageNet 均值和标准差归一化，返回最高分类得分对应的标签 |
| `src_mindspore/mobilenetv2.py` | 课程 MobileNetV2 网络定义。目录中附有源文件的 Apache 2.0 许可文本 |
| `results/course_all_aug.ckpt` | 完整的最终网络权重 |

本目录的 `main.py` 从原 Notebook 第 5 节的两个评分单元格直接导出，指向最终权重。按上表保留相对位置，在 `01-image-classification/` 中运行即可。

## 阅读和复训最终分类头

1. 从 `training/augment.py` 看同一张训练图如何转换成八个版本。原图、翻转、两种裁剪、仿射变换、光照变化、压缩和红蓝通道交换都保留原标签。
2. 再看 `train_final.py`。函数 `extract` 用冻结的骨干提取 1,280 维特征，`fit_classes` 学习全部 26 类的权重。旧头与新头校准后按 30% 和 70% 合并。最后 `correct_eggs` 用真实碎壳特征和课程训练图校正蛋壳这一行。
3. 要复训，请从课程工作区取得 `garbage_26x100` 数据集。在 `01-image-classification/` 中运行下列命令，将最后一个参数替换为自己的实际路径。

```bash
python train_final.py "/你的数据目录/garbage_26x100"
```

脚本只读取 `garbage_26x100/train/` 的 26 个类别目录，课程完整训练集为 2,343 张。结果写到 `results/retrained.ckpt`。

复训从之前的 `garbage_tuned_best.ckpt` 所对应的骨干和旧分类头出发。它与最终权重的骨干逐项相同，所以包只存一份完整模型。复训时从最终文件取骨干，从 `training/base_head.npy` 取上一版分类头。额外 45 张碎壳训练照片随包提供预提取特征（11 种视图），可离线参与拟合。`example/eggshell.jpg` 来自独立留出的照片。摄影者、原图和许可见 [图片来源](training/image_sources.md)。

本机从全部课程训练图复训后，分类头与随包权重的最大绝对差约为 1.12 × 10⁻⁸，在公开验证集上的 130 个预测逐张一致。
