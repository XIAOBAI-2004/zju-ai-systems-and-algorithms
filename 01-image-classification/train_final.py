"""按入选配置复训 course_all_aug 的分类头。课程 train 与公开 val 分开使用。"""

import argparse
from pathlib import Path

import mindspore as ms
import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.special import expit

from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2
from training.augment import VIEWS, image_view

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "training"
FINAL = ROOT / "results" / "course_all_aug.ckpt"
CLASS_DIRS = [f"{group:02d}_{label:02d}" for group, count in enumerate((10, 8, 4, 4))
              for label in range(count)]
EGG = CLASS_DIRS.index("03_02")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def extract(backbone, dataset):
    files = sorted((dataset / "train").glob("*/*.jpg"))
    if {p.parent.name for p in files} != set(CLASS_DIRS):
        raise ValueError(f"请传入含 train/ 的 garbage_26x100 目录，当前找到 {len(files)} 张图片。")
    labels = np.array([CLASS_DIRS.index(p.parent.name) for p in files], dtype=np.int32)
    prefix = backbone.features[:-2]
    tail = mobilenet_v2(backbone.features[-2], backbone.features[-1])
    prefix.set_train(False)
    tail.set_train(False)
    views = []
    for view in VIEWS:
        pooled = []
        for start in range(0, len(files), 24):
            batch = []
            for i, path in enumerate(files[start:start + 24], start):
                with Image.open(path) as im:
                    batch.append(image_view(np.array(im.convert("RGB")), view, i))
            maps = prefix(ms.Tensor(np.stack(batch)))
            if view in VIEWS[:4]:
                # 原训练把这四种中间特征以 float16 缓存，再用 float32 计算最后两块。
                maps = ms.Tensor(maps.asnumpy().astype(np.float16).astype(np.float32))
            pooled.append(tail(maps).mean(axis=(2, 3)).asnumpy())
        views.append(np.concatenate(pooled))
        print(f"{view}: 已提取 {len(files)} 张图的特征", flush=True)
    return views, labels


def calibrated(w, original):
    scores = unit(original) @ w.T
    top_two = np.partition(scores, -2, axis=1)[:, -2:]
    scale = 3.0 / max(float(np.median(top_two[:, 1] - top_two[:, 0])), 1e-6)
    return (w - w.mean(axis=0)) * scale, scale


def fit_classes(views, labels):
    x = unit(np.concatenate(views))
    y = np.tile(labels, len(views))
    targets = np.eye(26, dtype=np.float32)[y]
    n, channels = x.shape
    reg = 3e-5

    def objective(flat):
        w = flat.reshape(channels, 26).astype(np.float32)
        logits = x @ w
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        denom = exp.sum(axis=1, keepdims=True)
        loss = (np.log(denom[:, 0]) - logits[np.arange(n), y]).mean()
        grad = x.T @ (exp / denom - targets) / n + reg * w
        return float(loss + 0.5 * reg * (w * w).sum()), grad.ravel().astype(np.float64)

    solution = minimize(objective, np.zeros(channels * 26), jac=True, method="L-BFGS-B",
                        options={"maxiter": 80, "ftol": 1e-9})
    return solution.x.reshape(channels, 26).T.astype(np.float32)


def correct_eggs(head, views, labels):
    with np.load(ASSETS / "eggshell_features.npz") as data:
        shell_views, groups = data["features"], data["groups"]
    original = np.concatenate([unit(v) for v in views])
    shells = np.concatenate([unit(v) for v in shell_views])
    x = np.concatenate([original, shells])
    y = np.concatenate([np.tile(labels == EGG, len(views)), np.ones(len(shells))]).astype(np.float32)
    # 额外照片按摄影者均衡。同一个人的连拍分摊一份权重。
    _, group_ids, counts = np.unique(groups, return_inverse=True, return_counts=True)
    shell_weights = np.tile(1 / counts[group_ids], len(shell_views)) / len(shell_views)
    weights = np.concatenate([np.full(len(original), 1 / len(views)), 8.0 * shell_weights])
    weights = (weights / weights.sum()).astype(np.float32)
    scores = x @ head.T
    margin = scores[:, EGG] - np.delete(scores, EGG, axis=1).max(axis=1)
    reg = 2e-4

    def objective(flat):
        delta = flat.astype(np.float32)
        logits = margin + x @ delta
        value = np.sum(weights * np.logaddexp(0.0, (1 - 2 * y) * logits))
        grad = x.T @ (weights * (expit(logits) - y)) + reg * delta
        return float(value + 0.5 * reg * (delta @ delta)), grad.astype(np.float64)

    solution = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                        options={"maxiter": 100, "gtol": 1e-7, "ftol": 1e-11})
    result = head.copy()
    result[EGG] += solution.x.astype(np.float32)
    return result


def train(dataset):
    ms.set_context(mode=ms.GRAPH_MODE)
    parameters = ms.load_checkpoint(str(FINAL))
    backbone = MobileNetV2Backbone()
    backbone_params = {k: v for k, v in parameters.items() if k != "dense.weight"}
    missing, extra = ms.load_param_into_net(backbone, backbone_params, strict_load=True)
    if missing or extra:
        raise ValueError(f"骨干权重不匹配: {missing}, {extra}")
    backbone.set_train(False)
    views, labels = extract(backbone, dataset)
    old = np.load(ASSETS / "base_head.npy")
    old_head, scale = calibrated(old, views[0])
    new_head, _ = calibrated(fit_classes(views, labels), views[0])
    combined = 0.3 * old_head + 0.7 * new_head
    w = correct_eggs(combined, views, labels) / scale + old.mean(axis=0)
    parameters["dense.weight"].set_data(ms.Tensor(np.ascontiguousarray(w)))
    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="课程 garbage_26x100 目录")
    args = parser.parse_args()
    weights = train(args.dataset)
    output = ROOT / "results" / "retrained.ckpt"
    ms.save_checkpoint([{"name": k, "data": v} for k, v in weights.items()], str(output))
    print(f"复训结果已保存到 {output}")
