"""训练一个线性模型（--hidden 0）或单隐藏层 MLP，输入均为 50 个节点特征。"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "datasets/632d74d4e2843a53167ee9a1-momodel/DGraph/raw/dgraph.npz"


def load_inputs(path=RAW):
    with np.load(path) as raw:
        src, dst = raw["edge_index"].T
        kind, day = raw["edge_type"], raw["edge_timestamp"]
        y = torch.from_numpy(raw["y"]).float()
        train, valid = raw["train_mask"], raw["valid_mask"]
        values = np.empty((len(y), 50), dtype=np.float32)
        values[:, :20] = raw["x"]

    # 每个方向 15 列：总联系数、11 类联系数、首次/最近/平均联系日期。
    for nodes, start in ((src, 20), (dst, 35)):
        count = np.bincount(nodes, minlength=len(y))
        values[:, start] = np.log1p(count)
        for k in range(1, 12):
            values[:, start + k] = np.log1p(np.bincount(nodes[kind == k], minlength=len(y)))
        first = np.full(len(y), day.max() + 1)
        last = np.zeros(len(y), dtype=day.dtype)
        np.minimum.at(first, nodes, day)
        np.maximum.at(last, nodes, day)
        values[:, start + 12] = np.where(count > 0, first, 0)
        values[:, start + 13] = last
        values[:, start + 14] = np.bincount(nodes, weights=day, minlength=len(y)) / np.maximum(count, 1)

    x = torch.from_numpy(values)
    deviation, mean = torch.std_mean(x[train], dim=0, unbiased=False)
    scale = deviation.clamp_min(1e-6)
    x.sub_(mean).div_(scale).clamp_(-8, 8)
    return x, y, torch.from_numpy(train), torch.from_numpy(valid), mean, scale


def make_model(inputs, hidden):
    if hidden == 0:
        return nn.Linear(inputs, 1)
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def train_weights(data, hidden, lr, epochs, seed, epoch_caps=(), lr_drop_after=None, model_factory=make_model):
    x, y, train, valid, _, _ = data
    torch.manual_seed(seed)
    order = torch.Generator().manual_seed(seed)
    model = model_factory(x.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tune = valid[::2]
    tune_y = y[tune].numpy()
    best_auc = -1.0
    at_cap = {}
    print(f"{type(model).__name__}  size={hidden}  lr={lr}  train={len(train):,}", flush=True)
    for epoch in range(1, epochs + 1):
        if lr_drop_after is not None and epoch == lr_drop_after + 1:
            for group in optimizer.param_groups:
                group["lr"] = lr / 4
        model.train()
        for batch in train[torch.randperm(len(train), generator=order)].split(4096):
            optimizer.zero_grad()
            logits = model(x[batch]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(tune_y, model(x[tune]).squeeze(1).numpy())
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
        if epoch in epoch_caps:
            at_cap[epoch] = dict(best_epoch=best_epoch, tuning_auc=float(best_auc), state_dict=best_state)
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: 选轮验证 AUC {auc:.5f}，最佳 {best_auc:.5f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, float(best_auc), at_cap


def save_run(data, model, hidden, lr, epochs, seed, best_epoch, best_auc, lr_drop_after=None):
    x, y, _, valid, mean, scale = data
    with torch.inference_mode():
        probability = torch.cat([model(batch).flatten().sigmoid() for batch in x.split(16384)])
    all_auc = roc_auc_score(y[valid], probability[valid])
    check = valid[1::2]
    check_auc = roc_auc_score(y[check], probability[check])
    metrics = dict(hidden=hidden, lr=lr, epochs=epochs, seed=seed, lr_drop_after=lr_drop_after, best_epoch=best_epoch,
                   tuning_auc=float(best_auc), other_half_auc=float(check_auc), valid_auc=float(all_auc))
    folder = ROOT / "results/simple_runs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"h{hidden}_lr{lr}_s{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}.pt"
    torch.save(dict(state_dict=model.state_dict(), mean=mean, scale=scale, metrics=metrics,
                    positive_probability=probability), path)
    print(f"最佳轮 {best_epoch}，完整验证 AUC {all_auc:.5f}，另一半 {check_auc:.5f}\n{path}", flush=True)
    return path, metrics


def fit(data, hidden, lr, epochs, seed, lr_drop_after=None):
    model, best_epoch, best_auc, _ = train_weights(data, hidden, lr, epochs, seed, lr_drop_after=lr_drop_after)
    return save_run(data, model, hidden, lr, epochs, seed, best_epoch, best_auc, lr_drop_after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=64, help="隐藏单元数，默认 64，0 表示线性模型")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr-drop-after", type=int, help="可选：完成指定轮数后把学习率降为初值的 1/4")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--publish", action="store_true", help="把本次最佳权重另存为 results/simple_model.pt，供预测代码使用")
    args = parser.parse_args()
    if args.hidden < 0 or args.epochs < 1 or not 0 < args.lr < float("inf"):
        parser.error("hidden 需非负，epochs 和 lr 需为正数")
    if args.lr_drop_after is not None and not 0 < args.lr_drop_after < args.epochs:
        parser.error("--lr-drop-after 应介于 1 与 epochs - 1 之间")
    torch.set_num_threads(3)
    path, _ = fit(load_inputs(), args.hidden, args.lr, args.epochs, args.seed, args.lr_drop_after)
    if args.publish:
        target = ROOT / "results/simple_model.pt"
        shutil.copyfile(path, target)
        print(f"Mo 模型：{target}")
