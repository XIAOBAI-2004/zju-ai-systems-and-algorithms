"""用 MLP 的 50 个输入训练因子分解机或分箱加性模型，rank 控制两两交互的容量。"""

import argparse
import json
from datetime import datetime
from functools import partial

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from train_simple import ROOT, load_inputs, train_weights


class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, rank):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.factors = nn.Parameter(torch.empty(input_dim, rank))
        nn.init.normal_(self.factors, std=0.03)

    def forward(self, x):
        # 等价于累加所有 i < j 的 x_i * x_j * (v_i 与 v_j 的内积)。
        cross = (x @ self.factors).square() - x.square() @ self.factors.square()
        return self.linear(x) + 0.5 * cross.sum(dim=1, keepdim=True)


def bin_features(x, train_ids, bins):
    # 每一列的边界只用训练样本拟合，重复的分位数合并。
    codes = torch.empty(x.shape, dtype=torch.uint8)
    cuts = []
    for col in range(x.shape[1]):
        v = x[:, col].numpy()
        t = v[train_ids.numpy()]
        distinct = np.unique(t)
        if len(distinct) <= bins:
            bounds = (distinct[:-1] + distinct[1:]) / 2
        else:
            bounds = np.unique(np.quantile(t, np.arange(1, bins) / bins)).astype(np.float32)
        cuts.append(torch.from_numpy(bounds))
        codes[:, col] = torch.from_numpy(np.searchsorted(bounds, v).astype(np.uint8))
    return codes, cuts


class BinnedFM(nn.Module):
    def __init__(self, input_dim, rank, bins):
        super().__init__()
        self.rank = rank
        self.bias = nn.Parameter(torch.zeros(()))
        self.table = nn.Embedding(input_dim * bins, rank + 1)
        nn.init.normal_(self.table.weight, std=0.01)
        self.register_buffer("offset", torch.arange(input_dim) * bins)

    def forward(self, x):
        terms = self.table(x.long() + self.offset)
        score = self.bias + terms[:, :, 0].sum(dim=1, keepdim=True)
        if self.rank:
            factors = terms[:, :, 1:]
            cross = factors.sum(dim=1).square() - factors.square().sum(dim=1)
            score = score + 0.5 * cross.sum(dim=1, keepdim=True)
        return score


def train(rank, lr, epochs, seed, bins=0):
    data = load_inputs()
    cuts = []
    factory = FactorizationMachine
    if bins:
        codes, cuts = bin_features(data[0], data[2], bins)
        data = (codes, *data[1:])
        factory = partial(BinnedFM, bins=bins)
    model, best_epoch, tune_auc, _ = train_weights(
        data, rank, lr, epochs, seed, model_factory=factory)
    x, y, _, valid, mean, scale = data
    with torch.no_grad():
        risk = torch.cat([model(b).flatten().sigmoid() for b in x.split(16384)])
    metrics = dict(rank=rank, bins=bins, lr=lr, epochs=epochs, seed=seed, best_epoch=best_epoch,
                   tuning_auc=tune_auc, other_half_auc=float(roc_auc_score(y[valid[1::2]], risk[valid[1::2]])),
                   valid_auc=float(roc_auc_score(y[valid], risk[valid])))
    out = ROOT / "results/fm" / f"rank{rank}_bins{bins}_s{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True)
    torch.save(dict(positive_probability=risk, state_dict=model.state_dict(), metrics=metrics,
                    mean=mean, scale=scale, bin_edges=cuts), out / "simple_model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"完整验证 AUC {metrics['valid_auc']:.5f}，复核半区 {metrics['other_half_auc']:.5f}\n{out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--bins", type=int, default=0, help="0 使用原数值；正值按训练分位数分箱，rank=0 时只拟合单列效应")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rank < 0 or args.epochs < 1 or not 0 < args.lr < float("inf"):
        parser.error("rank 需非负，epochs 和 lr 需为正数")
    if args.bins != 0 and not 2 <= args.bins <= 256:
        parser.error("--bins 应为 0 或 2 到 256 之间的整数")
    torch.set_num_threads(3)
    train(args.rank, args.lr, args.epochs, args.seed, args.bins)
