"""金融异常检测训练入口。--score-best 复算最佳模型分数，--rebuild-best 从原始图重训。"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from utils import DGraphFin
from utils.evaluator import Evaluator


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "632d74d4e2843a53167ee9a1-momodel"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 层名与原始 MLP 的权重键一致。
        self.lins = nn.ModuleList([nn.Linear(20, 128), nn.Linear(128, 2)])

    def forward(self, x):
        return F.log_softmax(self.lins[1](F.relu(self.lins[0](x))), dim=-1)


def load_data():
    data = DGraphFin(root=str(DATA_DIR), name="DGraph")[0]
    # 沿用原题的全图特征标准化和节点划分。
    mean, std = data.x.mean(0), data.x.std(0)
    data.x = (data.x - mean) / std.clamp_min(1e-12)
    data.y = data.y.flatten()
    return data


def load_splits(device):
    data = load_data()
    x, y = data.x, data.y
    train, valid = data.train_mask, data.valid_mask
    return x[train].to(device), y[train].to(device), x[valid].to(device), y[valid].to(device)


def train(epochs, lr, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, train_y, valid_x, valid_y = load_splits(device)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-7)
    evaluator = Evaluator("auc")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    run_name = datetime.now().strftime("mlp_%Y%m%d_%H%M%S_%f")
    best_auc, best_epoch, best_path = -1.0, 0, None

    print(f"设备 {device} | 训练节点 {len(train_y):,} | 验证节点 {len(valid_y):,}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            probabilities = model(valid_x).exp()
            valid_auc = evaluator.eval(valid_y, probabilities)["auc"]

        if valid_auc > best_auc:
            path = results / f"{run_name}_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), path)
            if best_path is not None:
                best_path.unlink()
            best_auc, best_epoch, best_path = valid_auc, epoch, path
        print(f"Epoch {epoch:3d}/{epochs} | Loss {loss.item():.4f} | "
              f"Valid AUC {valid_auc:.5f} | Best {best_auc:.5f}", flush=True)

    print(f"最佳验证 ROC-AUC {best_auc:.5f}（第 {best_epoch} 轮）\n权重：{best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在 DGraph 上训练、复现和评估金融异常检测模型。")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--boost", action="store_true", help="训练使用关系类型、时间和邻居特征的提升树")
    modes.add_argument("--export-trees", nargs="+", type=Path, metavar="运行目录", help="打包选定的树模型供 Mo 使用")
    modes.add_argument("--rebuild-best", action="store_true", help="按已选配方从原始数据重新训练并打包最佳模型")
    modes.add_argument("--score-best", action="store_true", help="只复算已打包模型的本地验证 ROC-AUC")
    parser.add_argument("--mix-weights", nargs="+", type=float, help="打包树模型时的组合权重，默认等权")
    parser.add_argument("--known-neighbors", action="store_true", help="在提升树中加入训练标签的交叉拟合邻居统计")
    parser.add_argument("--event-features", action="store_true", help="加入联系间隔、联系发生时对方的活跃时长等统计")
    parser.add_argument("--peer-activity", action="store_true", help="加入共同联系人在相近日期的活动和训练邻居统计")
    parser.add_argument("--two-hop", action="store_true", help="加入共享联系人的其他用户特征")
    parser.add_argument("--cohort-risk", action="store_true", help="统计相似日期、相似联系数量的训练用户的折外欺诈率")
    parser.add_argument("--typed-labels", action="store_true", help="按关系类型统计折外训练邻居的欺诈率")
    parser.add_argument("--granular-cohorts", action="store_true", help="在相近活动日期内进一步按用户特征分组")
    parser.add_argument("--relation-profiles", action="store_true", help="按关系类型统计邻居资料与联系时期风险")
    parser.add_argument("--engine", choices=("catboost", "lightgbm"), default="catboost", help="提升树实现")
    parser.add_argument("--depth", type=int, default=6, help="提升树深度，默认 6")
    parser.add_argument("--trees", type=int, default=850, help="提升树最大棵数，默认 850")
    parser.add_argument("--negative-ratio", type=int, default=28, help="每个训练正例最多抽取的负例数")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数，默认 200")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam 学习率，默认 0.01")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.lr < float("inf"):
        parser.error("--epochs 和 --lr 需要为正数，--lr 需要有限")
    if any((args.known_neighbors, args.event_features, args.peer_activity, args.two_hop,
            args.cohort_risk, args.typed_labels, args.granular_cohorts, args.relation_profiles)) and not args.boost:
        parser.error("图事件和训练邻居特征用于 --boost")
    if args.mix_weights and not args.export_trees:
        parser.error("--mix-weights 用于 --export-trees")
    if args.score_best:
        from sklearn.metrics import roc_auc_score
        import numpy as np

        saved = torch.load(ROOT / "results/optimized_model.pt", map_location="cpu", weights_only=True)
        with np.load(DATA_DIR / "DGraph/raw/dgraph.npz") as raw:
            ids = raw["valid_mask"]
            auc = roc_auc_score(raw["y"][ids], saved["positive_probability"][ids].numpy())
        print(f"验证节点 {len(ids):,} | ROC-AUC {auc:.7f}")
    elif args.rebuild_best:
        from utils.boost_graph import rebuild_best
        rebuild_best(DATA_DIR, ROOT / "results")
    elif args.export_trees:
        from utils.boost_graph import export_boosted
        export_boosted(args.export_trees, DATA_DIR, ROOT / "results", args.mix_weights)
    elif args.boost:
        if not 1 <= args.depth <= 16 or min(args.trees, args.negative_ratio) < 1:
            parser.error("--depth 应在 1 到 16 之间，--trees 和 --negative-ratio 需要为正数")
        from utils.boost_graph import train_boosted
        train_boosted(DATA_DIR, ROOT / "results", args.seed, args.negative_ratio, args.depth,
                      args.trees, use_labels=args.known_neighbors, detailed_events=args.event_features,
                      use_peers=args.peer_activity, engine=args.engine, two_hop=args.two_hop,
                      use_cohorts=args.cohort_risk, typed_labels=args.typed_labels,
                      granular=args.granular_cohorts, profiles=args.relation_profiles)
    else:
        train(args.epochs, args.lr, args.seed)
