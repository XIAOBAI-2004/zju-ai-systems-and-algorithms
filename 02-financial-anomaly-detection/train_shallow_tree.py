"""用 MLP 的同一组 50 维输入训练单个浅层提升树模型，并打包 Mo 所需的 CPU 预测结果。"""

import argparse
import json
from datetime import datetime

import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from train_simple import ROOT, load_inputs


def train(depth, epochs, seed, engine="catboost"):
    x, y, train_ids, valid, mean, scale = load_inputs()
    tune, check = valid[::2], valid[1::2]
    if engine == "catboost":
        model = CatBoostClassifier(
            depth=depth, iterations=epochs, learning_rate=0.07, l2_leaf_reg=5,
            loss_function="Logloss", eval_metric="AUC", border_count=128,
            thread_count=3, random_seed=seed, allow_writing_files=False, verbose=200,
        )
        model.fit(x[train_ids].numpy(), y[train_ids].numpy(),
                  eval_set=(x[tune].numpy(), y[tune].numpy()),
                  early_stopping_rounds=100, use_best_model=True)
        tree_count = model.tree_count_
    else:
        import lightgbm as lgb

        fit = lgb.Dataset(x[train_ids].numpy(), label=y[train_ids].numpy())
        stop = lgb.Dataset(x[tune].numpy(), label=y[tune].numpy(), reference=fit)
        model = lgb.train(dict(objective="binary", metric="auc", max_depth=depth,
                               num_leaves=min(63, 2**depth), learning_rate=0.07, max_bin=128,
                               min_data_in_leaf=160, lambda_l2=5, num_threads=3,
                               force_col_wise=True, seed=seed, verbosity=-1),
                          fit, num_boost_round=epochs, valid_sets=[stop],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        tree_count = model.best_iteration

    out = ROOT / "results/shallow_tree" / f"{engine}_d{depth}_s{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True)
    native = out / ("model.cbm" if engine == "catboost" else "model.lgb")
    model.save_model(str(native))
    probability = torch.empty(len(y))
    for start in range(0, len(y), 65536):
        block = x[start:start + 65536].numpy()
        p = model.predict_proba(block, thread_count=3)[:, 1] if engine == "catboost" else model.predict(block, num_threads=3)
        probability[start:start + len(block)] = torch.from_numpy(p)
    metrics = dict(engine=engine, depth=depth, seed=seed, tree_limit=epochs, trees=tree_count, features=x.shape[1],
                   tuning_auc=float(roc_auc_score(y[tune], probability[tune])),
                   other_half_auc=float(roc_auc_score(y[check], probability[check])),
                   valid_auc=float(roc_auc_score(y[valid], probability[valid])))
    target = out / "simple_model.pt"
    payload = dict(positive_probability=probability, metrics=metrics, mean=mean, scale=scale)
    payload[engine + "_weights"] = torch.frombuffer(bytearray(native.read_bytes()), dtype=torch.uint8)
    torch.save(payload, target)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"完整验证 AUC {metrics['valid_auc']:.5f}，复核半区 {metrics['other_half_auc']:.5f}\n{target}")
    return target, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--engine", choices=("catboost", "lightgbm"), default="catboost")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.depth <= 16 or args.trees < 1:
        parser.error("depth 应为 1 到 16，trees 需为正数")
    torch.set_num_threads(3)
    train(args.depth, args.trees, args.seed, args.engine)
