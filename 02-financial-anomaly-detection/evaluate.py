"""用已训练的权重复算验证 AUC，也可为 Mo 生成五模型预测缓存。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from train import DATA_DIR, ROOT, MLP
from train_fm import BinnedFM
from train_simple import load_inputs, make_model

MODEL_NAMES = ("mlp20", "linear50", "mlp50", "fm50", "catboost50", "lightgbm50", "ensemble")


def load_weights(name):
    return torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu", weights_only=True)


def load_tree(engine, weights):
    if engine == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier().load_model(blob=weights.numpy().tobytes())
    import lightgbm

    return lightgbm.Booster(model_str=weights.numpy().tobytes().decode("utf-8"))


def mlp20(raw, ids):
    model = MLP().eval()
    model.load_state_dict(load_weights("mlp20"))
    x = torch.from_numpy(raw["x"]).float()
    x = (x[ids] - x.mean(0)) / x.std(0).clamp_min(1e-12)
    return model(x).exp()[:, 1].numpy()


def predict50(name, x):
    saved = load_weights(name)
    metrics = saved["metrics"]
    if name in ("catboost50", "lightgbm50"):
        engine = name.removesuffix("50")
        model = load_tree(engine, saved[f"{engine}_weights"])
        if engine == "catboost":
            return model.predict_proba(x.numpy(), thread_count=3)[:, 1]
        return model.predict(x.numpy(), num_threads=3)
    if name == "fm50":
        model = BinnedFM(x.shape[1], metrics["rank"], metrics["bins"])
        values = x.numpy()
        x = torch.from_numpy(np.column_stack([
            np.searchsorted(edges.numpy(), values[:, i])
            for i, edges in enumerate(saved["bin_edges"])
        ]))
    else:
        model = make_model(x.shape[1], metrics["hidden"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return torch.cat([model(block).flatten().sigmoid() for block in x.split(16384)]).numpy()


def ensemble(data_dir, ids, saved):
    from utils.boost_graph import build_features, matrix

    recipe = json.loads((ROOT / "results/optimized_recipe.json").read_text())
    logits = []
    for index, (member, config) in enumerate(zip(saved["members"], recipe["models"]), start=1):
        engine = config["engine"]
        print(f"计算五模型组合成员 {index}/{len(saved['members'])}（{engine}）", flush=True)
        feature_config = {key: value for key, value in config.items()
                          if key not in ("engine", "negative_ratio", "depth", "iterations")}
        features, train_ids, _, labels = build_features(data_dir, **feature_config)
        model = load_tree(engine, member["weights"])
        scores = np.empty(len(ids), dtype=np.float32)
        for start in range(0, len(ids), 60000):
            batch = ids[start:start + 60000]
            x = matrix(features, batch)
            if engine == "catboost":
                prediction = model.predict(x, prediction_type="RawFormulaVal", thread_count=3)
            else:
                prediction = model.predict(x, raw_score=True, num_threads=3)
            scores[start:start + len(batch)] = prediction
        del features
        negatives = int((labels[train_ids] == 0).sum())
        kept = min(negatives, config["negative_ratio"] * (len(train_ids) - negatives))
        probability = torch.sigmoid(torch.from_numpy(scores) + np.log(kept / negatives))
        logits.append(torch.logit(probability, eps=1e-7) * member["mix_weight"])
    return torch.stack(logits).sum(0).sigmoid()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="包含 DGraph/raw/dgraph.npz 的目录")
    parser.add_argument("--export-mo", action="store_true", help="额外生成五模型的全图预测缓存")
    args = parser.parse_args()
    if args.export_mo and args.model not in ("ensemble", "all"):
        parser.error("--export-mo 需要 --model ensemble 或 --model all")
    torch.set_num_threads(3)
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    raw_path = args.data_dir / "DGraph/raw/dgraph.npz"
    if not raw_path.is_file():
        parser.error(f"找不到 {raw_path}，请按 README 从课程工作区取得数据")
    with np.load(raw_path) as raw:
        valid = raw["valid_mask"]
        labels = raw["y"].reshape(-1)
        if "mlp20" in models:
            print(f"mlp20: {roc_auc_score(labels[valid], mlp20(raw, valid)):.5f}", flush=True)
    fixed = [name for name in models if name.endswith("50")]
    if fixed:
        all_x, *_ = load_inputs(raw_path)
        x = all_x[valid]
        del all_x
        for name in fixed:
            print(f"{name}: {roc_auc_score(labels[valid], predict50(name, x)):.5f}", flush=True)
        del x
    if "ensemble" in models:
        saved = load_weights("ensemble")
        ids = np.arange(len(labels)) if args.export_mo else valid
        prediction = ensemble(args.data_dir, ids, saved)
        print(f"ensemble: {roc_auc_score(labels[valid], prediction[valid] if args.export_mo else prediction):.5f}")
        if args.export_mo:
            path = ROOT / "results/optimized_model.pt"
            torch.save(dict(saved, positive_probability=prediction), path)
            print(f"Mo 模型：{path}")


if __name__ == "__main__":
    main()
