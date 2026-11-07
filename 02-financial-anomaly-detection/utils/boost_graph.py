"""利用 DGraph 的节点、关系类型、时间和邻居信息训练梯度提升树。"""

import json
from datetime import datetime

import numpy as np
import torch
from catboost import CatBoostClassifier
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score


def collect_features(raw, detailed_events=False, two_hop=False):
    x = raw["x"].astype(np.float32)
    src, dst = raw["edge_index"].T
    kind, day = raw["edge_type"], raw["edge_timestamp"]
    n = len(x)
    adjacency = csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    directions = (("out", src, adjacency), ("in", dst, adjacency.T.tocsr()))
    features = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    profiles = []
    for direction, nodes, edges in directions:
        counts = np.bincount(nodes, minlength=n)
        features[f"{direction}_degree"] = counts.astype(np.float32)
        features[f"{direction}_distinct"] = np.diff(edges.indptr).astype(np.float32)
        for k in np.unique(kind):
            num = np.bincount(nodes[kind == k], minlength=n)
            features[f"{direction}_type_{k}"] = num.astype(np.float32)

        # 日期已脱敏。没有边的节点用 NaN 表示缺失，让树自己处理。
        first = np.full(n, day.max() + 1, dtype=np.float32)
        last = np.zeros(n, dtype=np.float32)
        np.minimum.at(first, nodes, day)
        np.maximum.at(last, nodes, day)
        first[counts == 0] = np.nan
        last[counts == 0] = np.nan
        denom = np.maximum(counts, 1)
        mean = np.bincount(nodes, weights=day, minlength=n) / denom
        mean[counts == 0] = np.nan
        variance = np.bincount(nodes, weights=day.astype(float)**2, minlength=n) / denom - mean**2
        for name, values in (("first", first), ("last", last), ("span", last-first),
                             ("mean_day", mean), ("std_day", np.sqrt(np.maximum(variance, 0)))):
            features[f"{direction}_{name}"] = values.astype(np.float32, copy=False)
        # 分别统计活跃天数和一天内的最多联系次数。
        width = int(day.max()) + 1
        node_days, freq = np.unique(nodes * width + day, return_counts=True)
        active_nodes = node_days // width
        busiest = np.zeros(n, dtype=np.float32)
        np.maximum.at(busiest, active_nodes, freq)
        features[f"{direction}_active_days"] = np.bincount(active_nodes, minlength=n).astype(np.float32)
        features[f"{direction}_busiest"] = busiest
        profiles.extend((np.log1p(counts).astype(np.float32), np.nan_to_num(mean).astype(np.float32)))

    features["mutual_contacts"] = np.diff(adjacency.multiply(adjacency.T).indptr).astype(np.float32)
    features["missing_count"] = (x < -0.99).sum(axis=1, dtype=np.float32)
    # 邻居自身特征以及邻居的度数/平均日期。分组计算以控制内存。
    for direction, _, edges in directions:
        degree = features[f"{direction}_degree"]
        denom = np.maximum(degree, 1)
        for start in range(0, 17, 4):
            end = min(start + 4, 17)
            neighborhood = (edges @ x[:, start:end]) / denom[:, None]
            neighborhood[degree == 0] = np.nan
            for i in range(start, end):
                features[f"{direction}_neighbor_x{i}"] = neighborhood[:, i-start]
        for i, profile in enumerate(profiles):
            values = (edges @ profile) / denom
            values[degree == 0] = np.nan
            features[f"{direction}_neighbor_profile{i}"] = values
    if detailed_events:
        add_event_features(features, adjacency, src, dst, day, kind)
    if two_hop:
        add_two_hop_features(features, adjacency, x, src, dst)
    return features, adjacency


def add_event_features(features, adjacency, src, dst, day, kind):
    n = adjacency.shape[0]
    for mode in ("weak", "strong"):
        _, group = connected_components(adjacency, directed=True, connection=mode)
        features[f"{mode}_component_size"] = np.log1p(np.bincount(group)[group]).astype(np.float32)
    common_types = np.argsort(np.bincount(kind))[-5:]
    width = int(day.max()) + 1
    for name, nodes, others, other_name in (("out", src, dst, "in"), ("in", dst, src, "out")):
        degree = features[f"{name}_degree"]
        first, last = features[f"{name}_first"], features[f"{name}_last"]
        for limit in (7, 30):
            near_first = day - first[nodes] < limit
            near_last = last[nodes] - day < limit
            for period, mask in (("start", near_first), ("end", near_last)):
                count = np.bincount(nodes[mask], minlength=n)
                features[f"{name}_{period}_{limit}d"] = (count / np.maximum(degree, 1)).astype(np.float32)
        active_days = np.unique(nodes * width + day)
        owner = active_days // width
        same = owner[1:] == owner[:-1]
        gap = np.zeros(n, dtype=np.float32)
        np.maximum.at(gap, owner[1:][same], np.diff(active_days)[same])
        features[f"{name}_longest_gap"] = gap

        # 联系发生时，对方已在图上活跃了多久、被联系过多少次？
        other_first = features[f"{other_name}_first"][others]
        other_age = day - other_first
        for field, v in (("other_age", other_age), ("new_contact", other_age < 7),
                         ("other_popular", features[f"{other_name}_degree"][others] >= 5)):
            summed = np.bincount(nodes, weights=v, minlength=n)
            a = (summed / np.maximum(degree, 1)).astype(np.float32)
            a[degree == 0] = np.nan
            features[f"{name}_{field}"] = a
        for k in common_types:
            mask = kind == k
            start = np.full(n, np.inf, dtype=np.float32)
            finish = np.full(n, -np.inf, dtype=np.float32)
            np.minimum.at(start, nodes[mask], day[mask])
            np.maximum.at(finish, nodes[mask], day[mask])
            start[~np.isfinite(start)] = np.nan
            finish[~np.isfinite(finish)] = np.nan
            features[f"{name}_type{k}_first"] = start
            features[f"{name}_type{k}_last"] = finish


def add_two_hop_features(features, adjacency, x, src, dst):
    """比较共享联系人或被同一人列出的其他用户；排除走两步回到自身的部分。"""
    n = len(x)
    for name, edges, owner, other in (("out", adjacency, src, dst),
                                      ("in", adjacency.T.tocsr(), dst, src)):
        degree = features[f"{name}_degree"]
        other_degree = np.maximum(np.asarray(edges.sum(axis=0)).ravel(), 1)
        own_mass = edges.power(2) @ (1.0 / other_degree)
        peer_mass = degree - own_mass
        cols = [1, 2, 4, 9, 12, 15, 16]
        for c in cols:
            values = x[:, c]
            walk = edges @ ((edges.T @ values) / other_degree)
            avg = np.full(n, np.nan, dtype=np.float32)
            np.divide(walk - values * own_mass, peer_mass, where=peer_mass > 1e-4, out=avg)
            features[f"{name}_peer_x{c}"] = avg
        for c in (1, 2, 15, 16):
            hi = np.full(n, -np.inf, dtype=np.float32)
            np.maximum.at(hi, owner, x[other, c])
            hi[degree == 0] = np.nan
            features[f"{name}_neighbor_max_x{c}"] = hi
            missing = np.bincount(owner, weights=x[other, c] < -0.99, minlength=n)
            features[f"{name}_neighbor_missing_x{c}"] = (missing / np.maximum(degree, 1)).astype(np.float32)
        for field in ("first", "last", "mean_day"):
            v = features[f"{name}_{field}"]
            mean_peer = edges @ ((edges.T @ np.nan_to_num(v)) / other_degree)
            avg = np.full(n, np.nan, dtype=np.float32)
            np.divide(mean_peer - np.nan_to_num(v) * own_mass, peer_mass,
                      where=peer_mass > 1e-4, out=avg)
            features[f"{name}_peer_{field}"] = avg
            features[f"{name}_peer_diff_{field}"] = v - avg
    for field in ("first", "last", "mean_day"):
        features["out_minus_in_" + field] = features["out_" + field] - features["in_" + field]


def matrix(features, ids):
    values = np.empty((len(ids), len(features)), order="F", dtype=np.float32)
    for col, feature in enumerate(features.values()):
        values[:, col] = feature[ids]
    return values


def training_neighbor_labels(adjacency, train_ids, train_labels, seed):
    """预测训练节点时隐藏该节点所在折的全部标签；其余节点仅能看到训练标签。"""
    n = adjacency.shape[0]
    incoming = adjacency.T.tocsr()
    out_degree = np.maximum(np.asarray(adjacency.sum(axis=1)), 1)
    in_degree = np.maximum(np.asarray(incoming.sum(axis=1)), 1)
    values = {}

    def store(name, counts, targets, prior):
        evidence, fraud = counts[targets].T
        for suffix, v in (("known", np.log1p(evidence)), ("fraud", np.log1p(fraud)),
                          ("rate", (fraud + prior) / (evidence + 1))):
            key = f"{name}_{suffix}"
            if key not in values:
                values[key] = np.empty(n, dtype=np.float32)
            values[key][targets] = v

    def fill(donors, targets):
        marks = np.zeros((n, 2), dtype=np.float32)
        marks[train_ids[donors], 0] = 1
        marks[train_ids[donors], 1] = train_labels[donors]
        prior = train_labels[donors].mean()
        sent = adjacency @ marks
        received = incoming @ marks
        store("listed", sent, targets, prior)
        store("listed_by", received, targets, prior)
        # 两个用户可能共享同一联系人；按中间节点度数分摊贡献。
        store("same_contact", adjacency @ (received / in_degree), targets, prior)
        store("listed_together", incoming @ (sent / out_degree), targets, prior)
        store("out_two_steps", adjacency @ (sent / out_degree), targets, prior)
        store("in_two_steps", incoming @ (received / in_degree), targets, prior)

    rng = np.random.default_rng(seed)
    donors = np.ones(len(train_ids), dtype=bool)
    for fold in np.array_split(rng.permutation(len(train_ids)), 5):
        donors[fold] = False
        fill(donors, train_ids[fold])
        donors[fold] = True
    remaining = np.ones(n, dtype=bool)
    remaining[train_ids] = False
    fill(donors, np.flatnonzero(remaining))
    return values


def peer_activity_features(raw, n, train_ids, train_labels, seed):
    """按联系人和日期排序，统计共同联系人的同期活动。标签统计同样隐藏查询节点所在折。"""
    src, dst = raw["edge_index"].T
    day = raw["edge_timestamp"]
    width = int(day.max()) + 1
    fold_id = np.full(n, -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    for fold, rows in enumerate(np.array_split(rng.permutation(len(train_ids)), 5)):
        fold_id[train_ids[rows]] = fold
    features = {}
    for direction, owner, contact in (("sent", src, dst), ("received", dst, src)):
        key = contact * width + day
        order = np.argsort(key, kind="stable")
        key, owner, day_sorted = key[order], owner[order], day[order]
        origin = key - day_sorted
        owner_fold = fold_id[owner]
        degree = np.maximum(np.bincount(owner, minlength=n), 1)
        for window in (0, 7, 30):
            low = np.searchsorted(key, origin + np.maximum(day_sorted - window, 0))
            high = np.searchsorted(key, origin + np.minimum(day_sorted + window, width - 1), side="right")
            # 去掉当前这条边。在同一联系人处，其余同期记录有多少？
            volume = high - low - 1
            name = f"peer_{direction}_{window}d"
            features[name + "_activity"] = (np.bincount(owner, weights=np.log1p(volume), minlength=n)
                                             / degree).astype(np.float32)
            maximum = np.zeros(n, dtype=np.float32)
            np.maximum.at(maximum, owner, volume)
            features[name + "_max"] = maximum
            if window == 0:
                continue
            counts = np.empty(n, dtype=np.float32)
            rates = np.empty(n, dtype=np.float32)
            for fold in range(-1, 5):
                donors = fold_id[train_ids] != fold
                targets = np.flatnonzero(fold_id == fold)
                take = owner_fold == fold
                # 前缀和使每条边只需两次查询即可取得时间窗内的已知/欺诈事件数。
                known = np.zeros(n, dtype=np.int8)
                known[train_ids[donors]] = 1
                fraud = np.zeros(n, dtype=np.int8)
                fraud[train_ids[donors]] = train_labels[donors]
                totals = []
                for indicator in (known, fraud):
                    prefix = np.empty(len(owner) + 1, dtype=np.int32)
                    prefix[0] = 0
                    np.cumsum(indicator[owner], dtype=np.int32, out=prefix[1:])
                    amount = prefix[high[take]] - prefix[low[take]]
                    node_sum = np.bincount(owner[take], weights=amount, minlength=n)
                    totals.append(node_sum[targets])
                evidence, positive = totals
                counts[targets] = np.log1p(evidence)
                rates[targets] = (positive + train_labels[donors].mean()) / (evidence + 1)
            features[name + "_known"] = counts
            features[name + "_fraud_rate"] = rates
    return features


def cohort_risk(features, train_ids, train_labels, seed, granular=False):
    """在相同活动日期、相似联系数量的训练用户中估计欺诈率，按折排除自身标签。"""
    n = len(features["x0"])
    folds = np.full(n, -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    for fold, rows in enumerate(np.array_split(rng.permutation(len(train_ids)), 5)):
        folds[train_ids[rows]] = fold
    result = {}
    for direction in ("out", "in"):
        for endpoint in ("first", "last"):
            days = np.nan_to_num(features[f"{direction}_{endpoint}"], nan=0).astype(np.int32)
            n_days = int(days.max()) + 1
            degree = features[f"{direction}_degree"]
            band = np.minimum(degree, 5).astype(np.int32)
            groupings = [("all", np.zeros(n, dtype=np.int32), (0, 3)), ("degree", band, (0, 3))]
            if granular and direction == "out" and endpoint == "last":
                # 同一时段比较资料或联系人类型相似的用户。分箱只由训练特征决定。
                x1 = features["x1"]
                observed = x1[train_ids]
                cuts = np.unique(np.quantile(observed[observed >= -0.99], [0.25, 0.5, 0.75]))
                x1_group = np.where(x1 < -0.99, 0, 1 + np.searchsorted(cuts, x1)).astype(np.int32)
                dominant = np.zeros(n, dtype=np.int32)
                largest = np.zeros(n, dtype=np.float32)
                for k in range(1, 12):
                    count = features[f"out_type_{k}"]
                    better = count > largest
                    dominant[better], largest[better] = k, count[better]
                types = np.rint(features["x2"]).astype(np.int32)
                groupings.extend([
                    ("x1", x1_group, (3, 14)), ("x2", types - types.min(), (3, 14)),
                    ("dominant_type", dominant, (3, 14)),
                    ("incoming_degree", np.minimum(features["in_degree"], 5).astype(np.int32), (3, 14)),
                    ("activity_span", np.digitize(features["out_span"], [1, 7, 30, 120]).astype(np.int32), (3, 14)),
                ])
            for grouped, groups, windows in groupings:
                stride = n_days + 1
                keys = groups * stride + days
                width = (int(groups.max()) + 1) * stride
                for window in windows:
                    out = np.empty(n, dtype=np.float32)
                    for fold in range(-1, 5):
                        donors = folds[train_ids] != fold
                        donor_keys = keys[train_ids[donors]]
                        known = np.bincount(donor_keys, minlength=width)
                        positive = np.bincount(donor_keys, weights=train_labels[donors], minlength=width)
                        rows = np.flatnonzero(folds == fold)
                        block = groups[rows] * stride
                        active_day = days[rows]
                        low = block + np.where(active_day > 0, np.maximum(active_day - window, 1), 0)
                        high = block + np.where(active_day > 0, np.minimum(active_day + window, n_days - 1), 0) + 1
                        cp = np.r_[0, positive.cumsum()]
                        ck = np.r_[0, known.cumsum()]
                        prior = train_labels[donors].mean()
                        out[rows] = (cp[high] - cp[low] + 40 * prior) / (ck[high] - ck[low] + 40)
                    result[f"cohort_{direction}_{endpoint}_{grouped}_{window}d"] = out
    return result


def typed_neighbor_labels(raw, n, train_ids, train_labels, seed):
    """沿常见关系类型分别统计已知训练邻居；同折训练邻居的标签不参与。"""
    src, dst = raw["edge_index"].T
    kinds = raw["edge_type"]
    folds = np.full(n, -1, dtype=np.int8)
    for i, rows in enumerate(np.array_split(np.random.default_rng(seed).permutation(len(train_ids)), 5)):
        folds[train_ids[rows]] = i
    fraud = np.zeros(n, dtype=np.float32)
    fraud[train_ids] = train_labels
    prior = np.full(n, train_labels.mean(), dtype=np.float32)
    for i in range(5):
        prior[folds == i] = train_labels[folds[train_ids] != i].mean()
    result = {}
    for name, owner, other in (("out", src, dst), ("in", dst, src)):
        observed = (folds[other] >= 0) & (folds[owner] != folds[other])
        for k in np.argsort(np.bincount(kinds))[-5:]:
            mask = observed & (kinds == k)
            known = np.bincount(owner[mask], minlength=n)
            positive = np.bincount(owner[mask], weights=fraud[other[mask]], minlength=n)
            result[f"{name}_known_type{k}"] = np.log1p(known).astype(np.float32)
            result[f"{name}_fraud_type{k}"] = ((positive + prior) / (known + 1)).astype(np.float32)
    return result


def relation_profiles(raw, features, train_ids, train_labels, seed):
    """区分关系类型下的邻居资料，并估计节点联系发生在高风险时段的程度。"""
    src, dst = raw["edge_index"].T
    kind, day = raw["edge_type"], raw["edge_timestamp"]
    last = np.nan_to_num(features["out_last"], nan=0).astype(np.int32)
    n = len(last)
    folds = np.full(n, -1, dtype=np.int8)
    for fold, rows in enumerate(np.array_split(np.random.default_rng(seed).permutation(len(train_ids)), 5)):
        folds[train_ids[rows]] = fold
    days = np.arange(int(day.max()) + 1)
    by_fold = np.empty((6, len(days)), dtype=np.float32)
    for fold in range(-1, 5):
        donors = folds[train_ids] != fold
        event_days = last[train_ids[donors]]
        known = np.bincount(event_days, minlength=len(days))
        fraud = np.bincount(event_days, weights=train_labels[donors], minlength=len(days))
        low = np.where(days, np.maximum(1, days - 3), 0)
        high = np.where(days, np.minimum(days[-1], days + 3), 0) + 1
        kn, fr = np.r_[0, known.cumsum()], np.r_[0, fraud.cumsum()]
        by_fold[fold+1] = (fr[high] - fr[low] + 40 * train_labels[donors].mean()) / (kn[high] - kn[low] + 40)
    result = {}
    for name, owner, other in (("out", src, dst), ("in", dst, src)):
        degree = features[f"{name}_degree"]
        for label, dates in (("event", day), ("neighbor_last", last[other])):
            risk = by_fold[folds[owner] + 1, dates]
            avg = (np.bincount(owner, weights=risk, minlength=n) / np.maximum(degree, 1)).astype(np.float32)
            peak = np.zeros(n, dtype=np.float32)
            np.maximum.at(peak, owner, risk)
            avg[degree == 0] = np.nan
            peak[degree == 0] = np.nan
            result[f"{name}_{label}_period_avg"] = avg
            result[f"{name}_{label}_period_peak"] = peak
        for k in np.argsort(np.bincount(kind))[-4:]:
            mask = kind == k
            source, target = owner[mask], other[mask]
            cnt = np.bincount(source, minlength=n)
            for c in (1, 2, 15, 16):
                values = features[f"x{c}"][target]
                avg = (np.bincount(source, weights=values, minlength=n) / np.maximum(cnt, 1)).astype(np.float32)
                avg[cnt == 0] = np.nan
                result[f"{name}_type{k}_neighbor_x{c}"] = avg
    return result


def select_training_nodes(train_ids, labels, seed, negative_ratio):
    positives = train_ids[labels[train_ids] == 1]
    negatives = train_ids[labels[train_ids] == 0]
    rng = np.random.default_rng(seed)
    negatives = rng.choice(negatives, min(len(negatives), negative_ratio * len(positives)), replace=False)
    return rng.permutation(np.concatenate((positives, negatives)))


def fit_tree(features, train_ids, valid_ids, labels, seed, negative_ratio, depth, iterations):
    fit_ids = select_training_nodes(train_ids, labels, seed, negative_ratio)
    print(f"拟合节点 {len(fit_ids):,}，特征 {len(features)}，树深 {depth}", flush=True)
    tree = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=0.07, l2_leaf_reg=5,
        border_count=96, loss_function="Logloss", eval_metric="AUC", thread_count=5,
        random_seed=seed, allow_writing_files=False, verbose=100,
    )
    # 只用一半官方验证节点选停止轮数，另报另一半的 AUC。
    tree.fit(matrix(features, fit_ids), labels[fit_ids],
             eval_set=(matrix(features, valid_ids[::2]), labels[valid_ids[::2]]),
             early_stopping_rounds=90, use_best_model=True)
    return tree


def fit_lightgbm(features, train_ids, valid_ids, labels, seed, negative_ratio, depth, iterations):
    import lightgbm as lgb

    fit_ids = select_training_nodes(train_ids, labels, seed, negative_ratio)
    print(f"LightGBM 拟合节点 {len(fit_ids):,}，特征 {len(features)}", flush=True)
    fit = lgb.Dataset(matrix(features, fit_ids), label=labels[fit_ids], feature_name=list(features))
    tune_ids = valid_ids[::2]
    tune = lgb.Dataset(matrix(features, tune_ids), label=labels[tune_ids], reference=fit,
                       feature_name=list(features))
    return lgb.train(
        dict(objective="binary", metric="auc", learning_rate=0.035,
             max_depth=depth, num_leaves=min(63, 2**depth), min_data_in_leaf=140,
             feature_fraction=0.83, bagging_fraction=0.85, bagging_freq=1, lambda_l2=6,
             max_bin=127, num_threads=5, seed=seed, verbosity=-1, force_col_wise=True),
        fit, num_boost_round=iterations, valid_sets=[tune],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )


def build_features(data_dir, seed, use_labels=False, detailed_events=False,
                   use_peers=False, two_hop=False, use_cohorts=False,
                   typed_labels=False, granular=False, profiles=False):
    with np.load(data_dir / "DGraph/raw/dgraph.npz") as raw:
        train_ids, valid_ids = raw["train_mask"], raw["valid_mask"]
        labels = raw["y"].reshape(-1)
        features, adjacency = collect_features(raw, detailed_events, two_hop)
        if use_peers:
            features.update(peer_activity_features(raw, len(labels), train_ids, labels[train_ids], seed))
        if typed_labels:
            features.update(typed_neighbor_labels(raw, len(labels), train_ids, labels[train_ids], seed))
        if profiles:
            features.update(relation_profiles(raw, features, train_ids, labels[train_ids], seed))
    if use_labels:
        features.update(training_neighbor_labels(adjacency, train_ids, labels[train_ids], seed))
    if use_cohorts or granular:
        features.update(cohort_risk(features, train_ids, labels[train_ids], seed, granular))
    del adjacency
    return features, train_ids, valid_ids, labels


def train_boosted(data_dir, results, seed, negative_ratio, depth, iterations, use_labels=False,
                  detailed_events=False, use_peers=False, engine="catboost", two_hop=False,
                  use_cohorts=False, typed_labels=False, granular=False, profiles=False):
    features, train_ids, valid_ids, labels = build_features(
        data_dir, seed, use_labels, detailed_events, use_peers, two_hop,
        use_cohorts, typed_labels, granular, profiles)
    if engine == "catboost":
        tree = fit_tree(features, train_ids, valid_ids, labels, seed, negative_ratio, depth, iterations)
        tree_count = tree.tree_count_
        tree.set_feature_names(list(features))
    else:
        tree = fit_lightgbm(features, train_ids, valid_ids, labels, seed, negative_ratio, depth, iterations)
        tree_count = tree.best_iteration
    n = len(labels)
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, 60000):
        ids = np.arange(start, min(start + 60000, n))
        block = matrix(features, ids)
        if engine == "catboost":
            scores[ids] = tree.predict(block, prediction_type="RawFormulaVal", thread_count=5)
        else:
            scores[ids] = tree.predict(block, raw_score=True, num_threads=5)

    # 修正对负类均匀下采样引入的先验偏移，后续以 float32 概率测试实际发布结果。
    neg = int((labels[train_ids] == 0).sum())
    pos = len(train_ids) - neg
    kept = min(neg, negative_ratio * pos)
    probs = torch.sigmoid(torch.from_numpy(scores) + np.log(kept / neg))
    auc = roc_auc_score(labels[valid_ids], probs[valid_ids].numpy())
    check = valid_ids[1::2]
    check_auc = roc_auc_score(labels[check], probs[check].numpy())
    tag = "known" if use_labels else "structure"
    run = results / f"trees_{tag}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run.mkdir(parents=True)
    tree.save_model(str(run / ("model.cbm" if engine == "catboost" else "model.lgb")))
    torch.save(probs, run / "scores.pt")
    metrics = dict(valid_auc=float(auc), other_half_auc=float(check_auc),
                   tree_count=tree_count, depth=depth, seed=seed, negative_ratio=negative_ratio,
                   feature_count=len(features), use_training_labels_as_features=use_labels,
                   detailed_events=detailed_events, peer_activity=use_peers, engine=engine,
                   two_hop=two_hop, cohort_risk=use_cohorts or granular, typed_labels=typed_labels,
                   granular_cohorts=granular, relation_profiles=profiles)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"完整验证 AUC {auc:.7f}，未用于选轮数的一半 {check_auc:.7f}\n{run}", flush=True)
    return run


def export_boosted(runs, data_dir, results, weights=None):
    """打包训练好的树及固定 DGraph 上的预测，使 Mo 可以直接在 CPU 上按节点查询。"""
    logits = []
    members = []
    if weights is None:
        weights = np.ones(len(runs), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(runs) or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("每个成员模型需要一个正的组合权重")
    weights = weights / weights.sum()
    recipe = []
    for run, weight in zip(runs, weights):
        metric = json.loads((run / "metrics.json").read_text())
        native = run / ("model.lgb" if metric.get("engine") == "lightgbm" else "model.cbm")
        scores = torch.load(run / "scores.pt", weights_only=True, map_location="cpu")
        logits.append(torch.logit(scores, eps=1e-7) * float(weight))
        members.append(dict(run=run.name, metrics=metric, format=native.suffix,
                            mix_weight=float(weight),
                            weights=torch.frombuffer(bytearray(native.read_bytes()), dtype=torch.uint8)))
        recipe.append(dict(engine=metric.get("engine", "catboost"), seed=metric["seed"],
                           negative_ratio=metric["negative_ratio"], depth=metric["depth"],
                           iterations=metric["tree_count"], use_labels=metric["use_training_labels_as_features"],
                           detailed_events=metric.get("detailed_events", False), use_peers=metric.get("peer_activity", False),
                           two_hop=metric.get("two_hop", False), use_cohorts=metric.get("cohort_risk", False),
                           typed_labels=metric.get("typed_labels", False), granular=metric.get("granular_cohorts", False),
                           profiles=metric.get("relation_profiles", False)))
    probability = torch.stack(logits).sum(0).sigmoid()
    with np.load(data_dir / "DGraph/raw/dgraph.npz") as raw:
        valid = raw["valid_mask"]
        auc = roc_auc_score(raw["y"][valid], probability[valid].numpy())
    path = results / "optimized_model.pt"
    torch.save(dict(positive_probability=probability, members=members), path)
    spec = dict(mix_weights=weights.tolist(), models=recipe)
    (results / "optimized_recipe.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"打包 {len(members)} 个模型，完整验证 AUC {auc:.7f}\n{path}")


def rebuild_best(data_dir, results):
    recipe = json.loads((results / "optimized_recipe.json").read_text())
    runs = [train_boosted(data_dir, results, **params) for params in recipe["models"]]
    export_boosted(runs, data_dir, results, recipe["mix_weights"])
