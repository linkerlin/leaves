#!/usr/bin/env python3
"""生成 rank:ndcg 梯度 golden（topk/mean + |ΔNDCG| 缩放，preds=0）及 XGBoost round-1 margin。"""
import json
import math
import sys

try:
    import numpy as np
    import xgboost as xgb
except ImportError:
    print("requires: numpy xgboost", file=sys.stderr)
    raise

SEED = 42
OUT = "rank_ndcg_grad_golden.json"

LABELS = [3.0, 1.0, 0.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.0]
GROUPS = [3, 2, 4]
PREDS = [0.0] * len(LABELS)
X = np.array(
    [
        [1.0, 0.0],
        [0.5, 0.1],
        [0.0, 1.0],
        [0.9, 0.0],
        [0.4, 0.2],
        [2.0, 0.0],
        [1.2, 0.3],
        [0.8, 0.5],
        [0.1, 0.9],
    ],
    dtype=np.float64,
)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def gain(rel: float) -> float:
    return 0.0 if rel <= 0 else (2.0**rel - 1.0)


def discount_at(pos: int) -> float:
    return 1.0 / math.log2(pos + 2.0)


def ideal_dcg(labels):
    sorted_l = sorted(labels, reverse=True)
    return sum(gain(r) * discount_at(i) for i, r in enumerate(sorted_l))


def current_ranks(preds):
    order = sorted(range(len(preds)), key=lambda i: (-preds[i], i))
    ranks = [0] * len(preds)
    for pos, idx in enumerate(order):
        ranks[idx] = pos
    return ranks


def delta_ndcg(labels, ranks, i, j, ideal):
    if ideal <= 0:
        return 0.0
    gi, gj = gain(labels[i]), gain(labels[j])
    di, dj = discount_at(ranks[i]), discount_at(ranks[j])
    return abs(gi * (dj - di) + gj * (di - dj)) / ideal


def minstd(seed):
    if seed == 0:
        seed = 1
    state = seed

    def intn(n):
        nonlocal state
        state = (state * 48271) % 2147483647
        return state % n

    def discard(k):
        nonlocal state
        for _ in range(k):
            state = (state * 48271) % 2147483647

    return intn, discard


def topk_pairs(n, k):
    limit = min(n, k)
    pairs = []
    for i in range(limit):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def mean_pairs(labels, preds, num_sample, boost_round, group_idx):
    n = len(labels)
    rank_idx = sorted(range(n), key=lambda i: (-preds[i], i))
    y_sorted = sorted(range(n), key=lambda i: (-labels[rank_idx[i]], i))
    intn, discard = minstd(boost_round)
    discard(group_idx)
    pairs = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[rank_idx[y_sorted[j]]] == labels[rank_idx[y_sorted[i]]]:
            j += 1
        n_left, n_right = i, n - j
        if n_left + n_right == 0:
            i = j
            continue
        for _ in range(num_sample):
            for pair_idx in range(i, j):
                ridx = intn(n_left + n_right)
                if ridx >= n_left:
                    ridx = ridx - i + j
                pairs.append((y_sorted[pair_idx], y_sorted[ridx]))
        i = j
    return pairs, rank_idx


def ndcg_grad_group(labels, preds, pair_method, num_pair, gi, br=0, norm=True):
    n = len(labels)
    rank_idx = sorted(range(n), key=lambda i: (-preds[i], i))
    ideal = ideal_dcg(labels)
    ranks = current_ranks(preds)
    if pair_method == "topk":
        pairs = topk_pairs(n, num_pair if num_pair > 0 else 32)
    else:
        pairs, rank_idx = mean_pairs(labels, preds, num_pair if num_pair > 0 else 1, br, gi)

    grad = [0.0] * n
    hess = [0.0] * n
    sum_lam = 0.0
    for rh, rl in pairs:
        if rh > rl:
            rh, rl = rl, rh
        ih, il = rank_idx[rh], rank_idx[rl]
        if labels[ih] == labels[il]:
            continue
        if labels[ih] < labels[il]:
            ih, il = il, ih
        scale = delta_ndcg(labels, ranks, ih, il, ideal)
        if scale <= 0:
            continue
        sh, sl = preds[ih], preds[il]
        sig = sigmoid(sh - sl)
        lam = (sig - 1.0) * scale
        h = max(sig * (1 - sig), 1e-16) * scale * 2.0
        grad[ih] += lam
        grad[il] -= lam
        hess[ih] += h
        hess[il] += h
        sum_lam += -2.0 * lam
    if norm and sum_lam > 0:
        fac = math.log2(1 + sum_lam) / sum_lam
        grad = [g * fac for g in grad]
        hess = [h * fac for h in hess]
    for i in range(n):
        if hess[i] < 1e-16:
            hess[i] = 1e-16
    return grad, hess


def xgb_margin(pair_method, num_pair):
    dm = xgb.DMatrix(X, label=np.array(LABELS, dtype=np.float64))
    dm.set_group(GROUPS)
    params = {
        "objective": "rank:ndcg",
        "seed": SEED,
        "eta": 1.0,
        "max_depth": 6,
        "lambda": 0.0,
        "alpha": 0.0,
        "min_child_weight": 0.0,
        "base_score": 0.0,
        "tree_method": "exact",
        "gamma": 0.0,
        "lambdarank_pair_method": pair_method,
        "lambdarank_num_pair_per_sample": num_pair,
        "lambdarank_normalization": True,
    }
    bst = xgb.train(params, dm, num_boost_round=1)
    return [float(v) for v in bst.predict(dm, output_margin=True)]


def main():
    topk_n = 32
    mean_n = 1
    grad_all_topk = []
    grad_all_mean = []
    start = 0
    group_topk = []
    group_mean = []
    for gi, gsz in enumerate(GROUPS):
        gl = LABELS[start : start + gsz]
        gp = PREDS[start : start + gsz]
        gt, ht = ndcg_grad_group(gl, gp, "topk", topk_n, gi)
        gm, hm = ndcg_grad_group(gl, gp, "mean", mean_n, gi)
        grad_all_topk.extend(gt)
        grad_all_mean.extend(gm)
        group_topk.append({"group": gi, "grad": gt, "hess": ht})
        group_mean.append({"group": gi, "grad": gm, "hess": hm})
        start += gsz

    golden = {
        "seed": SEED,
        "objective": "rank:ndcg",
        "groups": GROUPS,
        "labels": LABELS,
        "preds": PREDS,
        "topk": {
            "num_pair_per_sample": topk_n,
            "grad": grad_all_topk,
            "xgb_margin_round1": xgb_margin("topk", topk_n),
        },
        "mean": {
            "num_pair_per_sample": mean_n,
            "grad": grad_all_mean,
            "xgb_margin_round1": xgb_margin("mean", mean_n),
        },
        "group_topk": group_topk,
        "tolerance": {"grad": 1e-10, "hess": 1e-10, "margin": 1e-4},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
    print(f"wrote {OUT}")
    print("topk margin", golden["topk"]["xgb_margin_round1"])


if __name__ == "__main__":
    main()
