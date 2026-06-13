#!/usr/bin/env python3
"""生成 rank:pairwise 逐对 λ golden（全配对 RankNet，|ΔZ|=1）。

leaves 使用经典全配对 LambdaRank；XGBoost 3.x 内置 rank:pairwise 默认 lambdarank_pair_method=mean
会随机采样配对，故 golden 以全配对公式为准，并用同公式的 XGBoost custom objective 校验 round-1 margin。
"""
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
OUT = "rank_pairwise_grad_golden.json"

# 固定 micro 数据集（非 smoke 全量，便于逐对断言）
LABELS = [3.0, 1.0, 0.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.0]
GROUPS = [3, 2, 4]
# 初始 margin = 0（boosting 第 0 轮）
PREDS = [0.0] * len(LABELS)
# 特征仅用于 XGBoost DMatrix（梯度与特征无关）
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


def full_pairwise_group(labels, preds, weights=None):
    n = len(labels)
    grad = [0.0] * n
    hess = [0.0] * n
    pairs = []
    for i in range(n):
        wi = 1.0 if weights is None else max(weights[i], 1.0)
        for j in range(n):
            if labels[i] <= labels[j]:
                continue
            wj = 1.0 if weights is None else max(weights[j], 1.0)
            w = wi * wj
            rho = sigmoid(preds[i] - preds[j])
            lam = (1.0 - rho) * w
            h = rho * (1.0 - rho) * w
            grad[i] -= lam
            grad[j] += lam
            hess[i] += h
            hess[j] += h
            pairs.append(
                {
                    "hi": i,
                    "lo": j,
                    "rho": rho,
                    "lambda": lam,
                    "hess_pair": h,
                }
            )
    min_hess = 1e-16
    for i in range(n):
        if hess[i] < min_hess:
            hess[i] = min_hess
    return grad, hess, pairs


def full_pairwise_obj(predt, dtrain):
    y = dtrain.get_label()
    w = dtrain.get_weight()
    has_w = w is not None and len(w) == len(y)
    gptr = dtrain.get_uint_info("group_ptr")
    grad = np.zeros_like(predt)
    hess = np.zeros_like(predt)
    for gi in range(len(gptr) - 1):
        a, b = int(gptr[gi]), int(gptr[gi + 1])
        labs = y[a:b]
        preds = predt[a:b]
        wg = w[a:b] if has_w else None
        g, h, _ = full_pairwise_group(labs.tolist(), preds.tolist(), wg)
        for k, v in enumerate(g):
            grad[a + k] = v
        for k, v in enumerate(h):
            hess[a + k] = v
    return grad, hess


def main():
    labels = LABELS
    preds = PREDS
    groups = GROUPS
    assert sum(groups) == len(labels)

    grad_all = []
    hess_all = []
    group_pairs = []
    start = 0
    for g, gsz in enumerate(groups):
        gl = labels[start : start + gsz]
        gp = preds[start : start + gsz]
        ggrad, ghess, pairs = full_pairwise_group(gl, gp)
        grad_all.extend(ggrad)
        hess_all.extend(ghess)
        group_pairs.append({"group": g, "size": gsz, "pairs": pairs})
        start += gsz

    dm = xgb.DMatrix(X, label=np.array(labels, dtype=np.float64))
    dm.set_group(groups)
    params = {
        "seed": SEED,
        "eta": 1.0,
        "max_depth": 6,
        "lambda": 0.0,
        "alpha": 0.0,
        "min_child_weight": 0.0,
        "base_score": 0.0,
        "tree_method": "exact",
        "gamma": 0.0,
    }
    bst = xgb.train(params, dm, num_boost_round=1, obj=full_pairwise_obj)
    margin = [float(v) for v in bst.predict(dm, output_margin=True)]

    golden = {
        "seed": SEED,
        "objective": "rank:pairwise",
        "pair_mode": "full",
        "note": "Full pairwise RankNet (|delta_Z|=1). XGBoost 3.x builtin uses sampled pairs; "
        "xgb_custom_margin_round1 uses custom obj with this same formula.",
        "groups": groups,
        "labels": labels,
        "preds": preds,
        "grad": grad_all,
        "hess": hess_all,
        "group_pairs": group_pairs,
        "xgb_custom_margin_round1": margin,
        "tolerance": {"grad": 1e-12, "hess": 1e-12, "margin": 1e-5},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
    print(f"wrote {OUT}: {len(labels)} rows, {sum(len(g['pairs']) for g in group_pairs)} pairs")
    print("grad", grad_all)
    print("xgb margin", margin)


if __name__ == "__main__":
    main()
