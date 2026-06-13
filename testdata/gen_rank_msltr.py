#!/usr/bin/env python3
"""从 MSLR-WEB10K Fold1 子集生成排序 TSV 与 XGBoost 平行基准（生产向对比）。"""
import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import OrderedDict

try:
    import numpy as np
    import xgboost as xgb
except ImportError:
    print("requires: numpy xgboost", file=sys.stderr)
    raise

SEED = 42
N_FEAT = 136
TRAIN_MAX_Q = 120
TEST_MAX_Q = 40
NUM_ROUND = 50
NDCG_K = 10

MSLR_URL = "https://storage.googleapis.com/personalization-takehome/MSLR-WEB10K.zip"
CACHE_ZIP = os.path.join(".cache", "MSLR-WEB10K.zip")

OUT_TRAIN = "rank_msltr_train.tsv"
OUT_TEST = "rank_msltr_test.tsv"
OUT_NDCG_BASELINE = "rank_msltr_ndcg_xgb_baseline.json"
OUT_PAIRWISE_BASELINE = "rank_msltr_pairwise_xgb_baseline.json"


def ensure_zip(cache_path=CACHE_ZIP, url=MSLR_URL):
    if os.path.isfile(cache_path):
        return cache_path
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    print(f"downloading {url} -> {cache_path} ...", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    with open(cache_path, "wb") as f:
        f.write(data)
    print(f"cached {len(data)} bytes", file=sys.stderr)
    return cache_path


def parse_fold_subset(zf, member, max_queries):
    groups = OrderedDict()
    for raw in zf.open(member):
        line = raw.decode().strip()
        if not line:
            continue
        parts = line.split()
        label = float(parts[0])
        qid = int(parts[1].split(":")[1])
        if qid not in groups:
            if len(groups) >= max_queries:
                continue
            groups[qid] = []
        if qid in groups:
            feat = [0.0] * N_FEAT
            for tok in parts[2:]:
                idx_s, val_s = tok.split(":")
                feat[int(idx_s) - 1] = float(val_s)
            groups[qid].append((label, feat))

    rows = []
    group_sizes = []
    for qid in groups:
        docs = groups[qid]
        group_sizes.append(len(docs))
        for label, feat in docs:
            rows.append((qid, label, feat))
    return rows, group_sizes


def write_ranking_tsv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# MSLR-WEB10K subset: qid label feat1..feat136\n")
        for qid, label, feat in rows:
            feats = "\t".join(f"{v:.17g}" for v in feat)
            f.write(f"{qid}\t{label:.17g}\t{feats}\n")


def rows_to_dmatrix(rows, groups):
    x = np.array([r[2] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.float64)
    dm = xgb.DMatrix(x, label=y)
    dm.set_group(groups)
    return dm


def train_baseline(objective, dtrain, dtest):
    eval_metric = f"ndcg@{NDCG_K}"
    params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "eta": 0.1,
        "max_depth": 6,
        "lambda": 1.0,
        "seed": SEED,
        "tree_method": "hist",
    }
    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_ROUND,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False,
    )
    metric_key = list(evals_result["train"].keys())[0]
    train_hist = [float(v) for v in evals_result["train"][metric_key]]
    test_hist = [float(v) for v in evals_result["test"][metric_key]]
    return bst, metric_key, train_hist, test_hist, eval_metric


def write_baseline(path, objective, eval_metric, metric_key, train_hist, test_hist, extra):
    meta = {
        "dataset": "MSLR-WEB10K",
        "fold": "Fold1",
        "seed": SEED,
        "objective": objective,
        "eval_metric": eval_metric,
        "metric_key": metric_key,
        "ndcg_k": NDCG_K,
        "num_round": NUM_ROUND,
        "max_depth": 6,
        "learning_rate": 0.1,
        "lambda": 1.0,
        "tree_method": "hist",
        "n_feat": N_FEAT,
        "train_ndcg": train_hist,
        "test_ndcg": test_hist,
        "final_train_ndcg": train_hist[-1],
        "final_test_ndcg": test_hist[-1],
        "initial_train_ndcg": train_hist[0],
        "initial_test_ndcg": test_hist[0],
    }
    meta.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--local-zip",
        default="",
        help="use existing MSLR-WEB10K.zip instead of downloading",
    )
    ap.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    args = ap.parse_args()

    outputs = [OUT_TRAIN, OUT_TEST, OUT_NDCG_BASELINE, OUT_PAIRWISE_BASELINE]
    if not args.force and all(os.path.isfile(p) for p in outputs):
        print("outputs exist, use --force to regenerate")
        return

    zip_path = args.local_zip or ensure_zip()
    with zipfile.ZipFile(zip_path if args.local_zip else CACHE_ZIP) as zf:
        train_rows, train_groups = parse_fold_subset(zf, "Fold1/train.txt", TRAIN_MAX_Q)
        test_rows, test_groups = parse_fold_subset(zf, "Fold1/test.txt", TEST_MAX_Q)

    write_ranking_tsv(OUT_TRAIN, train_rows)
    write_ranking_tsv(OUT_TEST, test_rows)

    dtrain = rows_to_dmatrix(train_rows, train_groups)
    dtest = rows_to_dmatrix(test_rows, test_groups)
    extra = {
        "train_queries": len(train_groups),
        "test_queries": len(test_groups),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }

    for objective, out_path in (
        ("rank:ndcg", OUT_NDCG_BASELINE),
        ("rank:pairwise", OUT_PAIRWISE_BASELINE),
    ):
        _, metric_key, train_hist, test_hist, eval_metric = train_baseline(
            objective, dtrain, dtest
        )
        meta = write_baseline(
            out_path, objective, eval_metric, metric_key, train_hist, test_hist, extra
        )
        print(
            f"{objective}: {out_path}; "
            f"train {train_hist[0]:.4f}->{train_hist[-1]:.4f}; "
            f"test {test_hist[0]:.4f}->{test_hist[-1]:.4f}"
        )

    print(f"wrote {OUT_TRAIN} ({len(train_rows)} rows, {len(train_groups)} queries)")
    print(f"wrote {OUT_TEST} ({len(test_rows)} rows, {len(test_groups)} queries)")


if __name__ == "__main__":
    main()
