#!/usr/bin/env python3
"""生成排序 smoke 数据与 XGBoost rank:ndcg / rank:pairwise 平行基准（T5）。"""
import json
import sys

try:
    import numpy as np
    import xgboost as xgb
except ImportError:
    print("requires: numpy xgboost", file=sys.stderr)
    raise

SEED = 42
N_QUERIES = 24
DOCS_PER_QUERY = 8
N_FEAT = 4
NUM_ROUND = 30
TRAIN_FRAC = 0.75

OUT_TRAIN = "rank_smoke_train.tsv"
OUT_TEST = "rank_smoke_test.tsv"
OUT_NDCG_BASELINE = "rank_smoke_xgb_baseline.json"
OUT_PAIRWISE_BASELINE = "rank_smoke_pairwise_xgb_baseline.json"


def synth_queries(rng, n_queries, docs_per_query, n_feat):
    rows = []
    for q in range(n_queries):
        rel = rng.integers(0, 4, size=docs_per_query)
        for d in range(docs_per_query):
            feat = rng.normal(0.0, 1.0, size=n_feat)
            feat[0] += rel[d] * 1.5
            feat[1] += rng.normal(0.0, 0.2)
            rows.append((q, float(rel[d]), feat))
    return rows


def write_ranking_tsv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# qid label feat1 feat2 ...\n")
        for qid, label, feat in rows:
            feats = "\t".join(f"{v:.17g}" for v in feat)
            f.write(f"{qid}\t{label:.17g}\t{feats}\n")


def rows_to_dmatrix(rows, groups):
    x = np.array([r[2] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.float64)
    dm = xgb.DMatrix(x, label=y)
    dm.set_group(groups)
    return dm


def train_baseline(objective, dtrain, dtest, eval_metric):
    params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "eta": 0.3,
        "max_depth": 3,
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
    return bst, metric_key, train_hist, test_hist


def write_baseline(path, objective, eval_metric, metric_key, train_hist, test_hist, meta_extra):
    meta = {
        "seed": SEED,
        "objective": objective,
        "eval_metric": eval_metric,
        "metric_key": metric_key,
        "num_round": NUM_ROUND,
        "max_depth": 3,
        "learning_rate": 0.3,
        "lambda": 1.0,
        "tree_method": "hist",
        "n_queries": N_QUERIES,
        "docs_per_query": DOCS_PER_QUERY,
        "n_feat": N_FEAT,
        "train_ndcg": train_hist,
        "test_ndcg": test_hist,
        "final_train_ndcg": train_hist[-1],
        "final_test_ndcg": test_hist[-1],
        "initial_train_ndcg": train_hist[0],
        "initial_test_ndcg": test_hist[0],
    }
    meta.update(meta_extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    rng = np.random.default_rng(SEED)
    all_rows = synth_queries(rng, N_QUERIES, DOCS_PER_QUERY, N_FEAT)

    n_train_q = int(N_QUERIES * TRAIN_FRAC)
    train_rows = [r for r in all_rows if r[0] < n_train_q]
    test_rows = [r for r in all_rows if r[0] >= n_train_q]
    train_groups = [DOCS_PER_QUERY] * n_train_q
    test_groups = [DOCS_PER_QUERY] * (N_QUERIES - n_train_q)

    write_ranking_tsv(OUT_TRAIN, train_rows)
    write_ranking_tsv(OUT_TEST, test_rows)

    dtrain = rows_to_dmatrix(train_rows, train_groups)
    dtest = rows_to_dmatrix(test_rows, test_groups)
    extra = {
        "train_queries": n_train_q,
        "test_queries": N_QUERIES - n_train_q,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }

    for objective, out_path in (
        ("rank:ndcg", OUT_NDCG_BASELINE),
        ("rank:pairwise", OUT_PAIRWISE_BASELINE),
    ):
        _, metric_key, train_hist, test_hist = train_baseline(
            objective, dtrain, dtest, "ndcg"
        )
        meta = write_baseline(
            out_path, objective, "ndcg", metric_key, train_hist, test_hist, extra
        )
        print(
            f"{objective}: wrote {out_path}; "
            f"train {train_hist[0]:.4f}->{train_hist[-1]:.4f}; "
            f"test {test_hist[0]:.4f}->{test_hist[-1]:.4f}"
        )

    print(f"wrote {OUT_TRAIN} ({len(train_rows)} rows)")
    print(f"wrote {OUT_TEST} ({len(test_rows)} rows)")


if __name__ == "__main__":
    main()
