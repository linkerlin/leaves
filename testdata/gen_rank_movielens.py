#!/usr/bin/env python3
"""MovieLens 100K 排序 Demo：listwise 对标 rank:ndcg（LambdaMART）+ pairwise 平行基准。

XGBoost / leaves 无 rank:listwise 目标；多档相关性（1–5 星）的 listwise 学习请用 rank:ndcg。
"""
import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict

try:
    import numpy as np
    import xgboost as xgb
except ImportError:
    print("requires: numpy xgboost", file=sys.stderr)
    raise

SEED = 42
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
CACHE_ZIP = os.path.join(".cache", "ml-100k.zip")

MIN_RATINGS_PER_USER = 12
TRAIN_USERS = 60
TEST_USERS = 15
NUM_ROUND = 40
NDCG_K = 10
N_GENRE = 19

OUT_TRAIN = "rank_movielens_train.tsv"
OUT_TEST = "rank_movielens_test.tsv"
OUT_NDCG_BASELINE = "rank_movielens_ndcg_xgb_baseline.json"
OUT_PAIRWISE_BASELINE = "rank_movielens_pairwise_xgb_baseline.json"


def ensure_zip(cache_path=CACHE_ZIP, url=ML100K_URL):
    if os.path.isfile(cache_path):
        return cache_path
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    print(f"downloading {url} -> {cache_path} ...", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with open(cache_path, "wb") as f:
        f.write(data)
    print(f"cached {len(data)} bytes", file=sys.stderr)
    return cache_path


def parse_year(date_s):
    if not date_s or len(date_s) < 4:
        return 1995.0
    try:
        return float(date_s[-4:])
    except ValueError:
        return 1995.0


def load_movielens(zf):
    movie_feat = {}
    for raw in zf.open("ml-100k/u.item"):
        line = raw.decode("latin-1").strip()
        parts = line.split("|")
        if len(parts) < 5 + N_GENRE:
            continue
        mid = int(parts[0])
        year = parse_year(parts[2])
        genres = [float(x) for x in parts[5 : 5 + N_GENRE]]
        movie_feat[mid] = (year, genres)

    ratings = []
    for raw in zf.open("ml-100k/u.data"):
        u, m, r, _ts = raw.decode().strip().split("\t")
        ratings.append((int(u), int(m), float(r)))

    pop = defaultdict(int)
    rating_sum = defaultdict(float)
    for _u, m, r in ratings:
        pop[m] += 1
        rating_sum[m] += r

    by_user = defaultdict(list)
    for u, m, r in ratings:
        if m not in movie_feat:
            continue
        year, genres = movie_feat[m]
        feat = [
            np.log1p(pop[m]),
            rating_sum[m] / pop[m],
            (year - 1970.0) / 50.0,
        ]
        feat.extend(genres)
        by_user[u].append((m, r, feat))

    users = sorted(u for u, rows in by_user.items() if len(rows) >= MIN_RATINGS_PER_USER)
    return by_user, users


def split_users(users, train_n, test_n, seed):
    rng = np.random.default_rng(seed)
    picked = rng.choice(users, size=train_n + test_n, replace=False)
    return list(picked[:train_n]), list(picked[train_n : train_n + test_n])


def build_rows(by_user, user_ids, qid_base=0):
    rows = []
    groups = []
    for i, uid in enumerate(user_ids):
        qid = qid_base + i
        docs = by_user[uid]
        groups.append(len(docs))
        for _m, rating, feat in docs:
            rows.append((qid, rating, feat))
    return rows, groups


def write_ranking_tsv(path, rows):
    n_feat = len(rows[0][2]) if rows else 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("# MovieLens 100K: qid label feat1..featN (listwise demo -> rank:ndcg)\n")
        for qid, label, feat in rows:
            feats = "\t".join(f"{v:.17g}" for v in feat)
            f.write(f"{qid}\t{label:.17g}\t{feats}\n")
    return n_feat


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
        "max_depth": 4,
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
        "dataset": "MovieLens-100K",
        "demo_note": "listwise 对标 rank:ndcg；无 rank:listwise 目标",
        "seed": SEED,
        "objective": objective,
        "eval_metric": eval_metric,
        "metric_key": metric_key,
        "ndcg_k": NDCG_K,
        "num_round": NUM_ROUND,
        "max_depth": 4,
        "learning_rate": 0.1,
        "lambda": 1.0,
        "tree_method": "hist",
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
    ap = argparse.ArgumentParser(description="MovieLens 100K rank demo (listwise=rank:ndcg)")
    ap.add_argument("--local-zip", default="", help="existing ml-100k.zip path")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    outputs = [OUT_TRAIN, OUT_TEST, OUT_NDCG_BASELINE, OUT_PAIRWISE_BASELINE]
    if not args.force and all(os.path.isfile(p) for p in outputs):
        print("outputs exist, use --force to regenerate")
        return

    zip_path = args.local_zip or ensure_zip()
    with zipfile.ZipFile(zip_path if args.local_zip else CACHE_ZIP) as zf:
        by_user, users = load_movielens(zf)

    train_u, test_u = split_users(users, TRAIN_USERS, TEST_USERS, SEED)
    train_rows, train_groups = build_rows(by_user, train_u, 0)
    test_rows, test_groups = build_rows(by_user, test_u, len(train_u))

    n_feat = write_ranking_tsv(OUT_TRAIN, train_rows)
    write_ranking_tsv(OUT_TEST, test_rows)

    dtrain = rows_to_dmatrix(train_rows, train_groups)
    dtest = rows_to_dmatrix(test_rows, test_groups)
    extra = {
        "n_feat": n_feat,
        "min_ratings_per_user": MIN_RATINGS_PER_USER,
        "train_users": len(train_u),
        "test_users": len(test_u),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "label_range": "1-5 stars as relevance",
    }

    for objective, out_path in (
        ("rank:ndcg", OUT_NDCG_BASELINE),
        ("rank:pairwise", OUT_PAIRWISE_BASELINE),
    ):
        _, metric_key, train_hist, test_hist, eval_metric = train_baseline(
            objective, dtrain, dtest
        )
        write_baseline(
            out_path, objective, eval_metric, metric_key, train_hist, test_hist, extra
        )
        print(
            f"{objective}: {out_path}; "
            f"train {train_hist[0]:.4f}->{train_hist[-1]:.4f}; "
            f"test {test_hist[0]:.4f}->{test_hist[-1]:.4f}"
        )

    print(f"wrote {OUT_TRAIN} ({len(train_rows)} rows, {len(train_groups)} users)")
    print(f"wrote {OUT_TEST} ({len(test_rows)} rows, {len(test_groups)} users)")


if __name__ == "__main__":
    main()
