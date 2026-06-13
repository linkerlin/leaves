#!/usr/bin/env python3
"""生成 breast_cancer 训练集与 XGBoost 基准 AUC（T1 验收）。"""
import json
import sys

try:
    import numpy as np
    import xgboost as xgb
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print("requires: numpy sklearn xgboost", file=sys.stderr)
    raise

SEED = 42
X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

out_train = "breast_cancer_train.tsv"
with open(out_train, "w", encoding="utf-8") as f:
    for row, label in zip(X_train, y_train):
        feats = "\t".join(f"{v:.17g}" for v in row)
        f.write(f"{feats}\t{int(label)}\n")

out_test = "breast_cancer_labeled_test.tsv"
with open(out_test, "w", encoding="utf-8") as f:
    for row, label in zip(X_test, y_test):
        feats = "\t".join(f"{v:.17g}" for v in row)
        f.write(f"{feats}\t{int(label)}\n")

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eta": 0.1,
    "lambda": 1.0,
    "seed": SEED,
    "tree_method": "hist",
}
bst = xgb.train(params, dtrain, num_boost_round=50)
pred = bst.predict(dtest)
auc = float(roc_auc_score(y_test, pred))

meta = {
    "seed": SEED,
    "num_round": 50,
    "max_depth": 4,
    "learning_rate": 0.1,
    "lambda": 1.0,
    "tree_method": "hist",
    "train_rows": int(len(y_train)),
    "test_rows": int(len(y_test)),
    "test_auc": auc,
}
with open("breast_cancer_xgb_baseline.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"wrote {out_train} ({len(y_train)} rows)")
print(f"wrote {out_test} ({len(y_test)} rows)")
print(f"test_auc={auc:.6f}")
