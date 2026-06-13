#!/usr/bin/env python3
"""生成 testdata/shap_contribs_multiclass_smoke.tsv（XGBoost 多类 pred_contribs 黄金值）。"""
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

MODEL = "xgboost_multiclass_smoke.json"
OUT = "shap_contribs_multiclass_smoke.tsv"

X, y = make_classification(
    n_samples=200, n_features=6, n_informative=4, n_classes=3, random_state=42
)
bst = xgb.Booster()
bst.load_model(MODEL)

X2 = X[:2]
dm = xgb.DMatrix(X2)
margin = bst.predict(dm, output_margin=True)
contrib = bst.predict(dm, pred_contribs=True)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("# XGBoost multi:softprob pred_contribs golden\n")
    f.write(
        f"# n_samples={contrib.shape[0]} n_groups={contrib.shape[1]} "
        f"n_cols={contrib.shape[2]} n_features={contrib.shape[2]-1}\n"
    )
    for si in range(contrib.shape[0]):
        for k in range(contrib.shape[1]):
            f.write(f"# margin {si} {k} {margin[si, k]:.10f}\n")
            for fi in range(contrib.shape[2]):
                f.write(f"{si}\t{k}\t{fi}\t{contrib[si, k, fi]:.10f}\n")

print("wrote", OUT)
