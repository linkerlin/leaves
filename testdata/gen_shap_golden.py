#!/usr/bin/env python3
"""生成 testdata/shap_contribs_smoke.tsv（XGBoost pred_contribs 黄金值）。"""
import numpy as np
import xgboost as xgb

MODEL = "xgboost_smoke.json"
OUT = "shap_contribs_smoke.tsv"

bst = xgb.Booster()
bst.load_model(MODEL)

X = np.zeros((3, 8), dtype=np.float64)
X[0, 1] = 0.7
X[0, 5] = 0.6
X[1, 0] = 0.3
X[1, 2] = 0.8
X[2, 3] = 0.5

dm = xgb.DMatrix(X)
margin = bst.predict(dm, output_margin=True)
contrib = bst.predict(dm, pred_contribs=True)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("# sample feature value  (feature=n_features is bias term)\n")
    f.write(f"# n_samples={contrib.shape[0]} n_cols={contrib.shape[1]} n_features=8\n")
    for i in range(contrib.shape[0]):
        f.write(f"# margin {i} {margin[i]:.10f}\n")
        for j in range(contrib.shape[1]):
            f.write(f"{i}\t{j}\t{contrib[i, j]:.10f}\n")

print("wrote", OUT)
