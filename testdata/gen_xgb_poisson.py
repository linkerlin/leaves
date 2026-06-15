#!/usr/bin/env python3
"""训练 count:poisson JSON 模型。"""
import json
import numpy as np
import xgboost as xgb

OUT_JSON = "xgboost_poisson_smoke.json"
OUT_PRED = "xgboost_poisson_smoke_pred.txt"

rng = np.random.default_rng(1)
n, nf = 400, 5
X = rng.uniform(0, 1, size=(n, nf))
y = rng.poisson(lam=2.0 + X[:, 0], size=n).astype(np.float64)

dtrain = xgb.DMatrix(X, label=y)
params = {
    "objective": "count:poisson",
    "max_depth": 4,
    "eta": 0.15,
    "base_score": 2.0,
}
bst = xgb.train(params, dtrain, num_boost_round=10)
bst.save_model(OUT_JSON)

x0 = np.zeros((1, nf), dtype=np.float64)
pred = float(bst.predict(xgb.DMatrix(x0))[0])
with open(OUT_PRED, "w", encoding="utf-8") as f:
    f.write(f"{pred:.17g}\n")
print("saved", OUT_JSON, "pred0", pred)
