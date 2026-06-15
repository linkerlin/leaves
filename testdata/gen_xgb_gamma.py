#!/usr/bin/env python3
"""训练 reg:gamma JSON 模型，验证 Exp 变换加载。"""
import json
import numpy as np
import xgboost as xgb

OUT_JSON = "xgboost_gamma_smoke.json"
OUT_PRED = "xgboost_gamma_smoke_pred.txt"

rng = np.random.default_rng(0)
n, nf = 300, 6
X = rng.uniform(0.5, 2.0, size=(n, nf))
# gamma-like positive targets
y = rng.gamma(shape=2.0, scale=1.0, size=n) + 0.1 * X[:, 0]

dtrain = xgb.DMatrix(X, label=y)
params = {
    "objective": "reg:gamma",
    "max_depth": 3,
    "eta": 0.2,
    "base_score": 1.0,
}
bst = xgb.train(params, dtrain, num_boost_round=8)
bst.save_model(OUT_JSON)

x0 = np.zeros((1, nf), dtype=np.float64)
pred = float(bst.predict(xgb.DMatrix(x0))[0])
with open(OUT_PRED, "w", encoding="utf-8") as f:
    f.write(f"{pred:.17g}\n")

with open(OUT_JSON, encoding="utf-8") as f:
    doc = json.load(f)
assert doc["learner"]["objective"]["name"] == "reg:gamma"
print("saved", OUT_JSON, "pred0", pred)
