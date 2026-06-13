#!/usr/bin/env python3
"""训练 XGBoost 随机森林（num_parallel_tree>1）JSON 模型与预测黄金值。"""
import json
import numpy as np
import xgboost as xgb

OUT_MODEL = "xgboost_rf_smoke.json"
OUT_PRED = "xgboost_rf_smoke_pred.txt"

rng = np.random.default_rng(7)
n = 120
X = rng.normal(size=(n, 4))
y = ((X[:, 0] + 0.5 * X[:, 1] - X[:, 2]) > 0).astype(int)

dtrain = xgb.DMatrix(X, label=y)
params = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 1.0,
    "subsample": 0.8,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 4,
    "base_score": 0.5,
}
bst = xgb.train(params, dtrain, num_boost_round=3)
bst.save_model(OUT_MODEL)

# 验证模型元数据
with open(OUT_MODEL, encoding="utf-8") as f:
    model = json.load(f)
gp = model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]
npt = int(gp.get("num_parallel_tree", "1"))
if npt < 2:
    raise SystemExit(f"expected num_parallel_tree>=2, got {npt}")

Xtest = np.array(
    [
        [0.2, -0.1, 0.3, 0.0],
        [-0.5, 0.8, -0.2, 0.1],
        [1.0, 1.0, -1.0, 0.5],
    ],
    dtype=np.float64,
)
dm = xgb.DMatrix(Xtest)
margin = bst.predict(dm, output_margin=True)
prob = bst.predict(dm)

with open(OUT_PRED, "w", encoding="utf-8") as f:
    f.write("# sample margin prob f0 f1 f2 f3\n")
    for i in range(len(Xtest)):
        row = Xtest[i]
        f.write(
            f"{i}\t{margin[i]:.10f}\t{prob[i]:.10f}\t"
            f"{row[0]:.10f}\t{row[1]:.10f}\t{row[2]:.10f}\t{row[3]:.10f}\n"
        )

print("saved", OUT_MODEL, OUT_PRED, "num_parallel_tree=", npt)
