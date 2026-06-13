#!/usr/bin/env python3
"""训练 multi_output_tree（size_leaf_vector>1）XGBoost 模型与预测黄金值。"""
import json
import numpy as np
import xgboost as xgb

OUT_MODEL = "xgboost_multitarget_vector.json"
OUT_PRED = "xgboost_multitarget_vector_pred.txt"

rng = np.random.default_rng(42)
n = 160
X = rng.normal(size=(n, 3))
Y = np.column_stack([X[:, 0] + 0.2 * X[:, 1], 1.5 * X[:, 2] - 0.3 * X[:, 0]])

params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "multi_strategy": "multi_output_tree",
    "max_depth": 3,
    "eta": 0.35,
    "base_score": 0.0,
}
dtrain = xgb.DMatrix(X, label=Y)
bst = xgb.train(params, dtrain, num_boost_round=5)
bst.save_model(OUT_MODEL)

with open(OUT_MODEL, encoding="utf-8") as f:
    model = json.load(f)
tp = model["learner"]["gradient_booster"]["model"]["trees"][0]["tree_param"]
slv = int(tp.get("size_leaf_vector", "1"))
nt = int(model["learner"]["learner_model_param"].get("num_target", "1"))
if slv < 2 or nt < 2:
    raise SystemExit(f"expected vector leaf model, size_leaf_vector={slv} num_target={nt}")

Xtest = np.array(
    [
        [0.2, -0.1, 0.5],
        [0.0, 0.0, 0.0],
        [-0.4, 0.3, 0.8],
    ],
    dtype=np.float64,
)
dm = xgb.DMatrix(Xtest)
margin = bst.predict(dm, output_margin=True)

with open(OUT_PRED, "w", encoding="utf-8") as f:
    f.write("# sample margin0 margin1 f0 f1 f2\n")
    for i in range(len(Xtest)):
        row = Xtest[i]
        f.write(
            f"{i}\t{margin[i, 0]:.10f}\t{margin[i, 1]:.10f}\t"
            f"{row[0]:.10f}\t{row[1]:.10f}\t{row[2]:.10f}\n"
        )

print("saved", OUT_MODEL, OUT_PRED, "size_leaf_vector=", slv, "num_target=", nt)
