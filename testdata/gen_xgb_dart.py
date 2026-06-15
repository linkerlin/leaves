#!/usr/bin/env python3
"""训练 DART JSON 模型。"""
import json
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb

OUT_JSON = "xgboost_dart_smoke.json"
OUT_PRED = "xgboost_dart_smoke_pred.txt"

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    "booster": "dart",
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.2,
    "base_score": 0.5,
    "sample_type": "uniform",
    "normalize_type": "tree",
    "rate_drop": 0.1,
    "skip_drop": 0.5,
}
bst = xgb.train(params, dtrain, num_boost_round=12)
bst.save_model(OUT_JSON)

x0 = np.zeros((1, X.shape[1]), dtype=np.float64)
pred = float(bst.predict(xgb.DMatrix(x0))[0])
with open(OUT_PRED, "w", encoding="utf-8") as f:
    f.write(f"{pred:.17g}\n")

with open(OUT_JSON, encoding="utf-8") as f:
    doc = json.load(f)
assert doc["learner"]["gradient_booster"]["name"] == "dart"
gb = doc["learner"]["gradient_booster"]
assert "weight_drop" in gb or "weight_drop" in gb.get("model", {})
print("saved", OUT_JSON, "pred0", pred)
