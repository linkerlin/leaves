#!/usr/bin/env python3
"""训练 XGBoost gblinear JSON 模型，供 leaves 互操作测试。"""
import json
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb

OUT_JSON = "xgboost_gblinear_smoke.json"
OUT_PRED = "xgboost_gblinear_smoke_pred.txt"

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    "booster": "gblinear",
    "objective": "binary:logistic",
    "eta": 0.1,
    "base_score": 0.5,
}
bst = xgb.train(params, dtrain, num_boost_round=10)
bst.save_model(OUT_JSON)

x0 = np.zeros((1, X.shape[1]), dtype=np.float64)
pred = float(bst.predict(xgb.DMatrix(x0))[0])
with open(OUT_PRED, "w", encoding="utf-8") as f:
    f.write(f"{pred:.17g}\n")

with open(OUT_JSON, encoding="utf-8") as f:
    doc = json.load(f)
name = doc["learner"]["gradient_booster"]["name"]
assert name == "gblinear", name
weights = doc["learner"]["gradient_booster"]["model"]["weights"]
print("saved", OUT_JSON, "weights", len(weights), "pred0", pred)
