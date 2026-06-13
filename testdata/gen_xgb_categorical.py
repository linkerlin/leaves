#!/usr/bin/env python3
"""训练含原生 categorical 的 XGBoost JSON 模型，供 leaves 端到端测试。"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb

OUT = "xgboost_categorical_smoke.json"

rng = np.random.default_rng(42)
n = 200
f0 = rng.normal(size=n)
cat = rng.integers(0, 4, size=n)
# 简单规则：cat in {0,2} -> 0
y = ((f0 > 0) ^ np.isin(cat, [1, 3])).astype(int)

X = pd.DataFrame({"f0": f0, "f1": pd.Categorical(cat)})
dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
params = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.3,
    "base_score": 0.5,
}
bst = xgb.train(params, dtrain, num_boost_round=5)
bst.save_model(OUT)

# 冒烟：至少一棵树含 categorical split
with open(OUT, encoding="utf-8") as f:
    model = json.load(f)
trees = model["learner"]["gradient_booster"]["model"]["trees"]
has_cat = any(any(st == 1 for st in t.get("split_type", [])) for t in trees)
print("saved", OUT, "has_categorical_split", has_cat)
if not has_cat:
    raise SystemExit("model has no categorical splits")
