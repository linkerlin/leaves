#!/usr/bin/env python3
"""为 xgboost_multiclass_smoke.json 生成概率预测 golden。"""
import numpy as np
import xgboost as xgb

MODEL = "xgboost_multiclass_smoke.json"
OUT = "xgboost_multiclass_smoke_pred.txt"

bst = xgb.Booster()
bst.load_model(MODEL)
x0 = np.zeros((1, 6), dtype=np.float64)
probs = bst.predict(xgb.DMatrix(x0))
with open(OUT, "w", encoding="utf-8") as f:
    for p in probs[0]:
        f.write(f"{p:.17g}\n")
print("saved", OUT, "probs", probs[0])
