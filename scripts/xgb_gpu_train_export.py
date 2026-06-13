#!/usr/bin/env python3
"""Windows 原生 CUDA：XGBoost GPU 训练并导出 JSON，供 leaves 加载推理。"""
import argparse
import json
import sys

import numpy as np
import xgboost as xgb


def main() -> int:
    ap = argparse.ArgumentParser(description="XGBoost GPU train → JSON export")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--cols", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--device", default="cuda:0", help="cuda:0 或 cpu")
    ap.add_argument("--out", default="model_gpu.json")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    x = rng.standard_normal((args.rows, args.cols))
    y = (x[:, 0] * 0.5 + x[:, 1] * 0.3 + rng.standard_normal(args.rows) * 0.1 > 0).astype(float)

    dtrain = xgb.DMatrix(x, label=y)
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": args.device,
        "max_depth": 6,
        "eta": 0.3,
        "eval_metric": "logloss",
    }
    print(f"training device={args.device} rows={args.rows} cols={args.cols} rounds={args.rounds}")
    booster = xgb.train(params, dtrain, num_boost_round=args.rounds)
    booster.save_model(args.out)
    print(f"saved {args.out}")

    # 快速自检：JSON 可读
    with open(args.out, encoding="utf-8") as f:
        doc = json.load(f)
    if "learner" not in doc:
        print("warning: unexpected json shape", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
