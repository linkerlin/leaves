---
name: recsys-rank
description: >-
  以 leaves 训练 rank:ndcg/pairwise/listwise 或对标 XGBoost LTR，对召回候选精排。
  凡涉排序模型、LTR、LambdaMART、NDCG 训练、leaves.json 推理时用之。
---

# 排序精排

> 召回既广，排序择精；leaves 为刃，XGB 为尺，NDCG 为衡。

## 一、何时激活

- 用户提及「排序」「精排」「LTR」「XGBoost rank」「NDCG 训练」
- 流水线 Stage 3，输入 `rank/rank_train.tsv`，产出模型与组内 margin
- 依赖 `recsys-recall` 已转好 ranking TSV

## 二、leaves 训练（首选，纯 Go）

### 2.1 加载数据

```go
import (
    "github.com/linkerlin/leaves/data"
    "github.com/linkerlin/leaves/train"
    "github.com/linkerlin/leaves/io"
)

trainDM, _ := data.LoadRankingTSV("recsys/rank/rank_train.tsv", "\t")
testDM,  _ := data.LoadRankingTSV("recsys/rank/rank_test.tsv", "\t")
// 或自动嗅探
trainDM, _ := data.FromFileAuto("recsys/rank/rank_train.tsv")
```

### 2.2 配置 Learner

参照 `demos/movielens/cmd/train/main.go`：

```go
cfg := train.Config{
    Objective:    train.ObjectiveRankNDCG, // rank:pairwise | rank:listwise
    NumRound:     40,
    MaxDepth:     4,
    LearningRate: 0.1,
    Lambda:       1.0,
    TreeMethod:   train.TreeMethodHist, // auto | exact | gpu_hist
    Seed:         42,
    NDCGK:        10,
    EvalMetric:   "ndcg@10",
    // rank:ndcg 默认 topk 配对 k=32 + lambdarank_normalization
}
learner, _ := train.NewLearner(cfg)
_ = learner.Fit(trainDM)
```

**目标选择**：

| Objective | 场景 | XGB 对标 |
|-----------|------|----------|
| `rank:ndcg` | 默认，NDCG 优化 | 有 baseline |
| `rank:pairwise` | 成对偏好 | 有 baseline |
| `rank:listwise` | ListNet | leaves 独有 |

### 2.3 评估

```go
import "github.com/linkerlin/leaves/demos/movielens/rankutil"

trainPred, _ := rankutil.PredictMargins(learner, trainDM)
testPred,  _ := rankutil.PredictMargins(learner, testDM)
trainNDCG, _ := rankutil.NDCGAtK(trainDM, trainPred, 10)
testNDCG,  _ := rankutil.NDCGAtK(testDM, testPred, 10)
```

或 `metrics.Resolve("ndcg@10", metrics.Options{Groups: groups, NDCGK: 10})`。

### 2.4 保存与加载

```go
_ = io.SaveTrainModel("recsys/models/model_rank_ndcg.leaves.json", learner.Model(), cfg.Objective)

m, _ := io.LoadFromFile("recsys/models/model_rank_ndcg.leaves.json",
    &io.LoadOptions{LoadTransformation: false})
defer m.Close()
```

### 2.5 组内推理

参照 `demos/movielens/cmd/recommend/main.go`：

```go
start, count, _ := rankutil.GroupSlice(testDM, groupIdx)
vals := testDM.Data[start*cols : (start+count)*cols]
out := make([]float64, count)
_ = m.PredictDense(vals, count, cols, out, 0, 0)

items, _ := rankutil.RankGroup(testDM, fullPreds, groupIdx, topK)
// items[i].Score 即 LTR margin
```

**须**将 margin 回写 manifest：`User, Item, Tag, Score(margin)`。

## 三、XGBoost 对标（Python）

与 `testdata/gen_rank_smoke.py` 同构，供 baseline JSON：

```python
import xgboost as xgb
import numpy as np

def to_dmatrix(tsv_path, groups):
    # 解析 qid label feat...
    dm = xgb.DMatrix(X, label=y)
    dm.set_group(groups)  # groups = 每 qid 行数，如 [100,100,...]
    return dm

params = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "eta": 0.1, "max_depth": 4, "lambda": 1.0,
    "tree_method": "hist", "seed": 42,
}
bst = xgb.train(params, dtrain, num_boost_round=40,
    evals=[(dtrain,"train"),(dtest,"test")], evals_result=evals_result)
# 导出 baseline JSON，非必须导出 xgb 模型（leaves 自训）
```

**亦可**用 XGBoost 训练后导出 JSON，经 `io.LoadFromFile` 推理（无 rank 专用 golden，需自验）。

## 四、训练加速

```powershell
# 环境变量
$env:LEAVES_TRAIN_ACCEL = "auto"  # cpu | webgpu | auto

# Config
cfg.AccelMode = train.AccelFromEnv()
```

hist 加速：`treebuilder/hist_accel.go`（Born CPU）；Windows WebGPU：`hist_accel_webgpu_*.go`。

## 五、CLI 快捷（仓库内 demo）

```powershell
# 准备 MovieLens 数据
cd testdata && python gen_rank_movielens.py && cd ..

# 训练
go run ./demos/movielens/cmd/train
go run ./demos/movielens/cmd/train -objective rank:pairwise

# 推理 Top-K
go run ./demos/movielens/cmd/recommend -group 3 -topk 10

# CI 对标
go test ./train/... -run TestRankMovieLens -count=1
go test ./train/... -run 'TestRank.*TrendVsXGBoost' -count=1
```

## 六、输出产物

| 文件 | 说明 |
|------|------|
| `models/model_rank_*.leaves.json` | leaves 原生模型 |
| `models/xgb_baseline.json` | XGB NDCG 曲线（可选） |
| `rank/rank_test_scored.jsonl` | 每候选 margin + User/Item/Tag |
| `meta/rank_eval.json` | train/test NDCG@k |

## 七、完成标准

- [ ] test NDCG@10 可报告（或与 XGB baseline Δ 在容忍内）
- [ ] 每 test User 100 候选皆有 margin
- [ ] 模型可 `io.LoadFromFile` 加载并 `PredictDense`
- [ ] 可进入发牌（见 `recsys-deal`）

API 速查见 [`leaves-api.md`](leaves-api.md)。
