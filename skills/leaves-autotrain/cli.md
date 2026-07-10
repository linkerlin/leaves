# leaves CLI 参考

> 入口：`leaves <subcommand> [flags]`。仓库内未安装时用 `go run ./cmd/leaves ...`。
> 所有子命令的指标产物统一走 **metrics.json**，这是 Agent 闭环的唯一信号。

## metrics.json schema

```json
{
  "objective": "reg:squarederror",
  "metric": "rmse",
  "value": 0.1823,
  "maximize": false,
  "n_rows": 1000,
  "n_features": 12,
  "cv_folds": 5,
  "cv_mean": 0.1841,
  "cv_std": 0.0072,
  "fold_metrics": [0.179, 0.181, 0.190, 0.183, 0.188],
  "train_metric": 0.121,
  "best_round": 47,
  "params": {
    "rounds": 50, "depth": 6, "lr": 0.3, "lambda": 1.0,
    "tree_method": "hist", "seed": 42
  }
}
```

| 字段 | 何时出现 | 含义 |
|------|----------|------|
| `value` | 总是 | 主指标（无 `--cv` 时=训练集/val 指标；有 `--cv` 时=`cv_mean`） |
| `maximize` | 总是 | 该指标是否越大越好（Agent 判方向用） |
| `cv_mean`/`cv_std`/`fold_metrics` | 仅 `--cv` | 交叉验证统计 |
| `train_metric` | train 时 | 训练集指标，与 value 对照看过拟合 |
| `params` | train 时 | 本次所用超参（便于 Agent 回溯） |

## runs.jsonl schema（运行账本，每行一条）

由 `train --runs runs.jsonl --tag <名>` 追加；Agent 只读不写。字段为 metrics.json 的紧凑子集 + 标签时间戳：

```json
{"tag":"baseline","ts":"2026-07-09T04:06:20Z","model":"m.leaves.json",
 "metric":"rmse","value":0.236,"maximize":false,"cv_mean":0.241,"cv_std":0.07,
 "params":{"rounds":50,"depth":6,"lr":0.3,"lambda":1.0,"tree_method":"auto","seed":42}}
```

Agent 用法：读全文件 → 按 `maximize` 取 `value` 最优记录 → 其 `params` 作为下一轮起点与发布候选；尾部记录用于判定收敛（连续 3 次改进 <0.5%）。

## sniff

读数据画像，输出推荐 objective/metric + 标签统计。**闭环第一步**：Agent 据此自动选目标，无需用户告知任务类型。

```
leaves sniff --data PATH [--metrics PATH]
```

输出：
```json
{"file":"...","format":"csv","n_rows":1000,"n_cols":12,"n_features":11,
 "has_label":true,
 "label":{"detected":true,"min":0,"max":1,"n_unique":2,"kind":"binary"},
 "suggested_objective":"binary:logistic","suggested_metric":"logloss"}
```

`label.kind`：`binary`（∈{0,1}）/`classification`（整数且 ≤20 类）/`regression`（其他）。排序格式直接推荐 `rank:ndcg`。

## train

在任意数据上训练，输出模型 + metrics.json。

```
leaves train --data PATH [必需]
  --objective NAME        [必需：reg:squarederror|binary:logistic|multi:softmax|
                                  count:poisson|reg:tweedie|survival:cox|
                                  rank:ndcg|rank:pairwise|rank:listwise]
  --eval-metric NAME      [默认按 objective：rmse/logloss/mlogloss/ndcg@10...]
  --num-class N           [多分类必需]
  --rounds N              (默认 50)
  --depth N               (默认 6)
  --max-leaves N          (默认 0=depthwise；>0=lossguide)
  --lr F                  (默认 0.3)
  --lambda F              (默认 1.0)
  --min-child-weight F    (默认 1.0；映射 MinHessian)
  --gamma F               (默认 0)
  --max-bin N             (默认 256)
  --subsample F           (默认 1.0；排序强制 1.0)
  --colsample F           (默认 1.0)
  --tree-method NAME      (hist|exact|auto，默认 auto)
  --ndcg-k N              (排序，默认 10)
  --cv K                  (K 折交叉验证，默认 0=不交叉验证)
  --val PATH              (独立验证集；与 --cv 互斥建议)
  --early-stop N          (N 轮无改进停止)
  --seed N                (默认 42)
  --out-model PATH        (输出 leaves.json；省略则不存)
  --metrics PATH          (输出 metrics.json；省略则写 stdout)
  --runs PATH             (运行账本 JSONL；追加本次记录，Agent 优化记忆)
  --tag NAME              (本次运行标签，写入账本便于回溯)
  --emit-rounds PATH      (逐轮指标 JSONL；Agent 学习曲线诊断，仅单次训练)
```

行为：
- 数据经 `data.FromFileAuto` 自动嗅探（CSV/LIBSVM/ranking TSV）。
- `--cv K`：跑 K 折，`value= cv_mean`；不存模型（除非额外单跑一次）。
- 无 `--cv`：单次 Fit；有 `--val` 时 `value`=val 指标，否则=train 指标。

## eval

加载已存模型，在数据上算指标。用于对比 checkpoint / 选最优。

```
leaves eval --model PATH --data PATH
  --eval-metric NAME      [默认从模型 objective 推断]
  --objective NAME        [覆盖 objective；默认从模型推断，影响 margin→pred 变换]
  --metrics PATH          (输出 metrics.json)
```

## predict

加载模型，输出每行预测（JSONL）。

```
leaves predict --model PATH --data PATH --out PATH
  --format jsonl|csv       (默认 jsonl；csv 出单 "prediction" 列)
  --objective NAME         [默认从模型推断；binary:logistic 时附 probability]
```

`--format jsonl`（默认）：每行 JSON `{"row","margin"[,"probability"]}`
`--format csv`：单 `prediction` 列（部署对接用）

输出每行：`{"row":0,"margin":0.3421,"probability":0.5846}`（二分类才有 probability；回归/排序只有 margin）。

## explain

读模型，输出特征重要性或 SHAP 值（Agent 诊断"模型为什么这样预测"）。

```
leaves explain --model PATH [--type importance|shap] [--data PATH] [--max-rows N] [--metrics PATH]
```

- `--type importance`（默认，无需数据）：输出 per-feature gain 分数。
- `--type shap`（需 `--data`）：每行 SHAP 贡献 `[{row,base,features:[{name,value}]}]`，限 500 行。

## inspect

加载 leaves.json，输出模型元数据 JSON（发布前复核、动态决策依据）。仅支持 leaves.json 原生格式。

```
leaves inspect --model PATH [--metrics PATH]
```

输出：`{"file","objective","kind"（gbtree|dart|gblinear|sklearn_gbdt）,
"num_features","n_output_groups","n_trees","name"[,"feature_names"]}`。

## publish

把模型打成本地工件包。**不推 registry**（leaves 边界）。

```
leaves publish --model PATH --out-dir DIR
  --version NAME          (默认 "1.0.0")
  --quantize              (int8 阈值量化 + parity 报告)
  --export-xgb            (导出 model.xgb.json，leaves→XGBoost 3.x)
  --metrics PATH          (把训练 metrics.json 快照进 manifest)
  --data PATH             (量化 parity 用；省略则跳过 parity)
```

产物：

```
<out-dir>/
  model.leaves.json       主模型
  model.xgb.json          (--export-xgb 时)
  model.quant.json        (--quantize 时：int8 量化侧车，可被 predict 重建)
  quantize_report.json    (--quantize 时：parity vs Native + max_threshold_err)
  manifest.json           {version, objective, num_features, n_trees,
                          base_learners, files:{name,sha256,size},
                          metrics:{...}, created_at}
```

> `model.quant.json` 是**量化侧车**：只存 int8 阈值叠加层 + per-feature min/span，引用 base `model.leaves.json` 的森林。`leaves predict --model x.quant.json` 会自动加载 base+overlay 重建量化推理，margin 在 parity 门禁（默认 0.15）内对齐原模型。

## Agent 工作示例（回归，端到端）

```powershell
# 1) 基线 + CV
go run ./cmd/leaves train --data train.csv --objective reg:squarederror `
  --eval-metric rmse --cv 5 --rounds 50 --depth 6 --lr 0.3 `
  --out-model m.leaves.json --metrics m1.json
# Agent 读 m1.json: value=0.231, train_metric=0.205 → 轻微欠拟合

# 2) §4.2：加深度 + 加轮、降 lr
go run ./cmd/leaves train --data train.csv --objective reg:squarederror `
  --eval-metric rmse --cv 5 --rounds 200 --depth 8 --lr 0.1 --lambda 1.0 `
  --out-model m.leaves.json --metrics m2.json
# m2.json: value=0.198, train_metric=0.151 → 改善，train/val 差拉大→开始过拟合

# 3) 加正则抗过拟合
go run ./cmd/leaves train --data train.csv --objective reg:squarederror `
  --eval-metric rmse --cv 5 --rounds 200 --depth 8 --lr 0.1 `
  --lambda 3.0 --min-child-weight 5 --subsample 0.8 --colsample 0.8 `
  --out-model m.leaves.json --metrics m3.json
# m3.json: value=0.194 → 再改善但 <0.5%，达收敛判据

# 4) holdout 独立验证
go run ./cmd/leaves eval --model m.leaves.json --data holdout.csv `
  --eval-metric rmse --metrics holdout.json

# 5) 发布
go run ./cmd/leaves publish --model m.leaves.json --out-dir release/v1 `
  --version 1.0.0 --export-xgb --metrics m3.json
```

## 退出码

- `0` 成功；`1` 参数/IO 错误；`2` 训练/评估内部错误。
- Agent 据退出码判定失败，据 metrics.json 判定优劣。
