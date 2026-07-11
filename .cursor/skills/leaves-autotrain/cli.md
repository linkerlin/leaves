# leaves CLI 参考

> 入口：`leaves [--error-format text|json] <subcommand> [flags]`。仓库内未安装时用 `go run ./cmd/leaves ...`。
> 所有子命令的指标产物统一走 **metrics.json**，这是 Agent 闭环的唯一信号。
> 全局错误格式：`--error-format json` 或环境变量 `LEAVES_ERROR_FORMAT=json`（Agent 解析用）。

## metrics.json schema

```json
{
  "schema_version": 1,
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
    "min_child_weight": 1.0, "gamma": 0, "max_bin": 256,
    "subsample": 1.0, "colsample": 1.0,
    "tree_method": "hist", "seed": 42,
    "eval_metric": "rmse"
  }
}
```

| 字段 | 何时出现 | 含义 |
|------|----------|------|
| `schema_version` | 总是 | 契约版本（当前 `1`；破坏性变更时递增） |
| `value` | 总是 | 主指标（无 `--cv` 时=训练集/val 指标；有 `--cv` 时=`cv_mean`） |
| `maximize` | 总是 | 该指标是否越大越好（Agent 判方向用） |
| `n_features` | train 时 | 特征维（= `Matrix.NumCol()`；标签/qid 不计） |
| `cv_mean`/`cv_std`/`fold_metrics` | 仅 `--cv` | 交叉验证统计 |
| `train_metric` | train 时 | 训练集指标，与 value 对照看过拟合 |
| `params` | train 时 | **全部** CLI 旋钮（Agent 从 runs 复现最优；见下表） |
| `train_accel` | 单次 Fit 后 | 实际训练加速模式（`cpu`/`born_cpu`/`webgpu` 等）；**与推理 BackendAuto 无关**；CV-only 未存模型时可能缺省 |
| `final_model` / `final_round` | 使用 `--out-final` 时 | final-round 侧车路径与树轮数（早停截断前） |

`params` 完备字段：`rounds`, `depth`, `max_leaves`, `lr`, `lambda`, `min_child_weight`, `gamma`, `max_bin`, `subsample`, `colsample`, `tree_method`, `seed`, `eval_metric`；以及按需 `num_class`, `num_target`, `ndcg_k`, `early_stop`, `cv_folds`。

## runs.jsonl schema（运行账本，每行一条）

由 `train --runs runs.jsonl --tag <名>` 追加；Agent 只读不写。字段为 metrics.json 的紧凑子集 + 标签时间戳：

```json
{"tag":"baseline","ts":"2026-07-09T04:06:20Z","model":"m.leaves.json",
 "objective":"reg:squarederror",
 "metric":"rmse","value":0.236,"maximize":false,"cv_mean":0.241,"cv_std":0.07,
 "params":{"rounds":50,"depth":6,"lr":0.3,"lambda":1.0,"min_child_weight":1.0,
  "gamma":0,"max_bin":256,"subsample":1.0,"colsample":1.0,
  "tree_method":"auto","seed":42,"eval_metric":"rmse"}}
```

Agent 用法：
- 读全文件 → 按 `maximize` 取 `value` 最优记录 → 其 **完整** `params` 作为下一轮起点与发布候选。
- **一键复现（WP-17）**：`leaves train --data PATH --from-run runs.jsonl [--tag NAME] [覆盖 flags]`  
  - 有 `--tag`：取该 tag **最后一次**出现的行；无 `--tag`：按 maximize 自动选最优行。  
  - 账本 `params` + `objective` 填默认；**CLI 显式 flag 始终优先**。  
  - 仍须提供 `--data`（及需要时的 `--val`）；`--objective` 可省略（由账本补全）。
- 尾部记录用于判定收敛（连续 3 次改进 <0.5%）。

## sniff

读数据画像，输出推荐 objective/metric + 标签统计。**闭环第一步**：Agent 据此自动选目标，无需用户告知任务类型。

```
leaves sniff --data PATH [--metrics PATH] [--na-policy error|skip-row]
```

输出：
```json
{"file":"...","format":"csv","n_rows":1000,"n_cols":11,"n_features":11,
 "has_label":true,"has_qid":false,"feature_names":["f0","f1", "..."],
 "label":{"detected":true,"min":0,"max":1,"n_unique":2,"kind":"binary"},
 "suggested_objective":"binary:logistic","suggested_metric":"logloss"}
```

- `n_features` / `n_cols`：均为 **特征维**（= `Matrix.NumCol()`）。标签在 `Labels()`，排序 qid 在 Groups，**不计入**。
- `has_qid`：排序 TSV 为 true。
- 若有表头：`feature_names` 长度必须等于 `n_features`。
- `data_quality`：轻量质量报告（**不改数据**；Agent 预知风险用）

```json
"data_quality": {
  "numeric": true,
  "scanned_rows": 120,
  "total_rows": 120,
  "nan_cells": 0,
  "inf_cells": 0,
  "constant_features": ["x1"],
  "label_pos_ratio": 0.48,
  "warnings": ["constant_features=1", "small_n_rows"]
}
```

**扫描语义（非全量承诺）**：
- 行扫描上限 **`sniffMaxScan = 5000`**（实现常量）。`scanned_rows` ≤ `total_rows`；超过上限时常数列 / nan 统计仅基于前 5000 行。
- 大文件：`warnings` 可能未覆盖尾部异常；全量质量检查须在库外做。
- `skip-row` 时另见顶层 / `data_quality.skipped_rows`。

`label.kind`：`binary`（∈{0,1}）/`classification`（整数且 ≤20 类）/`regression`（其他）。排序格式直接推荐 `rank:ndcg`。

## 错误码（`--error-format json` / `LEAVES_ERROR_FORMAT=json`）

stderr JSON 形状：

```json
{"error":"usage","message":"...","hint":"...","retryable":false,"exit_code":1}
```

| `error` | exit | 场景 | Agent 建议 |
|---------|------|------|------------|
| `usage` | 1 | 缺 flag / 未知子命令 / 路径纪律 | 修 CLI，不重试同命令 |
| `data_load` | 1 | 路径不存在、格式嗅探/解析失败 | 查路径与格式 |
| `non_numeric` | 1 | 类别字符串等 | 库外 one-hot / label encode |
| `missing_value` | 1 | 空单元格 / NA（默认 na-policy） | 填补，或 `--na-policy skip-row` |
| `objective_mismatch` | 1 | multi:* 无 `--num-class` | sniff 取 `label.n_unique` |
| `model_load` | 1 | 模型路径/格式 | 用 leaves.json 或支持的 XGB/LGB |
| `cv_conflict` | 1 | `--strict-flags` 且 cv 与 val/early-stop/emit-rounds 并存 | 去掉一侧 flag |
| `internal` | 2 | 训练/评估内部 | 可降 rounds/depth 重试 |

默认 `--error-format text`：人类可读 + 可选 `hint:` 行；exit 语义相同。

## train

在任意数据上训练，输出模型 + metrics.json。

```
leaves train --data PATH [必需]
  --objective NAME        [必需：reg:squarederror|binary:logistic|multi:softmax|
                                  count:poisson|reg:tweedie|survival:cox|
                                  rank:ndcg|rank:pairwise|rank:listwise]
  --eval-metric NAME      [默认按 objective：rmse/logloss/mlogloss/ndcg@10...]
  --num-class N           [多分类必需]
  --num-target N          [多目标回归 ≥2；CSV 末 N 列为标签，one_output_per_tree]
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
  --save-best             (默认 true：早停后内存截断到 best_round 再存盘；false=保留 final-round)
  --out-final PATH        (可选：早停截断前另存 final-round；metrics 写 final_model/final_round)
  --strict-flags          (默认 false：--cv 与 --val/--early-stop/--emit-rounds 并存仅警告；true=失败 error=cv_conflict)
  --seed N                (默认 42)
  --out-model PATH        (输出 leaves.json；早停+save-best 时为 best；省略则不存)
  --metrics PATH          (输出 metrics.json；省略则写 stdout)
  --runs PATH             (运行账本 JSONL；追加本次记录，Agent 优化记忆)
  --tag NAME              (本次运行标签；与 --from-run 联用时兼作选行键)
  --from-run PATH         (从 runs.jsonl 加载 params 作默认；CLI 覆盖优先)
  --na-policy error|skip-row  (默认 error：空/NA 单元格失败；skip-row=丢弃整行，不做插补)
  --emit-rounds PATH      (逐轮指标 JSONL；Agent 学习曲线诊断，仅单次训练)
```

行为：
- 数据经 `data.FromFileAuto` 自动嗅探（CSV/LIBSVM/ranking TSV）。
- **多目标回归**：`--num-target N`（N≥2）时 CSV/TSV **末 N 列**为标签、其余为特征；objective 用 `reg:squarederror` 等（勿与 multi:/rank: 混用）；训练策略 **one_output_per_tree**（每轮 N 棵标量树）。
- `--cv K`：跑 K 折，`value= cv_mean`；不存模型（除非额外单跑一次）。
- **与 `--val` / `--early-stop` / `--emit-rounds` 并存**：cv 路径会忽略后者。默认仅 stderr 警告；Agent 建议加 **`--strict-flags`** 得到 `error=cv_conflict`（exit 1）。
- 无 `--cv`：单次 Fit；有 `--val` 时 `value`=val 指标，否则=train 指标。
- **早停定稿（默认）**：`--early-stop` + `--save-best`（默认 true）时，`--out-model` 为 **best_round** 轮，metrics 含 `best_round` / `stopped_round` / `model_round`（`model_round == best_round`）。
  - **实现是内存截断**（`Learner.ApplyBestRound` → `ForestIR.TruncateToNEstimators`），**不是**按 `best_round` 二次全量重训；成本 ≈ 存盘前 O(树数) 截断，无额外 boosting 轮。
  - 研究 final 轨迹：`--save-best=false`，或 **POST-12** 同时 `--out-final PATH`（截断前写出 final；metrics.`final_model` / `final_round`）。
  - `--out-final` 不得与 `--out-model` / `--metrics` 同路径。
- **`--from-run`**：见上 runs 节；定稿复现时优先用此路径，少拼 flag。
- **路径纪律**：`--out-model` 与 `--metrics` 不得为同一路径（CLI 会拒绝；否则 metrics 覆盖模型）。
- **`--na-policy`**：仅 CSV；缺失 = 空单元格或 `nan/na/null/none/n/a/?`。**默认 error**（不静默改分布）；`skip-row` 丢行并在 stderr 报告跳过行数。非数值类别仍失败（须预编码）。

## eval

加载已存模型，在数据上算指标。用于对比 checkpoint / 选最优。

```
leaves eval --model PATH --data PATH
  --eval-metric NAME      [默认从模型 objective 推断]
  --objective NAME        [覆盖 objective；默认从模型推断，影响 margin→pred 变换]
  --num-target N          [多目标：CSV 末 N 列标签；也可从模型 n_output_groups 推断]
  --metrics PATH          (输出 metrics.json)
  --na-policy error|skip-row  (默认 error)
```

多目标时 labels 为扁平 `n×N` 与预测对齐，RMSE/MAE 在全部目标元素上计算。

## predict

加载模型，输出每行预测（JSONL）。

```
leaves predict --model PATH --data PATH --out PATH
  --format jsonl|csv       (默认 jsonl；csv：单输出一列 / 多分类 argmax / 多目标 pred_0..k)
  --objective NAME         [默认从模型推断；binary:logistic 时附 probability]
  --na-policy error|skip-row  (默认 error)
```

- 单输出：`margin`（及 logistic 的 `probability`）。
- **多分类**（`multi:softmax|softprob`）：`class` + `probabilities`。
- **多目标回归**（`n_output_groups>1` 且非 multi:*）：`margins` / `predictions` 数组，**不做 softmax**。

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
  --metrics PATH          (把训练 metrics.json 快照进 manifest；reproduce 脚本必需)
  --data PATH             (量化 parity 用；省略则跳过 parity)
  --emit-repro-script ps1|sh|both  (写出 reproduce.ps1 / reproduce.sh；需 --metrics)
  --print-repro                    (stdout 打印一条完整 leaves train 复现命令；需 --metrics)
```

产物：

```
<out-dir>/
  model.leaves.json       主模型
  model.xgb.json          (--export-xgb 时)
  model.quant.json        (--quantize 时：int8 量化侧车，可被 predict 重建)
  quantize_report.json    (--quantize 时：parity vs Native + max_threshold_err)
  reproduce.ps1 / .sh     (--emit-repro-script 时)
  manifest.json           见下
```

`manifest.json` 关键字段：

| 字段 | 含义 |
|------|------|
| `version` | 用户传入的工件版本 |
| `objective` / `num_features` / `n_trees` / `n_boost_rounds` | 模型元数据 |
| `files[]` | `{name, sha256, size}` |
| `metrics` | 训练 metrics 快照（若 `--metrics`） |
| `reproduce` | 由 metrics.params 拼出的 `leaves train ...` 复现命令 |
| `schema_version` | manifest 契约版本（当前 1） |
| `leaves_cli` | CLI 契约锚点（当前 `agentic-1`） |
| `publish_note` | 明确：本地包；registry 推送在库外 CI |
| `created_at` | UTC 时间 |

> `model.quant.json` 是**量化侧车**：只存 int8 阈值叠加层 + per-feature min/span，引用 base `model.leaves.json` 的森林。`leaves predict --model x.quant.json` 会自动加载 base+overlay 重建量化推理，margin 在 parity 门禁（默认 0.15）内对齐原模型。

### 对接 registry（库外模板，LIB-31）

`leaves publish` **只**产出本地目录；推送由用户 CI 完成。常见模板：

```powershell
# --- 0) 本地发布 ---
go run ./cmd/leaves publish --model m.leaves.json --out-dir release/v1 `
  --version 1.0.0 --metrics m.json --data train.csv `
  --export-xgb --emit-repro-script both --print-repro

# --- 1) GitHub Release 附件 ---
gh release create v1.0.0 release/v1/* --title "model v1.0.0" --notes-file notes.md
# 或追加到已有 release：
# gh release upload v1.0.0 release/v1/* --clobber

# --- 2) S3 / 兼容对象存储 ---
aws s3 sync release/v1 s3://models/my-task/v1.0.0/ --delete
# 可选：manifest 旁写最新指针
# echo v1.0.0 | aws s3 cp - s3://models/my-task/latest.txt

# --- 3) 内部 HTTP registry（curl 示意）---
# $MAN = Get-Content release/v1/manifest.json -Raw
# curl -X PUT "https://registry.example/models/my-task/v1.0.0" `
#   -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d $MAN
# foreach ($f in Get-ChildItem release/v1 -File) {
#   curl -X PUT "https://registry.example/models/my-task/v1.0.0/files/$($f.Name)" `
#     -H "Authorization: Bearer $TOKEN" --data-binary "@$($f.FullName)"
# }

# --- 4) OCI / ORAS（可选，镜像仓库存 artifact）---
# oras push ghcr.io/org/models/my-task:v1.0.0 ./release/v1/
```

**Agent 纪律**：推送成功与否不由 leaves 退出码表示；先确认 `manifest.json` 的 `files[].sha256`，再跑库外脚本。

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

## 退出码与错误 JSON

- `0` 成功；`1` 参数/用法/数据/模型加载；`2` 训练/评估内部错误。
- Agent 据退出码判定失败，据 metrics.json 判定优劣。

`--error-format json`（或 `LEAVES_ERROR_FORMAT=json`）时 stderr 示例：

```json
{
  "error": "usage",
  "message": "--data 与 --objective 必需",
  "hint": "检查子命令必需 flag；详见 leaves --help 或 skills/leaves-autotrain/cli.md",
  "retryable": false,
  "exit_code": 1
}
```

常见 `error` code：`usage` | `data_load` | `non_numeric` | `missing_value` | `objective_mismatch` | `model_load` | `internal`。
