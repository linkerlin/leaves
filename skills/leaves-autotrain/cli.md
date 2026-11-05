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
| `n_trees` / `elapsed_ms` | train 时 | 模型树数与训练耗时 ms（EVO-02；「指标 vs 大小/耗时」权衡用；未存模型时省略） |
| `params` | train 时 | **全部** CLI 旋钮（Agent 从 runs 复现最优；见下表） |
| `train_accel` | 单次 Fit 后 | 实际训练加速模式（`cpu`/`born_cpu`/`webgpu` 等）；**与推理 BackendAuto 无关**；CV-only 未存模型时可能缺省 |
| `final_model` / `final_round` | 使用 `--out-final` 时 | final-round 侧车路径与树轮数（早停截断前） |

`params` 完备字段：`rounds`, `depth`, `max_leaves`, `lr`, `lambda`, `min_child_weight`, `gamma`, `max_bin`, `subsample`, `colsample`, `tree_method`, `seed`, `eval_metric`；以及按需 `num_class`, `num_target`, `ndcg_k`, `early_stop`, `cv_folds`, `val`（仅单跑路径记录；早停 run 忠实复现用）。

> **复现语义分工**：`manifest.reproduce` = 忠实重放该 run（CV 行带 `--cv K`，早停行带 `--val X --early-stop N`）；`--from-run` = 定稿/变异起点（**不回填 val**，默认全量重训；要早停路径显式传 `--val`）。

## runs.jsonl schema（运行账本，每行一条）

由 `train --runs runs.jsonl --tag <名>` 追加；Agent 只读不写。字段为 metrics.json 的紧凑子集 + 标签时间戳：

```json
{"tag":"baseline","ts":"2026-07-09T04:06:20Z","model":"m.leaves.json",
 "objective":"reg:squarederror",
 "metric":"rmse","value":0.236,"maximize":false,"cv_mean":0.241,"cv_std":0.07,
 "fold_metrics":[0.23,0.25],"n_trees":50,"elapsed_ms":812,
 "params":{"rounds":50,"depth":6,"lr":0.3,"lambda":1.0,"min_child_weight":1.0,
  "gamma":0,"max_bin":256,"subsample":1.0,"colsample":1.0,
  "tree_method":"auto","seed":42,"eval_metric":"rmse"}}
```

`fold_metrics`（EVO-02，仅 `--cv`）：各折指标，供折级 Pareto 选父；`n_trees`/`elapsed_ms`：模型树数与耗时（未存模型时省略）。

Agent 用法：
- 读全文件 → 按 `maximize` 取 `value` 最优记录 → 其 **完整** `params` 作为下一轮起点与发布候选。
- 演化搜索（SKILL §4.5）：`fold_metrics` 找「某折最优」的非支配集加权选父；`n_trees`/`elapsed_ms` 做大小/耗时权衡与筛选晋级。
- **一键复现（WP-17）**：`leaves train --data PATH --from-run runs.jsonl [--tag NAME] [覆盖 flags]`  
  - 有 `--tag`：取该 tag **最后一次**出现的行；**tag 不在账本时回落最优行**（stderr 注明；支持谱系流程「复现最优 + `--tag p:parent+mutation` 起新名」）。无 `--tag`：按 maximize 自动选最优行（新 tag 自动取 `<最优tag>_repro`）。  
  - 账本 `params` + `objective` 填默认；**CLI 显式 flag 始终优先**。  
  - 仍须提供 `--data`（及需要时的 `--val`）；`--objective` 可省略（由账本补全）。  
- 尾部记录用于判定收敛（连续 3 次改进 <0.5%）。

## sniff

读数据画像，输出推荐 objective/metric + 标签统计。**闭环第一步**：Agent 据此自动选目标，无需用户告知任务类型。

```
leaves sniff --data PATH [--metrics PATH] [--na-policy error|skip-row]
  [--label-col NAME] [--drop-cols C1,C2]
```

输出：
```json
{"file":"...","format":"csv","n_rows":1000,"n_cols":11,"n_features":11,
 "has_label":true,"has_qid":false,"feature_names":["f0","f1", "..."],
 "id_like_columns":["user_id"],"suggested_drop_cols":["user_id"],
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
| `missing_label` | 1 | 表头未命中默认标签名且未传 `--label-col` | 显式传 `--label-col <列名>` 或 `--num-target` |
| `feature_mismatch` | 1 | 评估/预测数据表头缺失模型所需的特征列，或无表头维数不符 | 检查数据表头列是否完整包含模型 `feature_names` |
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
  --na-policy error|skip-row|nan (默认 error；skip-row=丢弃整行；nan=保留特征缺失为 NaN 走树默认方向)
  --auto-encode           (自动将 CSV 非数值字符串特征列按字典序编码，并随模型持久化 category_maps)
  --label-col NAME        (显式指定标签列名或 0-based 索引；防末列误退化为特征)
  --drop-cols C1,C2       (显式剔除 ID/时间戳等非特征列，逗号分隔)
  --weight-col NAME       (样本权重列名或 0-based 索引，自动从特征列剥离并传入损失函数，AG-APEX)
  --scale-pos-weight F    (二分类正样本梯度/Hessian 权重乘子，默认 1.0，AG-APEX)
  --tweedie-variance-power F (reg:tweedie 方差幂参数 (1,2)，默认 1.5，AG-APEX)
  --multi-output-tree     (多分类/多目标使用向量叶单树，AG-APEX)
  --num-parallel-tree N   (每轮并行随机森林树数，默认 1，AG-APEX)
  --booster NAME          (gbtree|dart|gblinear，默认 gbtree)
  --rate-drop F           (DART 丢弃率，默认 0.1)
  --skip-drop F           (DART 跳过丢弃概率，默认 0.0)
  --monotone C1,C2,...    (单调约束，按特征列填 1/-1/0)
  --checkpoint PATH       (训练中途 checkpoint 路径)
  --checkpoint-every N    (每 N 轮保存一次 checkpoint)
  --resume PATH           (从已有模型/checkpoint 断点续训)
  --batch-rows N          (外存/分块流式直方图训练批大小)
  --lr-decay F            (每轮学习率衰减乘子，默认 1.0)
  --dry-run               (毫秒级预检：校验数据/参数/冲突，输出预检 JSON，不执行训练)
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
- **`--na-policy`**：仅 CSV；缺失 = 空单元格或 `nan/na/null/none/n/a/?`。**默认 error**（不静默改分布）；`skip-row` 丢行并在 stderr 报告跳过行数；`nan` 保留特征缺失为 `NaN`（走 GBDT 默认方向路由）。非数值类别列默认报 `non_numeric`，开启 `--auto-encode` 时自动按字典序编码为整数并持久化 `category_maps` 至 `leaves.json`（下游 `eval`/`predict`/`explain`/`compare`/`publish` 自动复用映射，OOV 映射为 `NaN`）。
- **二分类全息诊断（`binary_diagnostics`）**：`objective=binary:logistic` 时，`train` 与 `eval` 自动在 `metrics.json` 中输出 `binary_diagnostics`（含 `auc`、`accuracy_at_05`、`f1_at_05`、最优 F1 阈值 `best_threshold`、`best_f1`、`best_precision`、`best_recall`、`best_accuracy` 与 `confusion_matrix`）。
- **回归全息诊断（`regression_diagnostics`，AG-OMNI）**：单目标回归任务时，`train` 与 `eval` 自动在 `metrics.json` 中输出 `regression_diagnostics`（含 `r2`、`mae`、`rmse`、`medae`、`p90_abs_error`、`max_abs_error`、`mean_residual` 与 `outlier_ratio`）。
- **多分类全息诊断（`multiclass_diagnostics`，AG-OMNI）**：`multi:softmax|softprob` 时，`train` 与 `eval` 自动在 `metrics.json` 中输出 `multiclass_diagnostics`（含 `accuracy`、`macro_f1`、`macro_precision`、`macro_recall`、`per_class_f1` 与 $K\times K$ `confusion_matrix`）。
- **排序学习全息诊断（`ranking_diagnostics`，AG-APEX）**：`rank:ndcg|pairwise|listwise` 时，`train` 与 `eval` 自动在 `metrics.json` 中输出 `ranking_diagnostics`（含 `n_queries`、`mean_group_size`、`min_group_size`、`max_group_size`、`zero_relevant_queries`、`ndcg_at_1`、`ndcg_at_3`、`ndcg_at_5`、`ndcg_at_10`、`map` 与 `mrr`）。
- **模型特征 Schema 自动投影对齐（AG-APEX）**：当模型持久化了语义 `feature_names` 时，下游 `train --val`、`eval`、`predict`、`explain`、`compare` 与 `publish` 会自动按模型 `feature_names` 跳过多余列（免去重复传 `--drop-cols`）并重排列顺序；若数据缺少模型所需的任何特征列，立即报 `error=feature_mismatch`。

## autotune（有界预算反射式演化自治闭环，AG-APEX）

在单个有界命令内自动串联 `sniff`（脏数据自适应：自动推荐 `objective`/`metric`/`num_class`、自动开启 `--auto-encode` 与 `--na-policy nan`、自动剔除 ID/常数列、自动计算二分类 `--scale-pos-weight`）→ 第 1 轮 `baseline` → 第 $2 \dots B$ 轮 `diagnose` + `runs propose`/`crossover` 防重演化 → 最优模型定稿与谱系报告。

```
leaves autotune --data PATH [--val PATH] [--cv K] [--budget 4] [--rounds 30]
  [--objective NAME] [--eval-metric NAME] [--num-class N] [--num-target N]
  [--label-col NAME] [--drop-cols C1,C2] [--weight-col NAME]
  [--out-dir DIR] [--out-model PATH] [--metrics autotune_report.json] [--seed 42]
```

## eval

加载已存模型，在数据上算指标。用于对比 checkpoint / 选最优。

```
leaves eval --model PATH --data PATH
  --eval-metric NAME      [默认从模型 objective 推断]
  --objective NAME        [覆盖 objective；默认从模型推断，影响 margin→pred 变换]
  --num-target N          [多目标：CSV 末 N 列标签；也可从模型 n_output_groups 推断]
  --label-col NAME        [显式指定标签列名或 0-based 索引]
  --drop-cols C1,C2       [显式剔除冗余列]
  --auto-encode           [若模型已含 category_maps 则自动复用]
  --metrics PATH          (输出 metrics.json)
  --na-policy error|skip-row|nan  (默认 error；模型含 category_maps 时自动允许 OOV NaN)
```

多目标时 labels 为扁平 `n×N` 与预测对齐，RMSE/MAE 在全部目标元素上计算。

## compare（多模型同台对比与配对 Bootstrap 显著性检验，AG-DEEP）

在同一验证/测试集上评测 2 个及以上模型（支持 leaves.json / XGB / LGB / ONNX / sklearn），自动按最优排序，计算相对基线的 `delta_pct` 与确定性配对 Bootstrap 胜率及 95% 置信区间（`significant: true|false`）。

```
leaves compare --models m1.leaves.json,m2.leaves.json --data holdout.csv
  [--eval-metric NAME] [--bootstrap 200] [--seed 42]
  [--label-col NAME] [--drop-cols C1,C2] [--na-policy error|skip-row|nan] [--auto-encode]
  [--metrics compare.json]
```

## predict

加载模型，输出每行预测（JSONL 或 CSV）并可选输出推理统计摘要（`--metrics`）。

```
leaves predict --model PATH --data PATH --out PATH
  --format jsonl|csv       (默认 jsonl；csv：单输出一列 prediction / 多分类 argmax / 多目标 pred_0..k)
  --output value|margin|leaf|contrib  (默认 value；value 自动应用目标函数 Link 逆变换；leaf=叶节点索引；contrib=逐行 TreeSHAP 贡献)
  --threshold F            (二分类判定阈值，如 0.42；设置后输出 predicted_label: 0|1)
  --id-col NAME            (透传样本主键/ID 列到输出 JSONL/CSV，并自动从特征矩阵剔除)
  --label-col NAME         (若测试集含自定义标签列名，显式指定以从特征中剥离)
  --drop-cols C1,C2        (显式剔除冗余列)
  --auto-encode            (若模型含 category_maps 则自动复用)
  --objective NAME         [默认从模型推断；binary:logistic 自动 sigmoid，count:poisson/reg:gamma/reg:tweedie 自动 exp]
  --na-policy error|skip-row|nan  (默认 error)
  --metrics PATH           (可选输出模型推理统计摘要 JSON：pred_min/pred_max/pred_mean/pred_std/positive_rate)
```

- 单输出（`--output value`）：输出 `margin` 与变换后的 `prediction`（`binary:logistic` 附 `probability` 与 `predicted_label`；`count:poisson`/`reg:gamma`/`reg:tweedie` 的 `prediction = exp(margin)`）。
- **多分类**（`multi:softmax|softprob`）：`class` + `probabilities`。
- **多目标回归**（`n_output_groups>1` 且非 multi:*）：`margins` / `predictions` 数组，**不做 softmax**。

`--format jsonl`（默认）：每行 JSON `{"row","margin","prediction"[,"probability","predicted_label"]}`
`--format csv`：单 `prediction` 列（`--output value` 时已应用 Link 逆变换，部署对接用）

## explain

读模型（支持 leaves.json / XGBoost / LightGBM / ONNX / sklearn），输出特征重要性、SHAP 值（逐行或全局汇总）、二阶特征交互矩阵或可读树结构转储。

```
leaves explain --model PATH [--type importance|shap|shap-summary|interaction|dump]
  [--importance-type gain|weight|cover|total_gain|total_cover]
  [--summary] [--dump-format json|text|dot]
  [--data PATH] [--max-rows N] [--metrics PATH]
  [--label-col NAME] [--drop-cols C1,C2]
```

- `--type importance`（默认，无需数据）：支持 `--importance-type gain|weight|cover|total_gain|total_cover`，输出 `features` 映射、降序排列 `ranked_features` 及 `zero_gain_features`。
- `--type shap`（需 `--data`）：单输出返回 `[{row,base,features:[{name,value}]}]`；多分类（`n_output_groups > 1`）返回 `[{row,classes:[{class,base,features}]}]`，限 500 行。加 `--summary`（或 `--type shap-summary`）时输出全局 `mean_abs_shap` 降序汇总（极其节省上下文 token）。
- `--type interaction`（需 `--data`，AG-OMNI）：计算二阶 Tree SHAP 特征交互，输出 `top_interactions`（前 20 强特征对交互）与 `interaction_matrix`。
- `--type dump`（无需数据，AG-OMNI）：按 `--dump-format json|text|dot` 导出完整树结构或 Graphviz DOT 拓扑。

## inspect

加载任意受支持模型（leaves.json / `.quant.json` 量化侧车 / XGBoost JSON / LightGBM text / ONNX / sklearn JSON），输出深度模型结构画像 JSON（发布前复核、动态决策依据）。

```
leaves inspect --model PATH [--metrics PATH]
```

输出：`{"schema_version":1,"file","format","objective","kind"（gbtree|dart|gblinear|sklearn_gbdt）,
"num_features","n_output_groups","n_trees","n_boost_rounds","base_score","max_depth_observed",
"total_nodes","total_leaves","mean_leaves_per_tree","used_features_count","unused_features_count"[,"unused_features","feature_names"]}`。

## split（零泄漏数据切分与漂移审计，AG-NEXT / AG-ZENITH）

在训练前将单文件数据切分为 `train` / `val` / `test` 并输出 `split.json`（含 `leakage_ok`、`missing_cells`、各子集 `label_means` 与 `max_label_drift`）。

```
leaves split --data PATH --out-dir DIR
  [--strategy auto|random|stratified|group|time]
  [--ratios 0.7,0.15,0.15] [--seed 42] [--na-policy error|skip-row|nan]
  [--label-col NAME] [--group-col NAME] [--time-col NAME] [--drop-cols C1,C2]
  [--metrics PATH]
```

- `auto`：排序数据（含 `qid` 或 `--group-col`）自动选 `group`（同组绝不跨集泄漏）；分类标签自动选 `stratified`；指定 `--time-col` 时选 `time`（严格按时间序前→中→后切分，`leakage_ok: true`）。

## runs（运行账本原生分析与演化搜索算子，AG-NEXT / AG-OMNI）

无需手写脚本，直接对 `runs.jsonl` 执行统计、折级 Pareto 前沿、选父、超参去重、双亲交叉重组与定向变异提案。

```
leaves runs status        --runs runs.jsonl
leaves runs pareto        --runs runs.jsonl
leaves runs select-parent --runs runs.jsonl [--strategy pareto|top-k] [--seed 42]
leaves runs check-dup     --runs runs.jsonl [--params '{"depth":6,"lr":0.1,...}'] [--depth 6 --lr 0.1 ...]
leaves runs crossover     --runs runs.jsonl [--parent-a TAG] [--parent-b TAG] [--seed 42] [--metrics xover.json]
leaves runs propose       --runs runs.jsonl [--parent TAG] [--diagnose diag.json] [--seed 42] [--metrics prop.json]
```

## diagnose（全息 ASI 诊断包，AG-NEXT / AG-OMNI）

聚合 `metrics.json`、`rounds.jsonl`（学习曲线斜率与平台期）、模型特征重要性（`--model`，识别零增益特征）与数据画像（`--data`，识别 ID 列、常数列、非数值类别列与缺失值），产出结构化诊断结论、`suggested_drop_cols`、`suggested_mutations` 与可直接拼接执行的 `suggested_cli_args`。

```
leaves diagnose [--metrics M.json] [--model PATH] [--rounds R.jsonl] [--data D.csv] [--out diag.json]
```

输出含 `status`（`overfitting` | `underfitting` | `plateau` | `high_variance` | `feature_noise` | `data_hygiene` | `healthy`）、`zero_gain_features`、`top_features`、`suggested_drop_cols`、`suggested_mutations` 与 `suggested_cli_args`。

## schema（机器自省契约，AG-NEXT）

输出全部子命令 flags、错误码枚举与核心 JSON 产物契约，供 Agent 零样本工具发现。

```
leaves schema [--subcommand train]
```

## version

输出版本/构建信息 JSON（排查「装的哪个版本」用；`go install pkg@vX.Y.Z` 会带真实 tag）。

```
leaves version
→ {"go":"go1.26.5","version":"v2.5.3"}            # go install 安装
→ {"go":"go1.26.5","version":"(devel)","commit":"…"}  # 仓库内 go build/run
```

## lessons（跨任务记忆库，v2.7+）

存储 `~/.leaves/lessons.jsonl`（JSONL 追加写；`LEAVES_LESSONS_PATH` 可覆盖）。
策略（何时读/写）见 SKILL §4.6；这里只是存储与检索管道。输出均为单行 JSON（JSONL）。

```
leaves lessons add --task ml-ctr-v3 --lesson "小数据别上 subsample" [--evidence "0.31->0.34"] [--tag small-data,subsample]
→ {"ts":"2026-08-22T…","task":"ml-ctr-v3","lesson":"…","evidence":"…","tags":[…]}

leaves lessons search --query "subsample 小数据" [--limit 5]   # 词命中数降序；大小写不敏感
→ {"ts":…,"task":…,"lesson":…,"hits":2}

leaves lessons list [--task churn]                              # 按任务子串过滤
```

错误：`--task`/`--lesson`/`--query` 缺失 → exit 1（usage）；库文件损坏 → `data_load` 带修复 hint。

## publish

把模型打成本地工件包。**不推 registry**（leaves 边界）。

```
leaves publish --model PATH --out-dir DIR
  --version NAME          (默认 "1.0.0")
  --quantize              (int8 阈值量化 + parity 报告)
  --export-xgb            (导出 model.xgb.json，leaves→XGBoost 3.x)
  --export-onnx           (导出 model.onnx，leaves→ONNX ai.onnx.ml TreeEnsemble)
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
  model.onnx              (--export-onnx 时：ONNX TreeEnsembleRegressor/Classifier)
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
| `leaves_cli` | 生成该工件包的 CLI 版本（`vX.Y.Z` 或 `(devel)+<短commit>`，同 `leaves version`） |
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
