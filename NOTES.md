# NOTES — 仍影响用户的兼容说明

> **历史阶段流水账已收敛**：P0–T5 / Agentic / 库线 Phase A–E 的完成项见 [`TODO.md`](TODO.md) 与 git history。  
> 路线图：[`演进计划.md`](演进计划.md) · API 分层：[`docs/api-surface.md`](docs/api-surface.md) · 版本策略：[`docs/versioning.md`](docs/versioning.md)。

---

## 1. AutoTransform 默认（v4.3+，v2.x 仍有效）

`leaves.DefaultLoadOptions()` / `io.DefaultLoadOptions()` 默认 **`AutoTransform: true`**。

| 目标（示例） | `PredictSingle` / OutputValue |
|--------------|-------------------------------|
| `binary:logistic` 等 | 变换后值（如概率） |
| `reg:squarederror` 等 | 仍为 raw margin（无变换） |

需要 **默认 raw margin**（旧行为）：

```go
m, err := leaves.LoadFromFile(path, &leaves.LoadOptions{AutoTransform: false})
// 或 predict.Request{Output: predict.OutputMargin}
```

`contrib` / SHAP **始终在 margin 空间**，与 `AutoTransform` 无关。

遗留 `LGEnsembleFromFile(path, loadTransformation)` 第二参数语义不变；新代码请用 `LoadOptions`。

---

## 2. 计算底座 Born（2026-06 起）

- **已废弃**：GoMLX、`born_train` build tag、gogpu 直连推理。  
- **现用**：[Born](https://github.com/born-ml/born)；`NativeEngine` = 正确性 golden。  
- **BackendAuto 2.0**（**推理**选型）：见 [`docs/backend-auto.md`](docs/backend-auto.md)。  
- **训练加速**（独立）：`LEAVES_TRAIN_ACCEL` / `train.Config.AccelMode`——**不会**改推理 BackendAuto；交叉说明见 backend-auto §训练 vs 推理。

---

## 3. 命名与其它长期兼容点

- `NClasses` → 使用 `NRawOutputGroups` / `NOutputGroups`（变换后维度）。  
- XGBoost DART：预测时不要传 `nEstimators = 0`（见 XGB 文档）。  
- LightGBM DART：文件格式可能仍显示为 `lightgbm.gbdt` 名称。

更完整的外部库版本矩阵见 [`compatibility.md`](compatibility.md)。

---

## 4. 模块版本与 `go get`

模块路径为 `github.com/linkerlin/leaves/v2`（**含** `/v2` 后缀；commit cab6a6f 起迁移，早期 `v2.1.x` tag 打在无后缀路径上）。

```powershell
go get github.com/linkerlin/leaves/v2@latest
go install github.com/linkerlin/leaves/v2/cmd/leaves@latest
```

迁移前的旧 tag（`v2.1.x` 等）挂在 `github.com/linkerlin/leaves`（无后缀）下；若必须引用历史版本，按伪版本拉对应 commit。新代码一律用 `/v2` 路径。

---

## 5. 文档地图（避免重复阶段史）

| 文档 | 读什么 |
|------|--------|
| README | 上手与推荐入口 |
| docs/api-surface.md | 推荐 / 兼容 / 实验 API |
| docs/release-checklist.md | 发版勾选 |
| docs/versioning.md | v2.x 允许改什么 |
| 演进方案.md | Agent 闭环契约（已达成）+ §十六 演化搜索 |
| TODO.md | 可执行 backlog 与已完成存档 |

---

## 6. runs.jsonl / metrics.json 字段只增不删（2026-08 起）

- 账本与 metrics 的 Agent 信号字段**只增不删**：旧 Agent 忽略未知键即安全（演进方案 §13.1）。  
- 2026-08（EVO-02）新增：`n_trees`、`elapsed_ms`（train；未存模型时省略）、runs 行 `fold_metrics`（仅 `--cv`）。均为 omitempty，schema_version 维持 `1`，不构成破坏性变更。
