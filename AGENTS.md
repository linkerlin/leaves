尽可能 Go 语言来实现功能！
======
尽可能模块化！
======

## 计算底座（2026-06-13 决策，2026-06-15 文档同步）

- **已废弃**：GoMLX（`github.com/gomlx/gomlx`）、gogpu/wgpu 直连、`treebuilder/hist_accel_born.go`、`born_train` build tag
- **统一底座**：[Born](https://github.com/born-ml/born)（`github.com/born-ml/born`）
- **推理**：`tree.NativeEngine`（golden）+ `tree.BornEngine`（Born CPU / **WebGPU Windows**）；js/wasm 仅 Native
- **训练加速**：`treebuilder/hist_accel.go`（Born CPU 增益扫描）+ `hist_accel_webgpu_*.go`（WebGPU hist）；环境变量 `LEAVES_TRAIN_ACCEL` / `train.Config.AccelMode`
- **无中间 IR**：Born 路径不维护 `TreeData`/`ForestData` 快照；`tree/born_walk.go` 等对 `ForestIR` 做张量遍历
- **包边界**：`tree/` 不依赖 `train/`

## Backend 命名

| 常量 | 含义 |
|------|------|
| `BackendNative` | 纯 Go 标量遍历（golden） |
| `BackendBornCPU` | Born CPU 后端（SIMD） |
| `BackendBornGPU` | Born WebGPU 后端（Windows DX12） |
| `BackendAuto` | 按 workload 在 Native / Born 间选择 |

## Agent 自动化（SKILL 驱动，无 MCP）

Agent 通过 **SKILL 指导 + shell CLI + metrics.json** 完成全自动「训练→指标优化→发布」。
**Agent 即优化器**：搜索逻辑在 SKILL 文本里，不在 leaves 代码里（与「不内置搜索」哲学一致）。

- **通用 SKILL**：[`skills/leaves-autotrain/`](skills/leaves-autotrain/SKILL.md)（任意监督学习任务）
  - 闭环：嗅探数据 → `leaves train`（`--cv`）→ 读 metrics.json → 按 SKILL 决策表调参 → 再训 → 收敛 → `leaves publish`
  - CLI 参考 / metrics.json schema：[`skills/leaves-autotrain/cli.md`](skills/leaves-autotrain/cli.md)
- **推荐系统 SKILL**：[`skills/recsys-orchestrator/`](skills/recsys-orchestrator/SKILL.md)（召回→排序→发牌四段流水线）
- **通用 CLI**：`go run ./cmd/leaves <sniff|train|eval|predict|inspect|explain|publish>` —— sniff 自动推荐 objective；train 支持 `--cv`/`--runs`/`--tag`/`--emit-rounds`；explain 输出特征重要性/SHAP；子命令均写 metrics.json
- **闭环原语**：`train.NewLearner`/`Fit`/`Eval`/`CrossValidate`、`data.FromFileAuto`、`learner.Model()`→`io.SaveLeavesJSONFile`/`ExportXGBoostJSONFile`、`quantize.QuantizeForest`

## 文档

- 战略路线图：[`演进计划.md`](演进计划.md) v5.0
- 可执行 backlog：[`TODO.md`](TODO.md)（P0–T5 + v3.1 已完成）
- 回归矩阵：[`docs/testdata-matrix.md`](docs/testdata-matrix.md)

## 格式与 IO（当前）

- **训练数据**：`data/sniff.go` + `data/fromfile.go`（`FromFileAuto` / `LoadDataAuto`）
- **模型加载**：`io/load.go`（格式探测）、`io/transform_auto.go`（`AutoTransform` 默认 true）
- **便利 API**：`train/load.go`、`train_api.go`（`NewLearnerFromModelAndData`）
