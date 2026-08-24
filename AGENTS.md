尽可能 Go 语言来实现功能！
======
尽可能模块化！
======

## 计算底座（2026-06-13 决策，2026-06-15 文档同步）

- **已废弃**：GoMLX（`github.com/gomlx/gomlx`）、gogpu/wgpu 直连、`treebuilder/hist_accel_born.go`、`born_train` build tag
- **统一底座**：[Born](https://github.com/born-ml/born)（`github.com/born-ml/born`）
- **推理**：`tree.NativeEngine`（golden）；`tree.BornEngine` 为 parity/兼容路径（桌面实测慢 6–30×）。js/wasm 上 `BornEngine` 委托 Native；BackendAuto 的 WASM 规则一律 Native（`wasm_native`）
- **训练加速**：`treebuilder/hist_accel.go`（Born CPU 增益扫描）+ `hist_accel_webgpu_*.go`（WebGPU hist）；环境变量 `LEAVES_TRAIN_ACCEL` / `train.Config.AccelMode`
- **无中间 IR**：Born 路径不维护 `TreeData`/`ForestData` 快照；`tree/born_walk.go` 等对 `ForestIR` 做张量遍历
- **包边界**：`tree/` 不依赖 `train/`

## Backend 命名

| 常量 | 含义 |
|------|------|
| `BackendNative` | 纯 Go 标量遍历（golden） |
| `BackendBornCPU` | Born CPU 后端（SIMD） |
| `BackendBornGPU` | Born WebGPU 后端（Windows DX12） |
| `BackendAuto` | 默认 **Native**（2.1：batch 阈值「加速区」未复现）；Born 走显式指定或 `LEAVES_BACKEND_PROFILE=1` 实测选型；见 `docs/backend-auto.md` |

## Agent 自动化（SKILL 驱动，无 MCP）

Agent 通过 **SKILL 指导 + shell CLI + metrics.json** 完成全自动「训练→指标优化→发布」。
**Agent 即优化器**：搜索逻辑在 SKILL 文本里，不在 leaves 代码里（与「不内置搜索」哲学一致）。

- **通用 SKILL**：[`skills/leaves-autotrain/`](skills/leaves-autotrain/SKILL.md)（任意监督学习任务）
  - 闭环：嗅探数据 → `leaves train`（`--cv`）→ 读 metrics.json → 按 SKILL 决策表调参 → 再训 → 收敛 → `leaves publish`
  - CLI 参考 / metrics.json schema：[`skills/leaves-autotrain/cli.md`](skills/leaves-autotrain/cli.md)
- **推荐系统 SKILL**：[`skills/recsys-orchestrator/`](skills/recsys-orchestrator/SKILL.md)（召回→排序→发牌四段流水线 + §十生产闭环八段剧本）
- **推荐控制面包**（§十七 RC）：`recsys/{contract,split,eval,ledger,replay,monitor,release}` —— 快照/事件/决策/曝光/反馈/证据契约、时间切分防泄漏、三层门禁、账本、归因回放、监控、发布状态机 + `release.Adapter`（只产出推广/回滚请求，无网络副作用）；端到端演练 `recsys/loop`；指南 [`docs/recsys-loop.md`](docs/recsys-loop.md)
- **MovieLens Ranker**：[`demos/movielens/TUTORIAL.md`](demos/movielens/TUTORIAL.md) · [`skills/recsys-movielens-ranker/`](skills/recsys-movielens-ranker/SKILL.md)（demo 另有可选 `cmd/mcp`，**不是**主闭环依赖）
- **通用 CLI**：`go run ./cmd/leaves <sniff|train|eval|predict|inspect|explain|publish|lessons|version>` —— sniff 自动推荐 objective；train 支持 `--cv`/`--runs`/`--tag`/`--emit-rounds`；explain 输出特征重要性/SHAP；`lessons` 跨任务记忆；子命令均写 metrics.json（version 除外）
- **CLI 入口（用哪条命令）**：见 README「CLI 入口」表——训练发布用 `cmd/leaves`；推荐八段控制面用 `recsys/cmd/control`；离线四段用 `recsys/cmd/smoke`；MovieLens 案例用 `demos/movielens/cmd/*`（含可选 MCP）
- **闭环原语**：`train.NewLearner`/`Fit`/`Eval`/`CrossValidate`、`data.FromFileAuto`、`learner.Model()`→`io.SaveLeavesJSONFile`/`ExportXGBoostJSONFile`、`quantize.QuantizeForest`

## 文档

- 战略路线图：[`演进计划.md`](演进计划.md) v5.4（Phase A–E 第一轮已落地）
- **Agentic 收口方案**：[`演进方案.md`](演进方案.md) v2.2（Phase 0–5 已达成 + POST 加固 + §十六 EVO 演化搜索；§十七 RC 推荐生产闭环已落地）
- API 分层 / 发版：[`docs/api-surface.md`](docs/api-surface.md)、[`docs/release-checklist.md`](docs/release-checklist.md)、[`docs/versioning.md`](docs/versioning.md)、[`CHANGELOG.md`](CHANGELOG.md)
- 扩展点：[`docs/extension-points.md`](docs/extension-points.md)
- BackendAuto：[`docs/backend-auto.md`](docs/backend-auto.md)
- 互操作等级：[`docs/interop-matrix.md`](docs/interop-matrix.md)
- backlog / 维护状态：[`TODO.md`](TODO.md)（**v2.8.0 维护期**；新工作先写入 TODO 再实现）
- 回归矩阵：[`docs/testdata-matrix.md`](docs/testdata-matrix.md)
- Serving 模板（非官方框架）：[`examples/serving-template/`](examples/serving-template/)

## 格式与 IO（当前）

- **训练数据**：`data/sniff.go` + `data/fromfile.go`（`FromFileAuto` / `LoadDataAuto`）
- **模型加载**：`io/load.go`（格式探测 + LGB/XGB/leaves.json/ONNX/sklearn → ForestIR，无需根包 init）、`io/transform_auto.go`（`AutoTransform` 默认 true）
- **便利 API**：`train/load.go`、`train_api.go`（`NewLearnerFromModelAndData`）
