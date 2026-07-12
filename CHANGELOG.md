# Changelog

本文件记录面向用户的版本变更。格式参考 [Keep a Changelog](https://keepachangelog.com/)。  
发版前勾选 [`docs/release-checklist.md`](docs/release-checklist.md)。

## [Unreleased]

### Added

- **完整 ONNX Graph 导入**（`io.LoadOnnxGraph`，原列「明确不做」，现落地）：
  - 复用 `github.com/born-ml/born/onnx` 运行时（30+ 算子，opset 1–21），不在 leaves 重实现算子。
  - `LoadOnnxGraph(data []byte) (OnnxModel, error)`：返回 `OnnxModel`，`Predict([]float32) ([]float32, error)` 单样本前向；暴露 InputNames/OutputNames/OpsetVersion。
  - 与既有 `LoadONNX`（TreeEnsembleRegressor 子集，wasm 可用）互补：通用 NN / 图推理走 born，仅非 wasm（wasm stub 返回可操作错误）。
  - 关键：born 公共 `onnx.Model` 接口的方法引用 `internal/tensor.RawTensor`，但 `born/tensor` 公共包以 `type RawTensor = tensor.RawTensor` 别名重导出，使 leaves 可构造张量并调用 Forward。

---

## [2.2.0] - 2026-07-11

> **主题**：向量叶 `multi_output_tree` 训练（multi-target 一棵树、叶子为向量）  
> **Release 正文**：[`docs/release-notes-v2.2.0.md`](docs/release-notes-v2.2.0.md)

### Added

- **向量叶 `multi_output_tree` 训练**（XGBoost `multi_output_tree`，原列「明确不做」，现落地）：
  - `train.Config.MultiOutputTree bool`：开启后每轮长**一棵** `OutputDim=numGroups` 的向量叶树（非 `one_output_per_tree` 的每类一棵）。
  - 分裂增益跨输出组求和（同一 `(feat,thr)` 共享）；叶权重为逐类 `-G_c/(H_c+λ)` 向量。
  - treebuilder 内部统一为 k 维（grad/hess `[n*k]`、`splitGain` 跨类求和、leaf-major `LeafValue` flatten）；k=1 退化为标量（现有路径零回归）。
  - 适用 `multi:*` 与 `NumTarget>1`；`NewLearner` 校验需多输出。
  - Born/WebGPU 增益扫描对 `k>1` 回退纯 CPU（正确，较慢；标量 k=1 仍享加速）。

---

## [2.1.6] - 2026-07-11

> **主题**：并发 DATA RACE 与 transformation panic 修复；golangci-lint v2 门禁全量落地  
> **Release 正文**：[`docs/release-notes-v2.1.6.md`](docs/release-notes-v2.1.6.md)

### Fixed

- **并发 DATA RACE**（`treebuilder.parallelEvalHistFeats`）：多 goroutine 共享同一 `row` buffer 写入 `Dense.Row()`；改为每 goroutine 独立 buffer。影响多线程 `hist` 训练的正确性
- **transformation panic**（`TransformType(Exponential).Name()` 数组越界）：`transformNames` 缺 `exponential` 项；`TransformExponential.Name()` 因此错报为 "logistic"。加载 LightGBM exponential objective 模型时触发

### Changed

- 新增 **golangci-lint v2** CI 门禁：`govet`（关 composites）/ `ineffassign` / `unused` / `staticcheck`（关 QF1003/ST1000/ST1003）/ `errcheck`（`std-error-handling` 预设 + 测试豁免）/ **gofmt** formatter，全部 blocking，0 issues 基线
- 新增 **race detector** CI job（`go test -race -short ./...`）
- 清除 **~360 行死代码**（unused linter 全清零）：explain interventional SHAP 簇、treebuilder 被 `batchAccumulateHistWebGPU` 取代的 GPU hist 旧路径（`accumulateHistWebGPU`×3 / `prebuildGPUHists` / `gpuHistBuildEnabled` / `gpuHistMinSamples`）、各处孤立 helper；永跳测试（`lgmsltr`/`lghiggs`/`xghiggs` fixture 未入库）、失效 GoMLX 脚本、误入库产物
- 修存量 **ineffassign / staticcheck / errcheck**：真实 error 忽略（transform / Predict / predictLeafIndicesInner / Seek / Iterate）改 `_ =` 显式丢弃；dead 赋值 / 空分支 / bool 比较 / nil-len 等逐项修正

### Added

- **测试覆盖**：`transformation`（0→6，含 `Name()` 越界回归）、`predict`（0→5，含 Engine 接口契约锁定）、`booster`（0→6，含 GBLinear zero-grad 端到端）；`internal/xgbin` 补 golden 节点断言（关闭遗留 TODO）
- **包文档** `doc.go`：`train` / `explain` / `objective` / `metrics` / `mat` / `util`（pkg.go.dev 不再裸符号）
- **文档门禁测试**：`TestNoDeadReadmeZhLinks`（扫所有根 .md 防死链复发）、`TestDocVersionRefsConsistent`（从文档头部提取版本号，校验所有引用一致）

### Documentation

- 项目状态 **维护期 → 开发期**；版本漂移修复（演进计划 v5.4 / 演进方案 v1.5 全局一致）；`doc.go` 拼写修复（implemetation / exibit / beacase）

---

## [2.1.5] - 2026-07-11

> **主题**：MovieLens 四段流水线（prep→召回→LTR→发牌）  
> **Release 正文**：[`docs/release-notes-v2.1.5.md`](docs/release-notes-v2.1.5.md)

### Added

- **MovieLens 四段流水线（DEMO-ML-4stage）**
  - `recsys/movielens`：纯 Go 加载 ml-100k → 四元 Dataset + 片名
  - `pipeline.RunFromDataset`：prep→召回→LTR→发牌（合成/真实数据共用）
  - `go run ./recsys/cmd/movielens` · `agent four-stage` · MCP `movielens_four_stage`
  - 召回策略：部分正样本 + 未交互补齐；发牌 `fillOverflow` 可凑满 DeckSize

### Documentation

- TUTORIAL §11.1、recsys-orchestrator / MovieLens SKILL、TODO 快照对齐

---

## [2.1.4] - 2026-07-11

> **主题**：MovieLens 推荐片名旁车 + walkthrough  
> **Release 正文**：[`docs/release-notes-v2.1.4.md`](docs/release-notes-v2.1.4.md)

### Added

- MovieLens 推荐旁车 `rank_movielens_*_meta.jsonl`（movie_id/title）；`agent recommend` / `cmd/recommend` 展示片名
- `demos/movielens/scripts/walkthrough.ps1` / `.sh` 端到端脚本
- `demos/movielens/rankutil/meta.go` 加载旁车

### Documentation

- TUTORIAL / README 同步 meta 与 walkthrough

---

## [2.1.3] - 2026-07-11

> **主题**：MovieLens Ranker Agent/MCP 全流程 demo 与教程  
> **Release 正文**：[`docs/release-notes-v2.1.3.md`](docs/release-notes-v2.1.3.md)

### Added

- **MovieLens Ranker Agent/MCP demo**
  - `demos/movielens/cmd/agent`：stdout JSON CLI（status/prepare/train/eval/recommend/full-pipeline）
  - `demos/movielens/cmd/mcp`：stdio MCP server（`movielens_*` tools）
  - `demos/movielens/agentops`：CLI/MCP 共用实现
  - 详尽教程 [`demos/movielens/TUTORIAL.md`](demos/movielens/TUTORIAL.md)
  - Agent SKILL [`skills/recsys-movielens-ranker/`](skills/recsys-movielens-ranker/SKILL.md)

### Documentation

- `recsys-orchestrator` / README / AGENTS 交叉链接 MovieLens Agent 路径

## [2.1.2] - 2026-07-11

> **主题**：Explain 性能缓存、可拆 serving 模板、SK/testdata 门禁  
> **Release 正文**：[`docs/release-notes-v2.1.2.md`](docs/release-notes-v2.1.2.md)

### Added

- **Serving 模板（LIB-30）**：[`examples/serving-template/`](examples/serving-template/) — 可整目录拆为独立仓（优雅退出、max batch、/ready、热加载演示、Dockerfile）；`examples/http` 仍为最小 embed
- **SK / testdata 门禁（LIB-11/12）**：sklearn 协议矩阵文档 + 失败用例；`TestTestdataMatrixArtifactsExist`

### Changed

- **Explain 性能（LIB-22）**：`TreeExplainer` 缓存树节点权重与背景 margin，多样本 SHAP 复用 path 缓冲（语义不变；`go test ./explain`）

### Documentation

- [benchmark-baseline.md](docs/benchmark-baseline.md) 增加 Tree SHAP bench 指引
- NOTES：模块路径与 `v2.x` tag 的代理提示

## [2.1.1] - 2026-07-11

> **主题**：v2.1 后加固 — Agent 契约门禁、ONNX TreeEnsemble 子集、多目标回归  
> **Release 正文**：[`docs/release-notes-v2.1.1.md`](docs/release-notes-v2.1.1.md)

### Added

- **CLI / Agent 契约加固**
  - `skills/` ↔ `.cursor/skills/` 镜像入库 + `TestSkillsMirrorSync` CI 门禁
  - `train --strict-flags` → `error=cv_conflict`（cv 与 val/early-stop 冲突）
  - `publish --print-repro`：stdout 打印复现 train 命令
  - `train --out-final`：早停时另存 final-round 侧车（`final_model` / `final_round`）
  - metrics.`train_accel`：Fit 后实际训练加速模式（与推理 BackendAuto 无关）
- **扩展示例** [`examples/extension/`](examples/extension/)：`custom:l1` + `max_abs_error`
- **Bench 样例工件** [`docs/bench/`](docs/bench/)：`BenchRecord` JSONL 格式对照
- **ONNX（实验）**：`TreeEnsembleRegressor` 极小子集导入（`BRANCH_LEQ`/`SUM`/`NONE`）
- **多目标训练（LIB-21）**：`data.MultiTarget` + `train.Config.NumTarget` + CLI `--num-target`（CSV 末 N 列标签；one_output_per_tree）
  - `predict` / `eval --num-target` 多目标闭环；[`examples/multitarget/`](examples/multitarget/)

### Changed

- cli.md：错误码表、data_quality 扫描上限（5000 行）、save-best「内存截断非重训」、registry 对接模板（S3/gh/OCI）
- [`docs/backend-auto.md`](docs/backend-auto.md)：训练 vs 推理交叉说明；第二轮候选边界
- ONNX 支持等级：占位 → **实验子集**（完整 Graph 仍不做）

### Documentation

- 演进方案 / TODO / interop-matrix / api-surface 同步

## [2.1.0] - 2026-07-10

> **主题**：Agentic 闭环契约收口 + 库线平台化（扩展点 / BackendAuto 2.0 / 互操作等级 / 发版治理）  
> **Release 正文**：[`docs/release-notes-v2.1.0.md`](docs/release-notes-v2.1.0.md)

### Added

- **Agentic CLI 闭环**（`cmd/leaves`）：`sniff|train|eval|predict|inspect|explain|publish`
  - 完备 `params` 账本、`--save-best` 默认、`--from-run` 复现
  - `--error-format json`、`manifest.reproduce`、`--emit-repro-script`
  - sniff `data_quality`、`--na-policy error|skip-row`
- **扩展点**：objective / metric 全量 `Register`；[`docs/extension-points.md`](docs/extension-points.md)
- **BackendAuto 2.0**：batch≥64→BornCPU、≥256+GPU→BornGPU、稀疏/小批 Native；[`docs/backend-auto.md`](docs/backend-auto.md)
- **互操作等级**：稳定 / 实验 / 占位；`*io.LoadError` 可操作 hint；[`docs/interop-matrix.md`](docs/interop-matrix.md)
- **发版治理**：[`docs/api-surface.md`](docs/api-surface.md)、[`docs/versioning.md`](docs/versioning.md)、[`docs/release-checklist.md`](docs/release-checklist.md)
- `tree.BenchRecord` 统一 bench JSONL；`SelectBackendExplained` 可解释选型

### Changed

- `DefaultLoadOptions` 默认 `AutoTransform: true`（见 [NOTES.md](NOTES.md)）
- BackendAuto：大 batch 无 GPU 时选 BornCPU（不再静默 Native）
- NOTES 瘦身为仍有效兼容注记；路线图 v5.4 / 演进方案 v1.4 收口

### Fixed

- sniff `n_features` 与 `feature_names` 一致
- 早停路径默认保存 best_round 模型（`model_round`）
- README 中英文互链（`README.md` ↔ `README.en.md`）

### Documentation

- skills/leaves-autotrain + Cursor 镜像；examples/autotrain walkthrough 刷新

### CI / 质量

- 本地 `go test ./... -count=1` 已通过（发版前建议再跑 CI 三 OS）
- 门禁：`docs` 文档存在性、CLI 契约、BackendAuto 决策表、io 支持等级

---

## 打 tag

```powershell
# 工作区已提交后：
git tag -a v2.1.0 -m "leaves v2.1.0: Agentic CLI + platform hardening"
git push origin v2.1.0
# GitHub Release 正文粘贴 docs/release-notes-v2.1.0.md
```

---

## 更早版本

历史能力基线（P0–T5 训练完备、Born 迁移、量化/WASM 等）见 git history 与 [`TODO.md`](TODO.md) 存档。
