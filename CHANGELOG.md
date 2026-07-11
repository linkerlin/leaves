# Changelog

本文件记录面向用户的版本变更。格式参考 [Keep a Changelog](https://keepachangelog.com/)。  
发版前勾选 [`docs/release-checklist.md`](docs/release-checklist.md)。

## [Unreleased]

### Added

- **MovieLens 四段流水线（DEMO-ML-4stage）**
  - `recsys/movielens`：纯 Go 加载 ml-100k → 四元 Dataset + 片名
  - `pipeline.RunFromDataset`：prep→召回→LTR→发牌（合成/真实数据共用）
  - `go run ./recsys/cmd/movielens` · `agent four-stage` · MCP `movielens_four_stage`
  - 召回策略：部分正样本 + 未交互补齐；发牌 `fillOverflow` 可凑满 DeckSize

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
