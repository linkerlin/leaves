# Changelog

本文件记录面向用户的版本变更。格式参考 [Keep a Changelog](https://keepachangelog.com/)。  
发版前勾选 [`docs/release-checklist.md`](docs/release-checklist.md)。

## [Unreleased]

---

## [2.6.0] - 2026-08-21

> **主题**：推荐生产闭环控制面（演进方案 §十七 RC0–RC4）——七包契约 + 八段端到端演练  
> **Release 正文**：[`docs/release-notes-v2.6.0.md`](docs/release-notes-v2.6.0.md)

### Added — 推荐生产闭环控制面（演进方案 §十七 RC，2026-08-21）

- **`recsys/contract`**：冻结 schema v1 契约——`DatasetSnapshot`（输入文件 sha256 + `FeatureSchemaHash`，`VerifyFiles` 任一不符即失败）、`InteractionEvent`/`DecisionEvent`/`ExposureEvent`/`FeedbackEvent`（UTC 时间、匿名键绊线、原因码、回链校验：重复 ID / 未知类型 / 反向时间 / 位次不符均失败）、`ReleaseEvidence`；JSONL 泛型读写。
- **`recsys/split`**：时间切分（`train_end < val_start < test_start`）+ as-of 泄漏门禁 `CheckLeakage` + cold-start 分层；隔离带事件丢弃。
- **`recsys/eval`**：三层离线门禁（data / candidate_rank / deal）——Recall@K、coverage、NDCG/MAP、deck 质量（fill/dup/tag overflow/item coverage）、cold/returning 分层；阈值业务配置，缺失时仅 `exploratory`，未知指标 `block`；产出 `evaluation.json`。
- **`recsys/ledger`**：决策/曝光/反馈 JSONL 账本（append 即校验回链与时间因果）；`DecisionFromDeal` 把发牌终稿映射为审计真源决策（补位条目带 `tag_overflow` 原因码）。
- **`recsys/replay`**：归因窗样本重建——只有可归因到 shown 曝光的窗内反馈构成正样本；迟到/孤立/抑制反馈计数入 `replay_report.json`；确定性排序。
- **`recsys/monitor`**：窗口聚合（ctr / orphan_feedback_rate / deck 质量）→ `monitor_report.json`，状态 `ok|warn|block` + reason code；**触发器** `TriggerSet`——指标连续 N 窗口越界（warn/block）→ `retrain/rollback_requested`，带冷却期与恢复重置（§17.6「配置驱动，不是 Agent 临场猜测」）。
- **`recsys/release`**：发布状态机（`exploratory→candidate→approved→promoted→observing→retrain/rollback_requested/retired`）——evidence 三层门禁齐全 + 模型 hash 一致才可 candidate；人工批准默认；`last_known_good` 只指向已记录证据；`Adapter` 接口 + `FakeAdapter` 只产出推广/回滚请求（无网络副作用）；`run_status.jsonl`。
- **`recsys/loop`**：八段端到端演练 `TestAgenticRecsysLoopDrill`（快照/切分 → 门禁 evidence → fake adapter promoted → 账本 → 健康观察 → 退化注入 → 回滚指向 last_known_good → replay → retrain）。
- 文档：`docs/recsys-loop.md`（RC-00 基线 + 控制面指南）；`skills/recsys-orchestrator` §十八段剧本（镜像同步）；README / AGENTS / `examples/serving-template` 对接说明。

---

## [2.5.5] - 2026-08-16

> **主题**：module zip 瘦身——21.8 MB → ~5 MB  
> **Release 正文**：[`docs/release-notes-v2.5.5.md`](docs/release-notes-v2.5.5.md)

### Removed

- **`bin/install_pjrt.exe` / `bin/verify_pjrt.exe`**（合计 ~36 MB 原始体积，全仓零引用的孤儿实验二进制）：此前随 module zip 分发，`go get` 每次下载。module zip 由 git 追踪文件构成——**这是所有「不该提交的产物会被 go get 用户下载」问题的根因**。
- **`examples/wasm/leaves.wasm`**（3.5 MB，可重建产物；README/CI 均为现场构建，wasm 体积门禁也是临时目录新构建而非读该文件）。
- **`.chong/`**（11 个文件，另一 Agent 工具的工作记忆/事件日志，与库无关）。

以上均加 `.gitignore`；testdata（回归矩阵必需）与 logo 保留。

---

## [2.5.4] - 2026-08-16

> **主题**：manifest 复现契约修复——reproduce 不再丢路径语义  
> **Release 正文**：[`docs/release-notes-v2.5.4.md`](docs/release-notes-v2.5.4.md)

### Fixed

- **manifest.reproduce 丢参**：原构建器漏 `--cv`/`--max-leaves`/`--num-target`/`--val`/`--early-stop`——CV run 的复现命令会退化为全量单训（与记录的 `cv_mean` 不可比）；早停 run 复现丢 `--val --early-stop`。现按 run 类型补全（CV 行带 `--cv K`；早停行带 `--val X --early-stop N`，val 路径记录于 `params.val`，仅单跑路径记录）。
- **`manifest.leaves_cli`**：硬编码占位 `agentic-1` → 真实版本标签（`vX.Y.Z` 或 `(devel)+<短commit>`，同 `leaves version`）。

### Changed

- **复现语义分工**（写实既有行为）：`--from-run` 为定稿/变异起点，**不回填 val**（默认全量重训，SKILL 定稿流程不变）；忠实重放某次 run 用 `manifest.reproduce` 或显式 `--val`。

---

## [2.5.3] - 2026-08-16

> **主题**：`leaves version` 子命令  
> **Release 正文**：[`docs/release-notes-v2.5.3.md`](docs/release-notes-v2.5.3.md)

### Added

- **`leaves version`**：输出版本/构建信息 JSON（`{version, go[, commit]}`，基于 `debug.ReadBuildInfo`）。`go install github.com/linkerlin/leaves/v2/cmd/leaves@vX.Y.Z` 安装的用户与 Agent 可自查版本（原报「未知子命令」）；仓库内 `go build/run` 显示 `(devel)`（+ commit，若 VCS 信息被嵌入）。

---

## [2.5.2] - 2026-08-16

> **主题**：`--from-run` 谱系流程修复（tag 未命中不再硬错）  
> **Release 正文**：[`docs/release-notes-v2.5.2.md`](docs/release-notes-v2.5.2.md)

### Changed

- **`train --from-run --tag`（新 tag）回落最优行**：tag 不在账本时原报 `usage` 错误退出——SKILL §4.5 的「复现最优 + `--tag p:parent+mutation` 起新谱系名」流程无法直接组合。现回落最优行作父代（stderr 注明父代 tag，可审计），用户新 tag 原样写入本次账本行。与 `--from-run` 联用且要切换单跑路径时显式 `--cv 0`（父代 `cv_folds` 会连带 CV 路径，SKILL 已注明）。

---

## [2.5.1] - 2026-08-16

> **主题**：CI 修复（自 v2.2.0 起慢性红灯收口）  
> **Release 正文**：[`docs/release-notes-v2.5.1.md`](docs/release-notes-v2.5.1.md)

### Fixed

- **CI wasm 构建**：`examples/wasm` 带 `//go:build js`，构建步骤缺 `GOOS=js GOARCH=wasm` → "build constraints exclude all Go files"。
- **CI lint job**：golangci-lint-action v6 不支持 golangci-lint v2 → 升 v7；lint 以 `GOOS=windows` 分析（加速面 helper 仅被 `//go:build windows` 的 GPU 文件调用，linux 构建会误判 unused；lint 自 v2.1.6 起 action 崩溃从未跑完，升 v7 后才暴露）。
- **CI Windows GPU panic**：runner 只有 WARP 软件设备，`IsAvailable()` 探测通过但运行时 `DXGI_ERROR_DEVICE_REMOVED`。新增环境变量 **`LEAVES_BORN_GPU=0|off|false`**（Windows）强制关闭 WebGPU 路径（训练与推理全部回落 CPU）；CI test job 已设置。
- **CRLF 测试解析**：`model` 测试在 CI Windows（autocrlf）下解析 testdata 尾部 `\r` 失败（`"-0.670\r"`）→ 解析前 TrimSpace。
- **runs.jsonl `elapsed_ms` 契约收紧**：账本行 **必带** `elapsed_ms`（去掉 omitempty；`0` 表示 <1ms）——原实现 sub-ms 训练会丢字段，与「每行必带耗时」的账本语义不符。

---

## [2.5.0] - 2026-08-16

> **主题**：Agent 演化搜索（GEPA 对标）+ 用户侧 Agent 入口 + `/v2` 文档对齐  
> **Release 正文**：[`docs/release-notes-v2.5.0.md`](docs/release-notes-v2.5.0.md)

### Added

- **演化搜索账本信号（EVO-02）**：`train` 的 metrics.json 与 runs.jsonl 账本行新增 `n_trees`（模型树数）、`elapsed_ms`（训练耗时毫秒）；`--cv` 时账本行新增 `fold_metrics`（各折指标）。供 Agent 做「指标 vs 模型大小/耗时」权衡、折级 Pareto 选父与筛选晋级（见 `skills/leaves-autotrain/SKILL.md` §4.5 演化搜索协议与 [`演进方案.md`](演进方案.md) §十六）。未存模型的 CV 路径 `n_trees` 省略。
- **SKILL §4.5 演化搜索协议（EVO-01/04）**：Hall-of-Fame + 折级 Pareto 选父、反射式变异（ASI → 假设 → 定向变异）、交叉重组、`--cv 2` 筛选→`--cv 5` 晋级、预算帽 ≤15、谱系 tag 约定（`p:<父>+<变异>` / `x:<A>|<B>`）。
- **用户侧 Agent 入口**：README「三步让 Agent 帮你训练」快速上手；`CLAUDE.md` 适配器（Claude Code 自动读 AGENTS.md）；SKILL walkthrough 以实跑数字重写（演示 §4.5 全协议）。

### Fixed

- README/README.en 安装命令：`go install github.com/linkerlin/leaves@latest` → `go install github.com/linkerlin/leaves/v2/cmd/leaves@latest`（原漏 `/v2` 模块路径与 `cmd/leaves`，仓库外无法安装）。
- 全仓文档模块路径对齐 `/v2`：godoc 徽章/链接、import 示例（README/README.en/api-surface/extension-points/versioning/serving-template/recsys-rank）、NOTES §4 过时的 `go get` 建议、`testscripts/compatibility_*.py` 的 require/replace 与 import 模板。

---

## [2.4.1] - 2026-07-11

> **主题**：模块路径迁移 `github.com/linkerlin/leaves` → `github.com/linkerlin/leaves/v2`

### Changed

- **模块路径迁移**（commit cab6a6f）：`go.mod` 声明 `github.com/linkerlin/leaves/v2`；后续 tag 打在 `/v2` 路径下。迁移前旧 tag 仍挂在无后缀路径。

---

## [2.4.0] - 2026-07-11

> **主题**：BackendAuto 第二轮 — opt-in profiling 探测  
> **Release 正文**：[`docs/release-notes-v2.4.0.md`](docs/release-notes-v2.4.0.md)

### Added

- **BackendAuto 第二轮：opt-in profiling**（`tree.ProfileBackend`）：
  - `ProfileBackend(caps, vals, nrows, ncols, iters) ProfileResult`：warm-up + 计时 Native / BornCPU / BornGPU，返回各后端实测 ns/op 与最快推荐（`Pick/Rule/Reason`，Rule 码 `profile_*`）。
  - **不改 2.0 默认决策表**（`SelectBackendExplained` 行为不变）；opt-in：调用方传代表性批量样本得到测量证据后再决定 Backend。
  - 不支持的后端（cat-small 森林 / 无 WebGPU）`Ok=false` 不参与推荐；Native 始终计时。

---

## [2.3.0] - 2026-07-11

> **主题**：完整 ONNX Graph 导入（复用 born 运行时，原列「明确不做」）  
> **Release 正文**：[`docs/release-notes-v2.3.0.md`](docs/release-notes-v2.3.0.md)

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
