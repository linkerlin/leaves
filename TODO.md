# leaves 演进 TODO

> **对齐文档**：[`演进计划.md`](演进计划.md) v5.4（库线）· [`演进方案.md`](演进方案.md) v2.2（Agentic + §十六 演化搜索 + §十七 RC）  
> **更新**：2026-08-16（开发期；EVO 演化搜索 + AGUX 用户入口 + DOC 模块路径对齐；全量 `go test ./... -count=1` 28 包绿）  
> **原则**：Native golden 不变；Born 直读 `ForestIR`；不做分布式/serving 框架 / 内置 HPO / 官方 registry。

**图例**：`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成 · `[-]` 明确不做

---

## 现状快照（2026-08-16 · 开发期）

| 线 | 方案状态 | 代码/发布 | 结论 |
|----|----------|-----------|------|
| **Agentic** | Phase 0–5 + POST 加固 | tag **v2.1.0** … **v2.4.0** | **完成** |
| **演化搜索（EVO）** | 演进方案 §十六 | SKILL §4.5 协议 + 账本 `fold_metrics`/`n_trees`/`elapsed_ms` + 测试锁定 | **完成** |
| **用户 Agent 入口（AGUX）** | TODO AGUX 节 | 安装修复 / CLAUDE.md / 三步上手 / walkthrough 实跑重写 | **完成** |
| **文档对齐（DOC）** | TODO DOC 节 | 全仓 `/v2` 模块路径 / NOTES §4+§6 / CHANGELOG / testscripts | **完成** |
| **库线 Phase A–E** | 第一轮 + 按需深化 | 扩展点 / BackendAuto 2.0 / interop / ONNX 子集 / multi-target / explain 缓存 / serving 模板 | **完成** |
| **Demo** | MovieLens ranker + 四段 | meta 旁车 / Agent·MCP / `agent four-stage` | **完成** |
| **历史** P0–T5 / v3.1 | 存档 | 均已交付 | 开发期 |

**对照结论**：可执行 backlog **已清空**；Unreleased 已落盘。默认 **维护 + 按需开新项**；新需求先写进本文件再实现。发版走 [`docs/release-checklist.md`](docs/release-checklist.md)。

**最新 tag**：https://github.com/linkerlin/leaves/releases/tag/v2.5.5

---

## 后续工作（现行 backlog）

> 新工作请在本节追加 `[ ]` 项。开发期默认无打开主线；案例 demo 可并列追加。

### RC — 推荐生产闭环（演进方案 §十七，2026-08-21）

> 库内可测控制面契约（纯 Go、无运行时依赖）；指南 [`docs/recsys-loop.md`](docs/recsys-loop.md)。

- [x] **RC-00** 现状基线：`docs/recsys-loop.md` §1 精确区分离线 `deal` 与线上 decision
- [x] **RC-01** `recsys/contract`：事件/快照/策略 schema + 校验（重复 ID / 未知类型 / 反向时间 / 非 UTC / PII 绊线均失败）
- [x] **RC-02** 文件/特征指纹进 snapshot（`HashFile`/`FeatureSchemaHash`/`VerifyFiles`：任一 hash 不匹配即失败）
- [x] **RC-03** `recsys/split`：时间切分 + as-of 泄漏门禁（`CheckLeakage`）+ cold-start 分层
- [x] **RC-04** `recsys/eval`：三层指标（data/candidate_rank/deal）+ 业务可配置阈值；无阈值只能 exploratory
- [x] **RC-05** `recsys/ledger`：决策/曝光/反馈 JSONL 账本（append 校验回链；`DecisionFromDeal` 原因码映射）
- [x] **RC-06** `recsys/replay`：归因窗 + 迟到/孤立/抑制排除 + 去重；确定性输出 + `replay_report.json`
- [x] **RC-07** `recsys/monitor`：窗口聚合（ctr/orphan/deck 质量）+ ok/warn/block + reason code；**触发器 `TriggerSet`**（连续越界 + 冷却期 + 恢复重置 → retrain/rollback_requested，§17.6 配置驱动）
- [x] **RC-08** `recsys/release`：状态机（exploratory→…→rollback_requested）+ evidence 校验（三层门禁齐全 + 模型 hash 一致）+ last_known_good 不可漂移
- [x] **RC-09** `release.Adapter` 接口 + `FakeAdapter`（只产出推广/回滚请求，无网络副作用）；真实 adapter 留给应用仓库
- [x] **RC-10** `skills/recsys-orchestrator` §十·八段剧本 + 三状态口径（离线提升/可推广候选/已上线观察）；镜像已同步
- [x] **RC-11** `recsys/loop/TestAgenticRecsysLoopDrill`：八步闭环演练（快照→门禁→promoted→账本→健康观察→退化注入→**触发器**回滚指向 last_known_good（含冷却抑制）→replay→retrain）
- [x] **RC-12** README（RC 控制面小节）/ AGENTS / serving-template 对接说明 / `docs/recsys-loop.md`

**验收**：

```powershell
go test ./recsys/... -count=1        # 含 contract/split/eval/ledger/replay/monitor/release/loop
go test ./docs -run TestSkillsMirrorSync -count=1
```

**遗留（不立项，按需开）**：`recsys/cmd` CLI 化控制面子命令；prep 主线改时间切分（现为 user 切分 + split 包独立提供）；自动推广配置（需应用侧签名/访问控制前提）。

---

### LINT-DEBT — 渐进式 lint 启用

> 门禁已全量落地（`.golangci.yml`：govet[关闭 composites] + ineffassign + **unused** + **staticcheck** + **errcheck** + **gofmt** formatter；CI `lint` job blocking）。
> 基线（2026-07-11）全部清零：~~errcheck 50~~ / ~~unused 21~~ / ~~staticcheck 20~~ / ~~gofmt 176~~。

- [x] **LINT-1** errcheck：~~存量 ~50 处~~。**已清零并启用 blocking**。用 v2 `std-error-handling` 预设排除惯例忽略项（Close/Flush/Fprint*/stdout.Write/Remove）；测试文件按 `path` 规则豁免（测试靠值断言）；生产路径 10 处真实忽略改 `_ =` 显式丢弃（transform.Transform、engine.Predict、predictLeafIndicesInner、transform、Seek、Iterate）。
- [x] **LINT-2** unused：~~存量 ~20 处~~。**已清零并启用 blocking**（删 21 处死代码：explain interventional SHAP 簇 + treeShapFast/computeNodeWeights、treebuilder 被 batchAccumulateHistWebGPU 取代的 GPU hist 旧路径 accumulateHistWebGPU×3/prebuildGPUHists/gpuHistBuildEnabled/gpuHistMinSamples、各处孤立 helper）。
- [x] **LINT-3** staticcheck：~~存量 ~20 处~~。**已清零并启用 blocking**。修 12 处真实项（SA4006 xgensemble 死赋值、SA9003×2 空分支、S1002 bool 比较、S1009 nil-len、S1021 递归闭包合并、QF1004 ReplaceAll、QF1008/QF1011）；关闭 QF1003（tagged-switch 风格，冻结兼容层不宜改写）+ ST1000/ST1003（包注释/命名风格）。
- [x] **LINT-4** 全仓 gofmt 归一：**已完成并锁定 gofmt formatter**。`gofmt -w .`（176 处 CRLF→LF + 对齐）+ `gofmt -s -w`（1 处 composite literal 类型名冗余）；CI `lint` job 经 `golangci/golangci-lint-action` 自动应用 formatter。

### Demo / 教程

- [x] **DEMO-ML** MovieLens 推荐 ranker 全流程 + Agent/MCP：[`demos/movielens/TUTORIAL.md`](demos/movielens/TUTORIAL.md)、`cmd/agent`、`cmd/mcp`、`agentops`
- [x] **DEMO-ML-meta** 推荐片名旁车 + walkthrough（v2.1.4）
- [x] **DEMO-ML-4stage** MovieLens → recsys 四段（prep/召回/LTR/发牌）：`recsys/movielens` + `agent four-stage`（**v2.1.5**）

### P0 — 契约与门禁残留（演进方案建议未落地）

对照 [`演进方案.md`](演进方案.md) §10.3 / WP-06 风险缓解 / WP-10 错误表。

| ID | 项 | 方案依据 | 现状（代码） | 建议 |
|----|----|----------|--------------|------|
| **POST-01** | skills 镜像 CI 哈希一致 | §10.3 / WP-06 | 镜像入库 + `TestSkillsMirrorSync` + CI step | **[x]** |
| **POST-02** | `cv_conflict` 可行动错误 | WP-10 | `--strict-flags` → `cv_conflict`；默认仍警告 | **[x]** |
| **POST-03** | 演进方案 WP / 附录 A 与 DoD 对齐 | §八 vs §十四 | 演进方案 v1.5 | **[x]** |

- [x] **POST-01** CI：`skills/` ↔ `.cursor/skills/` 内容哈希门禁（`TestSkillsMirrorSync` + 镜像入库；`.gitignore` 例外）  
- [x] **POST-02** `train`：`--strict-flags` → `error=cv_conflict`（默认仍仅警告）  
- [x] **POST-03** 演进方案正文 WP 验收勾选 / 附录 A 与 DoD 同步（v1.5）

**验收**：

```powershell
# POST-01 落地后
# CI 或本地：Compare-Object / Get-FileHash skills vs .cursor/skills
go test ./cmd/leaves -run 'Agentic|SaveBest|FromRun|NAPolicy|Publish' -count=1
```

---

### P1 — Agent 体验加固（方案可选/半完成）

| ID | 项 | 方案依据 | 现状 | 建议 |
|----|----|----------|------|------|
| **POST-10** | 高频错误码表 + 测试 | WP-10 | `TestErrorCodesHighFrequency` + cli.md 表 | **[x]** |
| **POST-11** | `--print-repro` | WP-11 | `publish --print-repro` + 测试 | **[x]** |
| **POST-12** | `--out-final` 侧车 | WP-03 方案 C | best=`--out-model`，final=截断前 | **[x]** |
| **POST-13** | metrics.`train_accel` | §12.3 | Fit 后写入；与推理无关 | **[x]** |
| **POST-14** | data_quality 扫描语义文档 | WP-12 | cli.md 写清 `sniffMaxScan=5000` | **[x]** |
| **POST-15** | save-best 截断语义写实 | §13.1 | cli.md + SKILL | **[x]** |

- [x] **POST-10** 错误 JSON 负面契约测试 + cli.md 错误码表（`TestErrorCodesHighFrequency`）  
- [x] **POST-11** `publish --print-repro` stdout 复现命令（`TestPublishPrintRepro`）  
- [x] **POST-12** `train --out-final`：早停截断前另存 final；metrics.`final_model`/`final_round`  
- [x] **POST-13** metrics.`train_accel`（Fit 后生效模式；与推理 Backend 无关）  
- [x] **POST-14** data_quality 扫描语义文档（`sniffMaxScan=5000`；可配置 flag 仍可选）  
- [x] **POST-15** save-best 实现语义写实（内存截断非重训；cli.md + SKILL）

---

### P2 — 库线按需深化（演进计划 §七 / 第一轮之后）

> Phase A–E **第一轮**已完成；以下默认 **不进主线**，有明确用户需求再开。

#### BackendAuto / Bench 第二轮

- [x] **LIB-01** 第二轮候选边界写入 [`docs/backend-auto.md`](docs/backend-auto.md)（**不**实现 profiling；有证据再开）  
- [x] **LIB-02** BenchRecord 样例工件 → [`docs/bench/`](docs/bench/) + `TestBenchSampleArtifact`  
- [x] **LIB-03** 训练 `LEAVES_TRAIN_ACCEL` 与推理 BackendAuto 交叉说明 → [`docs/backend-auto.md`](docs/backend-auto.md) §训练 vs 推理

#### 互操作 / 格式

- [x] **LIB-10** ONNX TreeEnsembleRegressor **实验子集**（BRANCH_LEQ/SUM/NONE；`SampleONNXStump` + 测试）
- [x] **LIB-11** scikit-learn：协议矩阵收窄（interop-matrix）+ `TestSklearn*` 失败/探测/golden  
- [x] **LIB-12** `TestTestdataMatrixArtifactsExist`：testdata-matrix 反引号工件对账

#### 扩展点 / 训练

- [x] **LIB-20** [`examples/extension/`](examples/extension/)：`custom:l1` + `max_abs_error`（Register 可跑 + 测试）  
- [x] **LIB-21** Multi-target 训练 **one_output_per_tree**（API + CLI `--num-target`；向量叶 multi_output_tree 仍不做）
- [x] **LIB-22** explain 性能：节点权重切片缓存 + 背景 margin 缓存 + path/phi 缓冲复用；`BenchmarkTreeSHAP*`

#### 部署 / 生态（库外）

- [x] **LIB-30** [`examples/serving-template/`](examples/serving-template/)：可拆独立仓脚手架（http 仍为最小 embed）
- [x] **LIB-31** registry 对接模板（S3 / gh release / curl / OCI）→ `skills/leaves-autotrain/cli.md`（**仅文档**）

---

### EVO — 演化搜索升级（演进方案 §十六，2026-08-15）

> GEPA（arXiv:2507.19457）对标：把 Agent 搜索从贪心坐标下降升级为「带反射的演化搜索」。策略在 SKILL，库只加信号。

- [x] **EVO-01** SKILL §4.5 演化搜索协议：Hall-of-Fame + 折级 Pareto 选父、反射式变异（ASI → 假设 → 定向变异）、交叉重组、筛选→晋级（`--cv 2`→`--cv 5`）、预算帽 ≤15、谱系 tag 约定（含 EVO-04）  
- [x] **EVO-02** 账本信号：runs 行补 `fold_metrics` / `n_trees` / `elapsed_ms`（`cmd/leaves/common.go` + `train.go`；cli.md schema 同步）  
- [x] **EVO-03** 优化环测试扩展：`TestAgenticOptimizeLoopSmoke` 锁定账本信号（n_trees/elapsed_ms/fold_metrics；CV 未存模型时 n_trees 合法省略）  
- [x] **EVO-04** 谱系：并入 EVO-01 tag 约定（`p:<父>+<变异>` / `x:<A>|<B>`），零代码  

**验收**：

```powershell
go test ./cmd/leaves -run "Agentic" -count=1
go test ./docs -count=1   # 镜像 + 文档版本引用门禁
```

**远期观察（不立项）**：跨任务 lessons 记忆（项目级 `lessons.md`，AlphaEvolve 式自改写决策表的雏形）。

---

### DOC — 文档对齐模块迁移 `/v2`（2026-08-16）

- [x] **DOC-01** 全仓文档 import/godoc/install 对齐 `github.com/linkerlin/leaves/v2`：README(.en)（godoc 徽章+链接、6 处 import 示例）、docs/api-surface、docs/extension-points、docs/versioning、examples/serving-template、skills/recsys-rank（SKILL + leaves-api；镜像已同步）  
- [x] **DOC-02** NOTES.md §4 重写（模块已迁 `/v2`，旧 `go get` 建议已失效）+ 新增 §6「信号字段只增不删」契约兼容说明（演进方案 §9.6 流程）  
- [x] **DOC-03** CHANGELOG Unreleased 补 EVO 账本信号 / SKILL §4.5 / 安装修复  
- [x] **DOC-04** `testscripts/compatibility_*.py`：require/replace 与 Go import 已按 `/v2` 修正（`py_compile` 过；`require /v2 + replace => 本地路径` 模式与 cases 实际 API 已用临时模块 `go build` 实证）。注意：harness 本身 POSIX-only（venv `bin/python`、`./executable`），Windows 仅能做语法/模式验证，全量跑需 Linux/macOS  

---

### CIFIX — CI 慢性红灯收口（v2.5.1，2026-08-16）

> 自 v2.2.0 起 CI 恒红（lint action 版本、wasm GOOS、CI Windows WARP GPU panic、CRLF 解析）。与 v2.5.0 同日修复发版。

- [x] **CIFIX-01** CI wasm job：构建步骤补 `GOOS=js GOARCH=wasm`（`examples/wasm` 带 `//go:build js`）  
- [x] **CIFIX-02** CI lint job：golangci-lint-action v6 → v7（v6 不支持 golangci-lint v2）；并以 `GOOS=windows` 分析——加速面 helper 仅被 `//go:build windows` 的 GPU 文件调用，linux 构建误判 unused（本地 Windows lint 0 issues 复核）  
- [x] **CIFIX-03** 新增 `LEAVES_BORN_GPU=0|off|false`（Windows）：`tree.BornWebGPUAvailable()` 强制 false，训练+推理 WebGPU 全回落 CPU；CI test job 设置该变量（WARP 探测通过但运行时 `DXGI_ERROR_DEVICE_REMOVED`）；`docs/backend-auto.md` 已文档化  
- [x] **CIFIX-04** `model/predict_contrib_p0_test.go`：TSV 解析前 TrimSpace（CI Windows autocrlf 下 `"-0.670\r"` 失败）  
- [x] **CIFIX-05** runs.jsonl `elapsed_ms` 去 omitempty：账本行必带（0=<1ms；CI 小数据 sub-ms 丢字段曾挂 TestAgenticOptimizeLoopSmoke）  
- [x] **AGUX-06** `--from-run --tag <新tag>` 回落最优行（v2.5.2）：原硬错阻断 §4.5 谱系流程；仓库外 `go install` 二进制用户路径实测全通；`--cv 0` 切换单跑路径已注明 SKILL/cli.md  
- [x] **AGUX-07** `leaves version` 子命令（v2.5.3）：`debug.ReadBuildInfo` 输出 `{version, go[, commit]}` JSON——`go install pkg@tag` 用户/Agent 可自查装的版本（证据：v2.5.2 验证时 `leaves version` 报「未知子命令」）；SKILL 速查卡 8→9 命令 + cli.md 新节  
- [x] **AGUX-08** manifest 复现契约修复（v2.5.4）：`buildReproduceCommand` 补 `--cv/--max-leaves/--num-target/--val/--early-stop`（原 CV run 复现退化为全量单训）；`params.val` 记录（仅单跑路径）；`leaves_cli` 真版本替代占位 `agentic-1`；复现语义分工写实（`--from-run` 定稿不回填 val / `manifest.reproduce` 忠实重放）——发现途径：v2.5.2 用户路径 manifest 复核  
- [x] **AGUX-09** module zip 瘦身（v2.5.5，21.8MB→~5MB）：fresh clone 审计发现 git 追踪文件=go get 分发内容——删孤儿 `bin/*.exe`（36MB 零引用）/可重建 `examples/wasm/leaves.wasm`（3.5MB）/.chong/（他工具记忆，经确认移除）；`.gitignore` 补三条；testdata（回归矩阵必需）保留  

**验收**：本地 `LEAVES_BORN_GPU=0 go test ./tree ./model` 绿；wasm 交叉编译过；CI 全 job 绿（v2.5.1 起）。

---

### AGUX — 用户侧 Agent 入口（2026-08-16）

> 目标：用户「装好 CLI → 把技能给 Agent → 说一句话」即开工。全部为文档/入口层，无策略代码。

- [x] **AGUX-01** README 安装命令修正：`go install github.com/linkerlin/leaves/v2/cmd/leaves@latest`（原漏 `/v2` 与 `cmd/leaves`）；模块路径文字同步  
- [x] **AGUX-02** [`CLAUDE.md`](CLAUDE.md) 适配器（`@AGENTS.md` 单行导入）——Claude Code 开箱读规约  
- [x] **AGUX-03** README Agent 段新增「三步让 Agent 帮你训练」用户快速上手（装 CLI → 给技能 → 一句话开工）  
- [x] **AGUX-04** SKILL walkthrough 以实跑数字重写（2026-08-16）：演示 §4.5 全协议——谱系 tag（`p:baseline+depth6_lr01+lambda2`）、反射轮（ASI：importance f1=0 + 曲线形态 → 假设 → 定向 lambda 变异 0.218→0.2129）、收敛判据（mcw 零改进）、`--from-run` 全量定稿 + `--emit-repro-script`；安装命令入 SKILL §三  
- [x] **AGUX-05** examples/autotrain README 对齐 §4.5（账本新字段 + 谱系 tag 提示）

**验收**：

```powershell
go test ./cmd/leaves ./docs -count=1   # 契约 + 镜像 + 文档门禁全绿
```

---

### 维护 — v2.1.x / 下一版本

- [x] **MNT-01** v2.1.1 / **v2.1.2** 已按 checklist 发版（tag + GitHub Release + CHANGELOG）  
- [x] **MNT-02** 模块路径无 `/v2` 时代理可能拒 `v2.1.x` tag → 已记 [NOTES.md](NOTES.md) §4
- [x] **MNT-03** 发版前关键包测试已跑；全量 `go test ./...` 仍建议 CI  
- [x] **MNT-04** skills 镜像 CI 门禁（`TestSkillsMirrorSync`）；改 SKILL 须同步 `.cursor/skills`  
- [x] **MNT-05** README badge / CHANGELOG 已对齐 **v2.1.3** … **v2.1.5**
- [x] **MNT-06** v2.1.5 发版：四段 MovieLens + TODO 快照 / Unreleased 收口

**快速回归**：

```powershell
go test ./... -count=1
go test ./cmd/leaves ./tree ./io ./objective ./metrics -count=1
go test ./recsys/pipeline/... -run TestSmokePipeline100PerUser -count=1
```

---

## 已完成存档（勿再拆为主线）

以下为历史 backlog，全部 `[x]`/`[-]`，仅供检索。

---

## P0 — v1.0 发布阻塞（推理语义闭环）✅

（略，均已完成 — 见 git history）

**验收**：`go test ./model/... ./explain/... -count=1`

---

## P1 — 产品化与 Born（Phase 1 / B4）✅

（略，均已完成）

**验收**：

```powershell
go test ./... -count=1
go test ./treebuilder/... -count=1
```

---

## P2 — 格式与训练进阶 ✅

### metrics 补齐 ✅

### 训练 T5 — 排序与约束 ✅

### IO 元数据补全 ✅

### P2 工程债

- [x] 全 testdata 回归矩阵文档化 → [`docs/testdata-matrix.md`](docs/testdata-matrix.md)
- [x] Benchmark 套件 CI 基线门禁 → [`docs/benchmark-baseline.md`](docs/benchmark-baseline.md) + `bench-gate` job + `TestBenchGateBornCPUSlowerBatch1`
- [-] 根包 IO 迁入 `io/`（`lgensemble_io.go` 等）— 破坏性大，根包保留兼容层
- [x] 根包全 API 委托 `model.Ensemble`（Predict* + IO 经 `LoadFromFile`/`legacy_ensemble`）
- [x] 文档：后端选择速查（README §计算底座）
- [x] `objective`/`metrics` Registry 插件化（`Register` + `builtins.go` init；多分类/排序仍 switch 构造）

---

## P3 — 部署与观测 ✅

### WASM

- [x] `GOOS=js GOARCH=wasm` 构建验证（Native CPU fallback）
- [x] `examples/wasm/`：HTML + 批预测 demo
- [x] 部署指南：模型体积、冷启动、batch 建议（`examples/wasm/README.md`）
- [x] 部署性能报告（文档化 + 手动 bench 指引，见 `docs/benchmark-baseline.md`）

### 量化 ✅

### 观测钩子 ✅

### ONNX（非主路径）

- [x] `io/onnx.go` 调研占位 + `ErrONNXNotImplemented`

### P3 可选深化

- [x] `predict.Request` 级耗时钩子 → `model.PredictWithProfile`
- [x] HTTP embed 中间件示例 → `examples/http/`

---

## T5 余下 — 训练完备 ✅

- [x] `ExternalMemoryMatrix` → `train.Learner` / `treebuilder` 接线（`BatchedMatrix` + global hist）
- [x] `survival:cox` / `survival:aft` 目标函数
- [x] `reg:tweedie` 训练目标
- [x] Checkpoint **续训**（`LoadCheckpoint` / `ResumeFit`）
- [x] `Learner.Eval(dm)` 公开 API
- [x] `data.FromCSV` / `FromCSVReader`
- [x] `max_leaves` / lossguide 生长策略
- [-] Multi-output tree **训练**（推理 `OutputDim` 已有；训练未排期）
- [-] `train.HyperparamSearch`（文档指向外部 Optuna，不做内置）

**验收**：

```powershell
cd testdata && python gen_rank_pairwise_grad.py && python gen_rank_ndcg_grad.py
go test ./objective/... ./train/... -short -count=1
go test ./train/... -run 'Rank|Monotone|Callback|Resume|Eval|MaxLeaves|Tweedie|Survival' -count=1
```

---

## 明确不做（[-]）

- [-] Spark / Dask / Ray / Federated / Rabit 分布式训练
- [-] CGO 绑 `libxgboost` / 复刻 `c_api.h`
- [-] 官方 HTTP/gRPC serving 框架（`examples/http` 为 embed demo）
- [-] 内置 Optuna/网格搜索
- [-] CUDA 直连推理（Born WebGPU 为 Windows GPU 路线）
- [-] inplace_predict / staged cache
- [-] 根包 IO 物理迁移（见 P2）
- [-] Multi-output tree 训练（见 T5）

---

## 迭代建议顺序（2026-06-15 更新）

```
1. ✅ P0 contrib + 文档同步
2. ✅ P1 Born B4 + parity 矩阵
3. ✅ P2 metrics + T5 rank/单调
4. ✅ P3 quantize + profile + Reload
5. ✅ P3 WASM demo + 部署指南
6. ✅ T5 外存 DMatrix 接线
7. ✅ T5 survival / tweedie + 续训 API
8. ✅ P2 工程债：testdata 矩阵文档 + benchmark CI
```

---

## 快速验收命令

```powershell
# 全量回归
go test ./... -count=1

# 训练 + WebGPU hist 加速（无需 born_train tag）
go test ./treebuilder/... -count=1
go test ./train/... -short -count=1

# parity / 量化 / bench
go test -run Parity -count=1
go test ./quantize/... -count=1
go test -run TestBenchGateBornCPUSlowerBatch1 -count=1
```

---

## v3.1 后续（可选深化）

> P0–T5 backlog 已清空；以下为产品化与互操作增强，按优先级排列。

- [x] `data.FromLIBSVM` / `data.FromFile` 统一训练数据入口
- [x] `survival:aft` 区间删失标签（`AFTIntervalMatrix` / `data.AFTDense`）
- [x] `examples/http` 批预测 + JSON 矩阵输入
- [x] WASM CI 体积报告（`leaves.wasm` 大小门禁 ≤16MiB，`TestWasmBinarySizeGate`）
- [x] 根包 `train` 类型别名（`train_api.go`：`Learner`/`NewLearner`/`ResumeFit`/`FitExternal`）
- [x] 训练数据内容嗅探（`data/sniff.go`：`FromFileAuto` / `LoadDataAuto`）
- [x] 模型加载 `AutoTransform` 默认 + 经典 XGB 二进制 header 探测（`io/transform_auto.go`）
- [x] `NewLearnerFromModelAndData` 端到端（`train/load_test.go`, `examples/train_from_model/`）
- [x] 演进计划 v4.3 嗅探/AutoTransform 同步

---

## Agentic 收口（见 [`演进方案.md`](演进方案.md) v1.0）

> 目标：契约诚实 → 定稿正确 → 技能可发现 → 优化环可回归。  
> 更新：2026-07-10

### Phase 0–1（最小宣称集核心）

- [x] **WP-01** sniff `n_features` = `Matrix.NumCol()`；`has_qid`；与 `feature_names` 一致
- [x] **WP-02** `paramsRecord` 完备（min_child_weight/gamma/subsample/colsample/max_bin/…）
- [x] **WP-00** 契约测试 `cmd/leaves/contract_test.go`
- [x] **WP-03** `--save-best`（默认 true）+ `ForestIR.TruncateToNEstimators` + metrics `model_round`/`stopped_round`
- [x] **WP-04** SKILL / cli.md 定稿语义同步
- [x] **WP-05** save-best 单测（`TestSaveBestDefault` / `TestSaveBestFalse`）
- [x] **WP-06** `.cursor/skills/leaves-autotrain` 镜像
- [x] **WP-07** 多轮优化环集成测试 `TestAgenticOptimizeLoopSmoke`
- [x] **WP-08** examples/autotrain walkthrough 数字刷新（2026-07-10 实跑；路径纪律 out-model≠metrics）
- [x] **WP-09** AGENTS / 演进计划交叉引用

### Phase 3–4（加固 / 宣称）

- [x] **WP-10** `--error-format json` / `LEAVES_ERROR_FORMAT` + `agentError` 分类
- [x] **WP-11** manifest：`reproduce` / `schema_version` / `leaves_cli` / `publish_note`
- [x] **WP-12** sniff `data_quality`（常数列 / nan / 不均衡 / small_n）
- [x] **WP-13** autotrain vs recsys 边界表（SKILL §八）
- [x] **WP-14–16** DoD 最小集已勾选；README Agentic 段已写实边界与演进方案链接

### Phase 5（可选深化）✅

- [x] **WP-17** `train --from-run runs.jsonl [--tag]` 一键复现（CLI 覆盖优先；`TestFromRunReproduce`）
- [x] **WP-18** `schema_version`（metrics / sniff / manifest）
- [x] **WP-19** publish `--emit-repro-script ps1|sh|both` → `reproduce.ps1` / `.sh`
- [x] **WP-20** `--na-policy error|skip-row`（默认 error；CSV 缺失丢行，不做插补）
- [x] **WP-21** 文档地图：README / 演进计划 / 演进方案 职责拆分，避免双叙事

---

## 库线 12 个月（见 [`演进计划.md`](演进计划.md) v5.4）

> Agentic 已收口；库线 Phase A–E 第一轮已落地。  
> 更新：2026-07-10

### Phase A — 文档与入口收口 ✅

- [x] 文档地图（README / 演进计划 / 演进方案 / TODO 职责）
- [x] Agentic 专项与库路线分离叙述
- [x] README / README.en 推荐入口与 BackendAuto 2.0 对齐
- [x] NOTES 历史叙述瘦身（兼容注记 + 指向 TODO/api-surface）

### Phase B — 扩展点收口

- [x] objective 全量 Register（multi:* / rank:*；`ByNameWithClass` 无 switch 回退）
- [x] metric 全量 Register（mlogloss/merror；`ndcg@K`/`map@K` 前缀解析）
- [x] 扩展开发文档 [`docs/extension-points.md`](docs/extension-points.md)
- [x] booster / tree method：**明确不做 registry**（文档边界说明）
- [x] 回归：`TestBuiltinRegistryComplete` / `TestResolveViaRegistryOnly`

### Phase C — BackendAuto 2.0 ✅（2026-07-10）

- [x] BackendAuto 2.0：CPU 阈值 64、GPU 256、SparseDensity、GPU 不可用回落 BornCPU
- [x] `SelectBackendExplained`（Rule + Reason）
- [x] 决策表文档 [`docs/backend-auto.md`](docs/backend-auto.md)
- [x] `tree.BenchRecord` 统一 JSONL 记录格式
- [x] 测试：`TestBackendAutoDecisionTable` 等；io 大 batch 路径同步

### Phase D — 互操作边界 ✅（2026-07-10）

- [x] 支持等级表 [`docs/interop-matrix.md`](docs/interop-matrix.md) + `io.SupportOf` / `SupportTable`
- [x] `*io.LoadError` 可操作 hint（detect/load）；`.onnx` 探测与占位失败
- [x] SK=实验、ONNX=占位策略写死；testdata-matrix / README 对齐
- [x] 测试：`TestSupportTableComplete` / `TestLoadONNXActionableError` 等

### Phase E — v2.1 发布准备 ✅（2026-07-10）

- [x] [`docs/release-checklist.md`](docs/release-checklist.md) 发版勾选
- [x] [`docs/versioning.md`](docs/versioning.md) v2.x 允许/禁止变更
- [x] [`docs/api-surface.md`](docs/api-surface.md) 推荐 / 兼容 / 实验 + 迁移
- [x] NOTES 瘦身；README 文档表更新
- [x] `docs.TestReleaseDocsPresent` 锁定关键文档存在
- [x] [`CHANGELOG.md`](CHANGELOG.md) Unreleased 汇总；README 中英互链修复
- [x] CHANGELOG 落成 `[2.1.0]` + [`docs/release-notes-v2.1.0.md`](docs/release-notes-v2.1.0.md)
- [x] **打 tag `v2.1.0` + push + GitHub Release**（2026-07-10；https://github.com/linkerlin/leaves/releases/tag/v2.1.0）

---

## 明确不做（现行，与方案一致）

> 完整历史列表见存档「明确不做」；下列为 **v2.1 后仍成立** 的边界。

- [-] 内置 HyperparamSearch / Optuna / 网格搜索 / AutoML 服务（搜索在 SKILL + Agent）
- [-] MCP server 作为闭环必需依赖
- [-] 官方 model registry / 云端实验板 / OCI 推送（publish = 本地工件包）
- [-] 分布式训练（Spark / Dask / Ray / Federated / Rabit）
- [-] 官方 HTTP/gRPC serving 框架（`examples/http` 仅为 embed demo）
- [-] 完整 ONNX Graph 导入（LIB-10 仅为 TreeEnsembleRegressor 子集）
- [-] 根包 IO 物理迁移进 `io/`（兼容层保留）
- [x] Multi-output **向量叶** 训练（`multi_output_tree` / `OutputDim>1` 生长）：**已实现 2026-07-11**（原列「明确不做」，A+B 落地。treebuilder 统一 k 维数学 + leaf-major flatten；booster `MultiOutputTree` 跳过 demux 建向量叶树；accel 对 k>1 回退 CPU。详见 CHANGELOG [Unreleased]）
- [-] CUDA 直连推理（GPU 路线 = Born WebGPU / Windows）
- [-] 把 recsys 召回/发牌并进 `cmd/leaves` 主 CLI
- [-] 完整特征存储 / 实时特征平台
- [-] 任意脏数据零清洗即训（类别/文本须库外预处理；`--na-policy skip-row` 仅丢行）

---

## 建议执行顺序（v2.1 后 · 已全部完成）

```text
1. ✅ POST-01…15 全套加固
2. ✅ LIB-01…03 / 20 / 31
3. ✅ LIB-10/11/12/21 + v2.1.1
4. ✅ LIB-22 explain 性能 + LIB-30 serving-template + v2.1.2
5. ✅ 进入开发期（本文件无打开 backlog）
```

### 按需可开（默认不做，需产品信号）

- 向量叶 `multi_output_tree` **训练**（推理/加载已有）
- 完整 ONNX Graph
- BackendAuto 第二轮 profiling
- 模块路径迁 `.../leaves/v2`（MAJOR）
- 独立 serving 产品仓（模板已在 `examples/serving-template`）