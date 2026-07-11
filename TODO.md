# leaves 演进 TODO

> **对齐文档**：[`演进计划.md`](演进计划.md) v5.4（库线）· [`演进方案.md`](演进方案.md) v1.4（Agentic）  
> **更新**：2026-07-11（含 LIB-10 ONNX 子集 + LIB-21 multi-target 训练）  
> **原则**：Native golden 不变；Born 直读 `ForestIR`；不做分布式/serving 框架 / 内置 HPO / 官方 registry。

**图例**：`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成 · `[-]` 明确不做

---

## 现状快照（2026-07-11）

| 线 | 方案状态 | 代码/发布 | 结论 |
|----|----------|-----------|------|
| **Agentic**（演进方案 Phase 0–5） | DoD D1–D10 + R1–R5 全绿 | `cmd/leaves` 契约/save-best/from-run/na-policy/error-json；tag **v2.1.0** | **收口完成** |
| **库线**（演进计划 Phase A–E 第一轮） | 文档/扩展点/BackendAuto 2.0/互操作/发版治理 | `docs/*` + Register + `SelectBackendExplained` + Release | **第一轮完成** |
| **历史** P0–T5 / v3.1 | 见下方存档 | 均已交付 | 维护期 |

**对照结论**：演进方案「明确达成」集合已实现；方案正文 §八 部分 WP 细项 checkbox 仍为历史 `[ ]`（与 §十四 DoD `[x]` 不一致，属文档债）。后续工作 = **方案残留加固 + 库线按需深化 + 维护**，不再扩 AutoML/registry。

---

## 后续工作（现行 backlog）

> 优先级：P0 契约/门禁债 · P1 Agent 体验 · P2 库/性能按需 · 维护。  
> 非目标见文末「明确不做」。

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
- [ ] **LIB-11** scikit-learn：协议矩阵收窄文档 + 失败用例；不扩全版本  
- [ ] **LIB-12** 稳定格式加载 **golden 矩阵** 与 `testdata-matrix` 自动对账（防文档漂移）

#### 扩展点 / 训练

- [x] **LIB-20** [`examples/extension/`](examples/extension/)：`custom:l1` + `max_abs_error`（Register 可跑 + 测试）  
- [x] **LIB-21** Multi-target 训练 **one_output_per_tree**（API + CLI `--num-target`；向量叶 multi_output_tree 仍不做）
- [ ] **LIB-22** explain 大模型 **性能**优化（能力已完整，按需）

#### 部署 / 生态（库外）

- [ ] **LIB-30** 独立 serving **示例仓库**或模板（本仓继续只保留 `examples/http` embed）  
- [x] **LIB-31** registry 对接模板（S3 / gh release / curl / OCI）→ `skills/leaves-autotrain/cli.md`（**仅文档**）

---

### 维护 — v2.1.x / 下一版本

- [ ] **MNT-01** 按 [`docs/release-checklist.md`](docs/release-checklist.md) 跑 v2.1.1+ 热修流程（契约变更必须升 `schema_version`）  
- [ ] **MNT-02** 验证模块代理 `go get github.com/linkerlin/leaves@v2.1.0`（代理滞后时记 NOTES）  
- [ ] **MNT-03** 发版前：`go test ./... -count=1` + `./cmd/leaves` + wasm/bench-gate CI 绿  
- [ ] **MNT-04** 改 SKILL 时同步 `.cursor/skills`（在 POST-01 落地前靠人工/PR 模板）  
- [ ] **MNT-05** README badge / CHANGELOG Unreleased 与下一 tag 对齐  

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
- [-] Multi-output **向量叶** 训练（`multi_output_tree` / `OutputDim>1` 生长；推理与 XGB 加载已有；训练为 one_output_per_tree）
- [-] CUDA 直连推理（GPU 路线 = Born WebGPU / Windows）
- [-] 把 recsys 召回/发牌并进 `cmd/leaves` 主 CLI
- [-] 完整特征存储 / 实时特征平台
- [-] 任意脏数据零清洗即训（类别/文本须库外预处理；`--na-policy skip-row` 仅丢行）

---

## 建议执行顺序（v2.1 后）

```text
1. ✅ POST-01…15 全套加固
2. ✅ LIB-01(文档边界) / 02 / 03 / 20 / 31
3. ✅ LIB-10 ONNX 子集 / LIB-21 multi-target one_output_per_tree
4. LIB-22 explain 性能 / 向量叶训练深化     ← 按需
5. LIB-30 独立 serving 仓                  ← 库外
6. MNT：发版 v2.1.1 时走 release-checklist
```
