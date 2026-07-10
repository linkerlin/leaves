# leaves 演进 TODO

> **对齐文档**：[`演进计划.md`](演进计划.md) v5.0（现状压缩 + 未来 12 个月路线图）
> **更新**：2026-06-20（P0–T5 + v3.1 已完成；本文件当前作为已完成 backlog 存档）
> **原则**：Native golden 不变；Born 直读 `ForestIR`；不做分布式/serving 框架。

**图例**：`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成 · `[-]` 明确不做

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
- [ ] **打 tag `v2.1.0` + push + GitHub Release**（人工确认后执行；正文用 release-notes）

