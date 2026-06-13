# leaves 演进 TODO

> **对齐文档**：[`演进计划.md`](演进计划.md) v4.0（XGBoost 3.3 全链路对标）  
> **更新**：2026-06-13  
> **原则**：Native golden 不变；Born 直读 `ForestIR`；不做分布式/serving 框架。

**图例**：`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成 · `[-]` 明确不做

---

## P0 — v1.0 发布阻塞（推理语义闭环）✅

### 预测 API 统一出口（对标 `doc/prediction.rst`）

- [x] `model/predict.go`：实现 `OutputContribution` → `explain.TreeSHAP` 接线
- [x] `model/predict.go`：实现 `OutputApproxContribution` → Saabas
- [x] `model/predict.go`：实现 `OutputInteraction` → 交互 SHAP
- [x] 测试：`predict_contribs` 可加性 + `model/predict_contrib_test.go`
- [x] 测试：leaves 自洽 golden 逐元素 bit-exact（`shap_contribs_leaves.tsv`）
- [x] 测试：XGBoost golden 可加性 + margin 一致（逐元素因 SHAP 分解不同，不对齐）
- [x] 测试：多类 / margin 空间 / `base_score` bias 语义（`predict_contrib_p0_test.go`）
- [x] 文档：README `predict.Request` contrib 示例
- [x] 文档：README 全面迁移到 `predict.Request`（value/margin/leaf 示例）

**验收**：`go test ./model/... ./explain/... -count=1`

### 文档与清单同步

- [x] README：XGBoost 3.x JSON/UBJ 加载指南
- [x] README：训练能力边界（T1–T4 ✅ / T5 未开始）
- [x] 演进计划 §6–§9 勾选状态与代码一致（v4.0；B4/contrib/P0 ✅）

---

## P1 — 产品化与 Born（Phase 1 / B4）✅

### Born WebGPU 推理

- [x] `BornConfig.UseGPU` → `born/backend/webgpu`（Windows DX12）
- [x] `born_gpu_walk.go`：WebGPU float32 张量遍历（i32 元数据 cast）
- [x] `BornWebGPUAvailable()` + `BornUsingGPU()` + GPU 不可用时回退 CPU
- [x] `BornSupports(BackendBornGPU)` 检查 WebGPU 可用性

### Born parity 全矩阵门禁

- [x] `TestBornParityMatrix`：batch 1/16/256 × BornCPU/BornGPU
- [x] `TestBornWebGPUParitySmoke`：GPU vs Native
- [x] `TestBornParityFormatMatrix`：LGB/XGB/SK 全格式 × batch × Born
- [x] `backend_bench_test.go` 报告写入 README

### 根包渐进委托

- [x] `ensemble_delegate.go`：`PredictDense` 经 `model.Ensemble` 代理（`TestEnsembleDelegatePredictDense`）
- [x] 废弃路径标注 `Deprecated`（`LG/XG/SKEnsembleFromFile`）

**验收**：

```powershell
go test ./... -count=1
go test -tags born_train ./treebuilder/... -count=1
```

---

## P2 — 格式与训练进阶（Phase 3 收尾 + T5 启动）

### metrics 补齐（对标 `src/metric/`）

- [x] `metrics/`：MAPE、RMSLE、MError、NDCG@k、MAP（`ranking.go`）
- [x] `metrics/registry.go`：`Resolve` + XGBoost 名称对齐表
- [x] `train/metric.go`：`EvalMetric` 接线 + `GroupedMatrix` groups
- [x] 测试：公式级趋势验收（`registry_test.go`；容忍度见注释）

### 训练 T5 预备

- [x] `objective/ranking.go`：rank 目标类型（训练未接入）
- [x] `objective/registry.go`：`Register` 插件预备
- [x] `data/external.go`：外存 `ExternalMemoryMatrix` 接口草案
- [x] `data/ranking.go`：`GroupedMatrix` / `DenseWithGroups`
- [ ] `objective/ranking.go`：`rank:pairwise` 训练梯度（T5）
- [ ] `treebuilder/constraints.go`：单调约束（`monotone_constraints`）

### IO 元数据补全

- [x] `io/xgb_export.go`：`feature_names` / `feature_types` 写入
- [x] `io/xgb_json.go`：解析 `feature_names`/`feature_types`；`loss_changes`/`sum_hessian`（已有 `xgb_stats_test`）

---

## P3 — 部署与观测（Phase 5）

### WASM

- [ ] `GOOS=js GOARCH=wasm` 构建验证（依赖 B4 WebGPU 或纯 CPU fallback）
- [ ] `examples/wasm/`：HTML + `LoadFromFile` + 批预测 demo
- [ ] 部署指南：模型体积、冷启动、batch 建议

### 量化（可选）

- [ ] `quantize/`：阈值 int8 + leaf float 保留
- [ ] parity：量化模型 vs Native `diff` 门禁

### 观测钩子

- [ ] `tree/profile.go`：`PredictDense` 耗时 / 树遍历层数统计（`testing` 与生产可选）
- [ ] `train/callback.go`：`LearningRateScheduler` 回调
- [ ] 可选：`model.Ensemble.Reload(path)` 热更新辅助

### ONNX（非主路径，可选）

- [ ] `io/onnx.go` 调研：仅「已有 ONNX 模型」导入

---

## 已完成（归档，勿回退）

### Born 架构（B0–B3）

- [x] 移除 GoMLX / PJRT；`go.mod` → Born `replace`
- [x] `BackendBornCPU` / `BackendBornGPU` / `BackendAuto`
- [x] 删除 `tree/born/` 子包与 `ForestData` 快照；`born_*.go` 直读 `ForestIR`
- [x] `walkTreeBatch` + `bornForestMarginsDense`；bool Where 规避；`maskNodeIndices`
- [x] `hist_accel_born.go`（`born_train` tag）

### 推理 Phase 0.5–3（代码已落地）

- [x] XGBoost 3.x JSON/UBJSON 加载
- [x] `io.LoadFromFile` + `model.Ensemble`
- [x] `linear/NativeLinearEngine` + gblinear 桥接修复
- [x] `predict.Request`：Value / Margin / Leaf
- [x] XGB categorical bitset + 多输出向量叶
- [x] `explain/`：SHAP / importance / dump
- [x] `metrics/` 基础集

### 训练 T1–T4

- [x] hist + gbtree + 6 objectives + gblinear + dart
- [x] CV / 早停 / checkpoint / categorical 训练
- [x] `ExportXGBoostJSON` + `leaves.json` round-trip
- [x] 多线程 hist + `born_train` 增益扫描

---

## 明确不做（[-]）

- [-] Spark / Dask / Ray / Federated / Rabit 分布式训练
- [-] CGO 绑 `libxgboost` / 复刻 `c_api.h`
- [-] 官方 HTTP/gRPC serving 框架（可提供 embed 示例，不做产品）
- [-] 内置 Optuna/网格搜索（文档指向外部工具即可）
- [-] CUDA 直连推理（Born WebGPU 为 Windows GPU 路线）

---

## 迭代建议顺序

```
1. P0 contrib API 接线          → v0.9.5 语义闭环
2. P1 Born B4 + parity 矩阵     → v1.0.0 产品化
3. P0 文档 + README             → v1.0.0 发布
4. P2 metrics + T5 rank         → v1.2 / v3.0 训练进阶
5. P3 WASM + profiling          → v2.1 部署
```

---

## 快速验收命令

```powershell
# 全量回归
go test ./... -count=1

# 训练 + Born 加速
go test -tags born_train ./treebuilder/... -count=1
go test ./train/... -count=1

# parity / bench
go test -run Parity -count=1
go test -bench=. -benchmem ./... -run=^$
```
