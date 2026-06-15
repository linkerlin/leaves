# leaves 演进 TODO

> **对齐文档**：[`演进计划.md`](演进计划.md) v4.1（文档与代码同步）  
> **更新**：2026-06-15  
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
go test -tags born_train ./treebuilder/... -count=1
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

# 训练 + Born 加速
go test -tags born_train ./treebuilder/... -count=1
go test ./train/... -short -count=1

# parity / 量化 / bench
go test -run Parity -count=1
go test ./quantize/... -count=1
go test -run TestBenchGateBornCPUSlowerBatch1 -count=1
```
