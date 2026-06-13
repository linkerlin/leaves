# leaves 演进 TODO



> **对齐文档**：[`演进计划.md`](演进计划.md) v4.1（文档与代码同步）  

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

- [x] README：训练能力边界（T1–T4 ✅；T5 rank+单调 ✅；survival/外存 未开始）

- [x] 演进计划 v4.1：§5–§14、附录 A/B/F 与代码一致



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



## P2 — 格式与训练进阶 ✅（核心项已完成）



### metrics 补齐（对标 `src/metric/`）



- [x] `metrics/`：MAPE、RMSLE、MError、NDCG@k、MAP（`ranking.go`）

- [x] `metrics/registry.go`：`Resolve` + XGBoost 名称对齐表

- [x] `train/metric.go`：`EvalMetric` 接线 + `GroupedMatrix` groups

- [x] 测试：公式级趋势验收（`registry_test.go`；容忍度见注释）



### 训练 T5 — 排序与约束（已完成）



- [x] `objective/ranking.go` + `train/rank_fit.go`：rank 训练环

- [x] `objective/registry.go`：`Register` 插件预备

- [x] `data/ranking.go`：`GroupedMatrix` / `DenseWithGroups`

- [x] `objective/ranking_grad.go`：`rank:pairwise` + 逐对 λ golden

- [x] `objective/ranking_pair.go`：`lambdarank_pair_method`（full/topk/mean）

- [x] `objective/ranking_ndcg_golden_test.go` + `gen_rank_ndcg_grad.py`

- [x] `train/rank_monotone_test.go`：排序 × 单调约束

- [x] `treebuilder/constraints.go`：`monotone_constraints`

- [x] `train/callback.go`：LR 调度 + `EvalMetric` 回调



### IO 元数据补全



- [x] `io/xgb_export.go`：`feature_names` / `feature_types` 写入

- [x] `io/xgb_json.go`：解析 `feature_names`/`feature_types`；`loss_changes`/`sum_hessian`



### P2 工程债（文档 / 结构，非阻塞）



- [ ] 全 testdata 回归矩阵文档化（格式×后端×batch 表）

- [ ] Benchmark 套件 CI 基线门禁

- [ ] 根包 IO 迁入 `io/`（`lgensemble_io.go` 等）

- [~] 根包全 API 委托 `model.Ensemble`（`ensemble_delegate.go` 部分完成）

- [ ] 文档：后端选择专章、Born 限制、超参 XGBoost 对照表

- [ ] `objective`/`metrics` Registry 插件化（减少大 switch）



---



## P3 — 部署与观测（Phase 5）~60%



### WASM（下一步主线）



- [ ] `GOOS=js GOARCH=wasm` 构建验证（Native CPU fallback）

- [ ] `examples/wasm/`：HTML + `LoadFromFile` + 批预测 demo

- [ ] 部署指南：模型体积、冷启动、batch 建议

- [ ] 部署性能报告（wasm vs Native）



### 量化 ✅



- [x] `quantize/`：阈值 int8 + leaf float 保留

- [x] parity：量化模型 vs Native `diff` 门禁



### 观测钩子 ✅



- [x] `tree/profile.go`：`PredictDense` 耗时 / 树遍历统计

- [x] `train/callback.go`：`LearningRateScheduler` + `TrainingCallback`（含 `EvalMetric`）

- [x] `model.Ensemble.Reload(path)` 热更新（需 `import leaves` 注册）



### ONNX（非主路径，可选）



- [ ] `io/onnx.go` 调研：仅「已有 ONNX 模型」导入



### P3 可选深化



- [ ] `predict.Request` 级耗时钩子（包装 `tree/profile.go`）

- [ ] HTTP embed 中间件示例（非官方 serving）



---



## T5 余下 — 训练完备 ~45% 待做



> rank + 单调已完成；下列为 v3.0 剩余交付。



- [ ] `ExternalMemoryMatrix` → `train.Learner` / `treebuilder` 接线

- [ ] `survival:cox` / `survival:aft` 目标函数

- [ ] `reg:tweedie` 训练目标（推理已可加载 LGB tweedie 模型）

- [ ] Checkpoint **续训**（`LoadCheckpoint` → 继续 `Fit`）

- [ ] `Learner.Eval(dm)` 公开 API

- [ ] `data.FromCSV` / `FromFile`（libsvm/csv）

- [ ] `max_leaves` / lossguide 生长策略（可选）

- [ ] Multi-output tree **训练**（推理 `OutputDim` 已有）

- [-] `train.HyperparamSearch`（文档指向外部 Optuna，不做内置）



**验收（T5 排序已完成）**：



```powershell

cd testdata && python gen_rank_pairwise_grad.py && python gen_rank_ndcg_grad.py

go test ./objective/... ./train/... -short -count=1

go test ./train/... -run 'Rank|Monotone|Callback' -count=1

```



---



### 收尾测试（2026-06-13）

- [x] `model/reload_internal_test.go`：Reload 未注册 / ReplaceEngine / DetachEngine
- [x] `quantize/gate_test.go`：Parity 门禁 Pass/Fail
- [x] `data/external_test.go`：ExternalMemoryMatrix 接口 stub
- [x] `train/callback_test.go`：回调错误中断训练
- [x] `model/predict_contrib_p0_test.go`：补 `import leaves` 注册

---



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

- [x] `predict.Request`：Value / Margin / Leaf / Contribution

- [x] XGB categorical bitset + 多输出向量叶

- [x] `explain/`：SHAP / importance / dump

- [x] `metrics/` 扩展集（含 ranking metrics）



### 训练 T1–T4 + T5 排序



- [x] hist + gbtree + 6 objectives + gblinear + dart

- [x] CV / 早停 / checkpoint / categorical 训练

- [x] `ExportXGBoostJSON` + `leaves.json` round-trip

- [x] 多线程 hist + GPU hist + Born 增益扫描

- [x] rank:pairwise/ndcg/listwise + golden + XGB 对齐



---



## 明确不做（[-]）



- [-] Spark / Dask / Ray / Federated / Rabit 分布式训练

- [-] CGO 绑 `libxgboost` / 复刻 `c_api.h`

- [-] 官方 HTTP/gRPC serving 框架（可提供 embed 示例，不做产品）

- [-] 内置 Optuna/网格搜索（文档指向外部工具即可）

- [-] CUDA 直连推理（Born WebGPU 为 Windows GPU 路线）

- [-] inplace_predict / staged cache（P2 可选，优先级低）



---



## 迭代建议顺序（2026-06-13 更新）



```

1. ✅ P0 contrib + 文档同步

2. ✅ P1 Born B4 + parity 矩阵

3. ✅ P2 metrics + T5 rank/单调

4. ✅ P3 quantize + profile + Reload

5. → P3 WASM demo + 部署指南          （当前主线）

6. → T5 外存 DMatrix 接线

7. → T5 survival / tweedie + 续训 API

8. → P2 工程债：testdata 矩阵文档 + benchmark CI

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

go test -bench=. -benchmem ./... -run=^$

```


