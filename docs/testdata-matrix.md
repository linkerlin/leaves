# testdata 回归矩阵

> 格式 × 后端 × batch 门禁一览（2026-06-15，**v4.3** 嗅探/AutoTransform）。  
> 运行：`go test ./... -count=1`；Born 矩阵见 `born_parity_format_test.go`。

## 推理 parity（加载 → 预测）

| 格式 | 模型文件 | 数据 | 测试 | 后端 | Batch |
|------|----------|------|------|------|-------|
| LGB 文本 | `lg_breast_cancer.txt` | `lg_breast_cancer_data.txt` | `TestBornParityFormatMatrix` | Native / BornCPU / BornGPU* | 1, 16, 256 |
| LGB JSON | `lg_dart_breast_cancer.json` | `breast_cancer_test.tsv` | 同上 | 同上 | 同上 |
| LGB model | `lg_dart_breast_cancer.model` | `breast_cancer_test.tsv` | 同上 | 同上 | 同上 |
| XGB 二进制 | `xgagaricus.model` | 内嵌 | 同上 | 同上 | 同上 |
| XGB JSON | `xgboost_smoke.json` | 内嵌 | 同上 + `io/io_test.go` | 同上 | 同上 |
| XGB UBJSON | `xgboost_smoke.ubj` | 内嵌 | 同上 | 同上 | 同上 |
| XGB RF JSON | `xgboost_rf_smoke.json` | 内嵌 | 同上 | 同上 | 同上 |
| XGB categorical | `xgboost_categorical_smoke.json` | 内嵌 | 同上 + export round-trip | 同上 | 同上 |
| XGB multi-target | `xgboost_multitarget_vector.json` | 内嵌 | `io/xgb_multitarget_vector_test.go` | 同上 | 同上 |
| XGB gblinear JSON | `xgboost_gblinear_smoke.json` | 内嵌 | `io/xgb_interop_test.go` | 同上 | 同上 |
| XGB gamma JSON | `xgboost_gamma_smoke.json` | 内嵌 | `io/xgb_interop_test.go` | 同上 | 同上 |
| XGB poisson JSON | `xgboost_poisson_smoke.json` | 内嵌 | `io/xgb_interop_test.go` | 同上 | 同上 |
| XGB dart JSON | `xgboost_dart_smoke.json` | 内嵌 | `io/xgb_interop_test.go`（嵌套 gbtree） | 同上 | 同上 |
| XGB multiclass JSON | `xgboost_multiclass_smoke.json` | 内嵌 | `io/xgb_interop_test.go`（softprob） | 同上 | 同上 |
| XGB 二进制 | `xgagaricus.model` | 内嵌 | `io/xgb_bin.go` + `io/xgb_interop_test.go` | 同上 | 同上 |
| SK pickle | `sk_gradient_boosting_classifier.model` | `sk_gradient_boosting_classifier_test.libsvm` | 同上 | 同上 | 同上 |

\* BornGPU 仅在 Windows WebGPU 可用时运行；不可用时跳过。

## 专项 golden

| 场景 | 数据 / 模型 | 测试 |
|------|-------------|------|
| SHAP contrib | `shap_contribs_*.tsv` | `model/predict_contrib_*_test.go` |
| 多类 SHAP | `shap_contribs_multiclass_*` | `predict_contrib_p0_test.go` |
| rank pairwise λ | `rank_pairwise_grad_golden.json` | `objective/ranking_grad_golden_test.go` |
| rank ndcg λ | `rank_ndcg_grad_golden.json` | `objective/ranking_ndcg_golden_test.go` |
| rank vs XGB | `rank_*_xgb_baseline.json` | `train/rank_*_test.go` |
| 量化 parity | 任意 smoke 模型 | `quantize/gate_test.go` |
| 外存 hist bins | 合成 Dense 切批 | `treebuilder/hist_bins_external_test.go` |
| tweedie 训练 | 合成 Dense | `train/tweedie_survival_test.go` `TestTweedieTrainSmoke` |
| survival:cox/aft | 合成 Dense / AFTInterval | `TestSurvivalCoxTrainSmoke`, `TestSurvivalAFTIntervalTrainSmoke` |
| checkpoint 续训 | 临时 ckpt | `train/resume_test.go` |
| Eval API | 合成 Dense | `train/eval_test.go` `TestLearnerEval` |
| FromFile 嗅探 | libsvm / rank TSV / breast_cancer TSV | `data/fromfile_test.go` |
| AutoTransform 默认 | — | `io/transform_auto_test.go` |
| InferObjective + 便利训练 | `xgboost_smoke.json` + `breast_cancer_train.tsv` | `train/load_test.go` |
| train_from_model demo | 同上 | `examples/train_from_model/`（手工 `go run`） |
| FromFile 显式格式 | CSV/libsvm | `data/csv_test.go`, `data/libsvm_test.go` |
| WASM 体积 | `examples/wasm` 构建产物 | `wasm_size_gate_test.go` |

## 训练趋势（非 bit-exact）

| 数据集 | 目标 | baseline JSON | 测试 |
|--------|------|---------------|------|
| rank smoke | rank:ndcg / pairwise | `rank_smoke_*_xgb_baseline.json` | `TestRank*TrendVsXGBoost` |
| MSLTR 子集 | rank:ndcg / pairwise | `rank_msltr_*_xgb_baseline.json` | `TestRankMSLTR*` |
| MovieLens | rank:ndcg / pairwise | `rank_movielens_*_xgb_baseline.json` | `TestRankMovieLens*` |
| breast cancer | reg:squarederror | `breast_cancer_xgb_baseline.json` | `train/t1_test.go` |

生成脚本见各 `testdata/gen_*.py`；缺失文件时相关测试 `t.Skip`。

## CI 映射

| Job | 命令 | 覆盖 |
|-----|------|------|
| test (3 OS) | `go test ./...` | 全矩阵 + 训练 |
| wasm | `go build` + `TestWasmBinarySizeGate` | js/wasm 编译 + 体积 ≤16 MiB |
| bench-gate (Windows) | `TestBenchGateBornCPUSlowerBatch1` | batch=1 BornCPU ≥20× Native |
