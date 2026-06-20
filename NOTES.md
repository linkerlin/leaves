## 2026年6月15日 — v4.3 格式嗅探与 AutoTransform 默认

**状态**：训练数据内容嗅探、模型 `AutoTransform` 默认开启、训练便利 API 已落地；路线图已切到 [`演进计划.md`](演进计划.md) v5.0。

| 交付 | 说明 |
|------|------|
| 训练数据嗅探 | `data/sniff.go`：`FromFileAuto` / `LoadDataAuto`（libsvm、排序 TSV、末列 label TSV、CSV） |
| 训练便利 API | `train/load.go`：`InferObjectiveFromModel`、`NewLearnerFromModelAndData`；根包 `train_api.go` 别名 |
| 模型加载 | `io/transform_auto.go`：`ResolveLoadTransformation`；`DefaultLoadOptions` 默认 `AutoTransform: true` |
| 格式探测 | `io/load.go`：XGB 二进制 header 探测、`.pkl`/`.joblib`、误用数值 `.txt` 报错 |
| Demo | `examples/train_from_model/` |

**行为变更（推理）**：`leaves.DefaultLoadOptions()` / `io.DefaultLoadOptions()` 现默认 `AutoTransform: true`。对 `binary:logistic`、`multi:softprob`、`reg:gamma` 等，`PredictSingle` / `OutputValue` 返回变换后值（如概率）；`reg:squarederror` 等无需变换的目标仍为 raw margin。

若需与旧版一致的 **默认 raw margin**：

```go
m, err := leaves.LoadFromFile(path, &leaves.LoadOptions{AutoTransform: false})
// 或 predict.Request{Output: predict.OutputMargin}
```

遗留 `LGEnsembleFromFile(path, loadTransformation)` 第二参数语义不变；新路径请用 `LoadOptions.AutoTransform` / `LoadTransformation`。

测试：`data/fromfile_test.go`、`io/transform_auto_test.go`、`train/load_test.go`。

---

## 2026年6月15日 — v4.2 文档与代码同步

**状态**：P0–T5 + v3.1 可选深化均已完成（见 [`TODO.md`](TODO.md)）。

| 交付 | 说明 |
|------|------|
| T5 完备 | survival/tweedie、外存 DMatrix、续训、Eval、FromFile、max_leaves |
| 部署 | WASM demo + CI 体积门禁；HTTP embed 批预测；quantize |
| 根包 API | `io.LoadFromFile`、`train_api.go`（`NewLearner`/`ResumeFit`） |

训练加速：已移除 `born_train` tag；使用 `LEAVES_TRAIN_ACCEL` + `treebuilder/hist_accel*.go`。

---

## 2026年6月13日 — Born 全面迁移

**决策**：废弃 GoMLX（`github.com/gomlx/gomlx`），不采用 gogpu/wgpu 直连；统一迁移至 [Born](https://github.com/born-ml/born) 作为张量计算底座。

| 变更 | 说明 |
|------|------|
| 推理加速 | `tree.GoMLXEngine` → `tree.BornEngine` |
| Backend | `BackendGoMLX`/`BackendSimpleGo` → `BackendBornGPU`/`BackendBornCPU` |
| 训练加速 build tag | ~~`gomlx_train`~~ → ~~`born_train`~~ **已移除**；改用 `LEAVES_TRAIN_ACCEL` |
| 正确性 golden | `NativeEngine` 不变 |
| 删除 | `scripts/install_pjrt.go`、`scripts/verify_pjrt.go` |

详见 [`演进计划.md`](演进计划.md) 的当前状态摘要与年度路线图；执行 backlog 见 [`TODO.md`](TODO.md)。

**v4.0**（同日）：对标 XGBoost 3.3 全链路审计——训练 T1–T4 已落地；推理 Phase 0.5–3 代码超前于旧文档。  
**v4.2**（2026-06-15）：P0（contrib + Born B4）与 T5/v3.1 全部闭合；本文档与 `演进计划.md` 已同步实态。  
**v4.3**（2026-06-15）：格式嗅探、`AutoTransform` 默认、训练便利 API；见上文 v4.3 节。

---

## 2019年3月16日

变换函数于此日引入。此前 _leaves_ 仅能输出原始预测值。今于所有模型加载函数（`XGEnsembleFromReader`、`XGEnsembleFromFile`、`XGBLinearFromReader`、`XGBLinearFromFile`、`SKEnsembleFromReader`、`SKEnsembleFromFile`、`LGEnsembleFromJSON`、`LGEnsembleFromReader`、`LGEnsembleFromFile`）中，新增一布尔型选项，名曰 `loadTransformation`。

譬如，旧时写法：

```go
model, err := leaves.LGEnsembleFromFile("lg_breast_cancer.model")
```

若欲保留旧日行为，当改为：

```go
model, err := leaves.LGEnsembleFromFile("lg_breast_cancer.model", false)
```

再者，`Ensemble` 之 `NClasses` 方法将更名为 `NRawOutputGroups`，其意不变——即模型对每个对象在原始预测中所给出之数值个数。另新增 `NOutputGroups` 方法，意谓模型对每个对象施加变换函数后所给出之数值个数。笼统言之，变换函数可改输出之维度也。须知，若当前变换函数为 `raw`：

```go
model.Transformation().Name() == "raw"
```

则有：

```go
model.NRawOutputGroups() == model.NOutputGroups()
```
