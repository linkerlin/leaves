# 互操作支持等级表

> 演进计划 **Phase D**：先写清边界，再扩格式。  
> 代码真源：[`io/support.go`](../io/support.go)（`SupportOf` / `SupportTable`）  
> 回归映射：[`testdata-matrix.md`](testdata-matrix.md)

## 等级定义

| 等级 | 英文 | 含义 |
|------|------|------|
| **稳定** | `stable` | CI + testdata 矩阵覆盖；API 变更走兼容流程 |
| **实验** | `experimental` | 可用但协议/版本面窄；失败时文档与 hint 写明限制 |
| **占位** | `placeholder` | API 存在但加载必失败，并给出转换路径 |
| **不支持** | `unsupported` | 无法识别或不在路线图 |

## 格式总表

| 格式 | 等级 | 扩展名 / 探测 | 能力摘要 | 失败时建议 | 回归锚点 |
|------|------|---------------|----------|------------|----------|
| **leaves.json** | 稳定 | `leaves_version` JSON | 训练主产物；推荐部署 | 检查 version 与 booster | `train/load_test.go` |
| **XGBoost JSON** | 稳定 | `.json` + `learner` | 3.x 默认；gbtree/dart/gblinear/多类等 | `save_model('m.json')` | `xgboost_*.json` + parity |
| **XGBoost UBJSON** | 稳定 | `.ubj` | 与 JSON 预测一致 | 用 `.ubj` 导出 | `xgboost_smoke.ubj` |
| **XGBoost binary** | 稳定 | `binf` / header 嗅探 | 经典 Booster 二进制 | 优先改 JSON/UBJ | `xgagaricus.model` |
| **LightGBM text** | 稳定 | `tree=` / `version=` | text model | 勿把数值 TSV 当模型 | `lg_breast_cancer.txt` |
| **LightGBM JSON** | 稳定 | `tree_info` | LGB JSON | 与 text 同源导出 | `lg_dart_*.json` |
| **scikit-learn** | **实验** | `.pkl` / `.joblib` / pickle 魔数 | 窄协议 GB 历史 pickle | **优先转 XGB JSON / leaves.json** | `sk_*.model`（实验） |
| **ONNX** | **实验**（子集）/**可用**（Graph） | `.onnx` | `LoadONNX`：TreeEnsemble **Regressor + Classifier** 子集；`LoadOnnxGraph`：完整 Graph（born 运行时，30+ 算子，非 wasm） | CATEGORY 分裂 / 其余 BRANCH 模式 / 多标签仍不支持 | `io/onnx_test.go`, `io/onnx_classifier_test.go`, `io/onnx_graph_native_test.go` |

## ONNX 策略

两条路径：

1. **TreeEnsemble 子集**（`LoadONNX`，wasm 可用）：`ai.onnx.ml` **TreeEnsembleRegressor**（`BRANCH_LEQ` + `LEAF`，`aggregate=SUM`，`post_transform=NONE`）与 **TreeEnsembleClassifier**（`BRANCH_LEQ` + `LEAF`；`aggregate=SUM/AVERAGE`；`post_transform=NONE/SOFTMAX`，或二类 `LOGISTIC`）。转 ForestIR 走 Native/Born 树引擎。
2. **完整 Graph**（`LoadOnnxGraph`，born 运行时，**非 wasm**）：任意算子（30+，opset 1–21），通用 NN/图推理；返回 `OnnxModel.Predict`。复用 `github.com/born-ml/born/onnx` 运行时，不在 leaves 内重实现算子。

仍不支持（子集内）：CATEGORY 分裂、其余 BRANCH 模式、Classifier 多标签、向量叶多目标单树。失败时 `*LoadError`（子集，level=`experimental`）或 `LoadOnnxGraph` 错误带可操作 hint。

## scikit-learn 策略（LIB-11 收窄）

1. 保持 **实验性**；**不**扩全版本 / 全 estimator 矩阵。  
2. 生产推荐：训练后导出 **XGB JSON** 或 **leaves.json**。  
3. 加载失败时 `LoadError.Hint` 标明实验边界。

### 协议矩阵（写实）

| 能力 | 状态 | 说明 / 回归锚点 |
|------|------|-----------------|
| `GradientBoostingClassifier` 历史 pickle | 实验可用 | `testdata/sk_gradient_boosting_classifier.model` + `TestSKGradientBoostingClassifier` |
| `GradientBoostingClassifier` / 多类 iris | 实验可用 | `testdata/sk_iris.model` + `TestSKIris`（允许少量 float32 边界 mismatch） |
| 探测 `.pkl` / `.joblib` / pickle 魔数 | 稳定探测 | `DetectFormat` → `FormatSklearn` |
| 任意 sklearn 版本 round-trip | **不做** | 新版 joblib/cloudpickle 协议可能失败 |
| HistGradientBoosting / RandomForest / Pipeline | **不做** | 无 loader |
| 类别特征 / 缺失处理对齐 SK | **不做** | 数值树子集 |
| 生产部署直接依赖 SK pickle | **不推荐** | 优先转 JSON |

失败用例：`io` 对损坏 pickle / 非 pickle 伪装文件返回可操作 `LoadError`（见 `TestSklearnLoadFailureActionable`）。

## 错误契约

失败时优先返回 `*io.LoadError`：

```text
io: load path [FormatName/stable|experimental|...]: message
hint: 下一步操作
```

| 字段 | 用途 |
|------|------|
| `Format` / `Level` | 支持等级 |
| `Op` | `detect` / `load` |
| `Hint` | Agent/人类可执行下一步 |
| `Unwrap` | 原始 cause |

```go
m, err := io.LoadFromFile(path, io.DefaultLoadOptions())
var le *io.LoadError
if errors.As(err, &le) {
    _ = le.Level // SupportExperimental 等
    _ = le.Hint
}
```

## 与 BackendAuto / 回归矩阵

- 格式等级描述 **能否加载**；加载后的推理后端见 [backend-auto.md](backend-auto.md)。  
- 仅 **稳定** 与已列 **实验** 行进入默认 `go test ./...` 承诺面。  
- 占位格式不进入 parity 矩阵。
