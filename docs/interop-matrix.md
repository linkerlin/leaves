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
| **ONNX** | **实验** | `.onnx` | **TreeEnsembleRegressor 子集**（BRANCH_LEQ/SUM/NONE） | 完整 Graph 仍请转 JSON/leaves | `io/onnx_test.go` |

## ONNX 策略（LIB-10 子集）

1. **实验性导入**：仅 `ai.onnx.ml` **TreeEnsembleRegressor**；`BRANCH_LEQ` + `LEAF`；`aggregate=SUM`；`post_transform=NONE`。  
2. **不做**：任意 Graph 算子链、Classifier 后处理、类别分裂、向量叶多目标单树、外部 initializer 依赖。  
3. 失败时 `*LoadError` level=`experimental`，hint 指向转 XGB JSON / leaves.json。  
4. 完整 Graph 仍 **明确不做**（见 TODO 非目标）。

## scikit-learn 策略（冻结）

1. 保持 **实验性**；不扩全版本 / 全 estimator 矩阵。  
2. 生产推荐：训练后导出 **XGB JSON** 或 **leaves.json**。  
3. 加载失败时 `LoadError.Hint` 会标明实验边界。

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
