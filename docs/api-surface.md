# API 表面：推荐 · 兼容 · 实验

> 用户应能一眼区分三条路径。实现细节见各包；互操作等级见 [interop-matrix.md](interop-matrix.md)。

## 1. 推荐路径（新代码请只用这些）

### 1.1 推理

```go
import (
    "github.com/linkerlin/leaves/v2"
    "github.com/linkerlin/leaves/v2/io"
)

m, err := leaves.LoadFromFile("model.json", leaves.DefaultLoadOptions())
// 或 io.LoadFromFile（LGB/XGB/leaves.json/ONNX/sklearn pickle 均无需根包 init）
// 默认 AutoTransform=true，Backend=Auto
p := m.PredictSingle(features, 0) // 返回 float64；多类/特征不足时静默 0（兼容层语义）
```

| 能力 | 入口 |
|------|------|
| 加载 | `LoadFromFile` / `io.LoadFromFile` + `LoadOptions` |
| 后端 | `LoadOptions.Backend` + `Workload`；决策表 [backend-auto.md](backend-auto.md) |
| 解释 | `m.Explain()` / `explain` 包 / CLI `leaves explain` |
| 量化 | `quantize` 包；CLI `publish --quantize` |
| 热更新 | `m.Reload(...)` |

### 1.2 训练

```go
dm, err := leaves.LoadDataAuto("train.csv") // 或 data.FromFileAuto
learner, err := leaves.NewLearner(leaves.TrainConfig{Objective: "reg:squarederror", NumRound: 50})
_ = learner.Fit(dm)
_ = learner.Save("out.leaves.json")
```

| 能力 | 入口 |
|------|------|
| 数据 | `LoadDataAuto` / `data.FromFileAuto` / `data.FromFile` |
| 学习器 | `NewLearner` / `train.NewLearner` |
| 从参考模型 | `NewLearnerFromModelAndData` |
| 续训 | `ResumeFit` / checkpoint |
| 扩展目标/指标 | `objective.Register` / `metrics.Register` — [extension-points.md](extension-points.md) |

### 1.3 Agent / CLI（零 Go 业务代码）

```text
leaves sniff → train (--cv/--runs/--from-run) → eval → inspect → explain → publish
         辅助：lessons (add|search|list) · version
```

- SKILL：`skills/leaves-autotrain/`  
- Demo：`examples/autotrain/`  
- 契约：`演进方案.md`

### 1.4 稳定差异化示例

| 能力 | 示例 / 文档 |
|------|-------------|
| 训练便利 | `examples/train_from_model/` |
| WASM | `examples/wasm/`（部署建议见其 README） |
| HTTP embed | `examples/http/`（**非**官方 serving 产品） |
| 量化发布 | `leaves publish --quantize` |

---

## 2. 兼容路径（仍维护，不作为新能力首发）

| 旧入口 | 替代（推荐） | 说明 |
|--------|--------------|------|
| `LGEnsembleFromFile(path, loadTransformation)` | `LoadFromFile` + `LoadOptions` | 内部委托 `model.Ensemble` |
| `XGEnsembleFromFile` / `XGBLinearFromFile` | 同上 | 同上 |
| `SKEnsembleFromFile` | 同上；格式为**实验** | 生产请转 JSON |
| 根包 `Ensemble` 直接方法 | `model.Ensemble` / `LoadFromFile` 返回值 | 根包保留别名 |
| `LoadTransformation` 仅布尔 | `AutoTransform` + `LoadTransformation` | 见 NOTES |

迁移示例：

```go
// 旧
m, err := leaves.LGEnsembleFromFile("lg.model", true)

// 新
m, err := leaves.LoadFromFile("lg.model", leaves.DefaultLoadOptions())
// 若只要 raw margin：
m, err = leaves.LoadFromFile("lg.model", &leaves.LoadOptions{AutoTransform: false})
```

---

## 3. 实验 / 占位

| 入口 | 等级 | 说明 |
|------|------|------|
| scikit-learn `.pkl`/`.joblib` | 实验 | 窄协议；失败见 `LoadError.Hint` |
| `io.LoadONNX` / `.onnx` | 实验 | TreeEnsemble **Regressor + Classifier** 子集（SUM/AVERAGE × NONE/SOFTMAX/LOGISTIC-二类）；复杂图走 `io.LoadOnnxGraph`（born 运行时，非 wasm）或转 XGB/leaves JSON |
| 官方 HTTP/gRPC serving | **不做** | 用 [`examples/http`](../examples/http)、仓内 [`examples/serving-template`](../examples/serving-template)，或独立仓 [leaves-serving](https://github.com/linkerlin/leaves-serving) |
| 自定义 objective/metric | 稳定机制 | 须自行 `Register`；不进默认 CLI 名表除非文档 |
| `recsys/{contract,split,eval,ledger,replay,monitor,release}` | 实验（契约冻结 schema v1，字段只增不删） | 推荐生产闭环控制面；指南 [recsys-loop.md](recsys-loop.md)；官方 registry / 在线 serving / 实时学习**不做** |

---

## 4. 包边界（给扩展作者）

| 包 | 职责 |
|----|------|
| `io` | 加载、格式探测、支持等级 |
| `train` / `data` / `objective` / `metrics` | 训练与扩展点 |
| `tree` / `model` | 推理 IR、Backend、Ensemble |
| `cmd/leaves` | Agent CLI |
| `recsys`（四段 + 控制面包） | 离线推荐流水线与生产闭环契约；`release.Adapter` 只产出请求，无网络副作用 |
| 根包 `leaves` | 便利别名 + **兼容层** |

`tree/` 不依赖 `train/`（见 AGENTS.md）。

---

## 5. 与发布

- 版本允许变更： [versioning.md](versioning.md)  
- 打 tag 前： [release-checklist.md](release-checklist.md)
