# leaves 扩展点指南

> 面向：想新增 **objective / metric** 的库内或库外开发者。  
> 对齐：[`演进计划.md`](../演进计划.md) Phase B（扩展点收口）。  
> 原则：**注册式优先**；Native golden 与训练主路径不因扩展而改语义。

---

## 1. 总览：什么能注册、什么不能

| 扩展面 | 机制 | 状态 | 入口 |
|--------|------|------|------|
| **Objective** | `objective.Register` | ✅ 全量注册（含 multi / rank） | [`objective/registry.go`](../objective/registry.go)、[`objective/builtins.go`](../objective/builtins.go) |
| **Metric** | `metrics.Register` | ✅ 全量注册（含 mlogloss / ndcg@K） | [`metrics/register.go`](../metrics/register.go)、[`metrics/builtins.go`](../metrics/builtins.go) |
| **Booster** | 无 registry | 🔒 边界固定 | `train` 内 `gbtree` / `dart` / `gblinear` 三选一 |
| **Tree method** | 无 registry | 🔒 边界固定 | `hist` / `exact` / `auto`（+ 训练 hist 加速） |
| **IO 格式** | `io` 探测链 | 实验性扩展需改 `io/` | 稳定：leaves.json / XGB JSON；实验：SK / ONNX TreeEnsemble 子集 |

**设计说明（booster / tree method 为何不做 registry）**

- Booster 与 treebuilder 深度耦合 `train.Config`、梯度形状、DART drop、gblinear 线性路径。
- 外部「再挂一个 booster」几乎必然要改训练循环与 IR；半吊子 registry 会误导扩展作者。
- **明确边界**：扩展作者优先做 **objective + metric**；新 booster/tree method 视为内核变更，需走演进计划评审。

---

## 2. 新增 Objective

### 2.1 实现 `objective.Func`

```go
type Func interface {
    Name() string
    GradHess(pred, label, weight float64) (grad, hess float64)
    InitialPred(labels []float64, weights []float64) float64
}
```

多分类实现 `Multiclass` 同类语义（逐类 margin）；排序实现见 `RankFunc` / `ConfigureRanking`。

### 2.2 注册

```go
import "github.com/linkerlin/leaves/v2/objective"

func init() {
    objective.Register("custom:huber", func(numClass int) (objective.Func, error) {
        // numClass 仅 multi:* 需要；其它目标可忽略
        return Huber{Delta: 1.0}, nil
    })
}
```

解析：

```go
obj, err := objective.ByNameWithClass("custom:huber", 0)
// 训练：train.Config{Objective: "custom:huber"}
```

### 2.3 注意

- **不要**再给 `ByNameWithClass` 加 `switch` 分支；一律 `Register`。
- 排序目标：默认 `RankTrainConfig` 由工厂给出；`train` 用 `ConfigureRanking` 注入 `ndcg_k` 等。
- CLI：`leaves train --objective custom:huber` 在进程内 `init` 注册后即可用（库外程序需 import 你的注册包）。

---

## 3. 新增 Metric

### 3.1 实现 `metrics.Metric`

```go
type Metric interface {
    Name() string
    HigherIsBetter() bool
    Evaluate(yTrue, yPred []float64) (float64, error)
    EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error)
}
```

### 3.2 注册

```go
import "github.com/linkerlin/leaves/v2/metrics"

func init() {
    metrics.Register("custom_score", func(o metrics.Options) (metrics.Metric, error) {
        // o.NumClass / o.Groups / o.NDCGK 按需使用
        return MyScore{}, nil
    })
}
```

解析：

```go
m, err := metrics.Resolve("custom_score", metrics.Options{})
// 训练：Config.EvalMetric = "custom_score"
// CLI：--eval-metric custom_score
```

### 3.3 命名约定

- `NormalizeName`：小写、`-`→`_`；objective 名可别名到 metric（如 `reg:squarederror`→`rmse`）。
- `ndcg@10` / `map@5`：Resolve 拆出 K 后查 `ndcg` / `map` 工厂（**无需**为每个 K 单独注册）。

---

## 4. 验收清单（扩展 PR）

- [ ] 仅通过 `Register` 接入，无新 `switch` 分支
- [ ] 单测：`ByNameWithClass` / `Resolve` 能拿到实例；错误路径（如 multi 无 num_class）明确
- [ ] 若进 CLI：在 `skills/leaves-autotrain/cli.md` 补充名称（可选）
- [ ] 不破坏 `go test ./objective ./metrics ./train -count=1`

---

## 5. 可运行示例

[`examples/extension/`](../examples/extension/)：注册 `custom:l1` + `max_abs_error`，`go run` / `go test` 闭环。

```powershell
go run ./examples/extension/
go test ./examples/extension/ -count=1
```

### 多目标回归（LIB-21）

```go
dm, _ := data.NewMultiTargetDense(X, n, p, Yflat /* n*k */, k, nil)
learner, _ := train.NewLearner(train.Config{
    Objective: "reg:squarederror",
    NumTarget: k, // one_output_per_tree
    NumRound:  40,
})
_ = learner.Fit(dm)
```

CLI：`leaves train --data mt.csv --objective reg:squarederror --num-target 2`（CSV 列序：`f...,y0,y1`）。  
**向量叶** `multi_output_tree` 训练不做；XGB 向量叶 **推理**仍支持。

## 6. 相关路径速查

| 任务 | 路径 |
|------|------|
| 训练接 objective | [`train/learner.go`](../train/learner.go) → `ByNameWithClass` |
| 训练接 metric | [`train/metric.go`](../train/metric.go) → `metrics.Resolve` |
| CLI train/eval | [`cmd/leaves/train.go`](../cmd/leaves/train.go)、[`eval.go`](../cmd/leaves/eval.go) |
| 内置列表 | `objective.RegisteredNames()` / `metrics.RegisteredNames()` |
| 可跑 demo | [`examples/extension/`](../examples/extension/) |

---

## 7. 与 Agentic / 库路线的关系

- **Agent 闭环**不依赖自定义扩展；扩展是库能力。
- 自定义 objective 的 Agent 使用方式：用户进程 `import _ "yours/plugin"` 后仍用 `leaves` CLI，或直接调 `train` API。
- 库 12 个月路线：[`演进计划.md`](../演进计划.md)；Agent 契约：[`演进方案.md`](../演进方案.md)。
