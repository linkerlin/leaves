## 2026年6月13日 — Born 全面迁移

**决策**：废弃 GoMLX（`github.com/gomlx/gomlx`），不采用 gogpu/wgpu 直连；统一迁移至 [Born](https://github.com/born-ml/born) 作为张量计算底座。

| 变更 | 说明 |
|------|------|
| 推理加速 | `tree.GoMLXEngine` → `tree.BornEngine` |
| Backend | `BackendGoMLX`/`BackendSimpleGo` → `BackendBornGPU`/`BackendBornCPU` |
| 训练加速 build tag | `gomlx_train` → `born_train` |
| 正确性 golden | `NativeEngine` 不变 |
| 删除 | `scripts/install_pjrt.go`、`scripts/verify_pjrt.go` |

详见 `演进计划.md` §3.6 与附录 C/F；执行 backlog 见 [`TODO.md`](TODO.md)。

**v4.0**（同日）：对标 XGBoost 3.3 全链路审计——训练 T1–T4 已落地；推理 Phase 0.5–3 代码超前于旧文档；真实 P0 缺口为 `predict.Request` 贡献值接线 + Born B4 WebGPU。

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
