# 版本与兼容策略（v2.x）

> 目标：用户能判断「这次升级安不安全」，而不依赖口头背景。  
> 模块：`github.com/linkerlin/leaves` · 当前线：**v2.x**（semver）

## 承诺分层

| 层 | 含义 | 例子 |
|----|------|------|
| **推荐 API** | 文档与示例主推；优先修 bug、加能力 | `LoadFromFile`、`train.NewLearner`、`cmd/leaves` |
| **兼容 API** | 保留至少整个 v2.x；新能力不首发于此 | `LGEnsembleFromFile`、`XGEnsembleFromFile`、根包旧 Ensemble 方法 |
| **实验 API** | 可改语义/收窄；README 标明 experimental | scikit-learn pickle 加载 |
| **占位 API** | 调用失败并 hint；不保证实现时间表 | （历史）完整 ONNX Graph |

详见 [api-surface.md](api-surface.md)。

## v2.x 内**允许**（非破坏或弱破坏）

在 **MINOR/PATCH** 中可以：

1. **新增** 导出函数、CLI flag、metrics 字段（只增不删；未知字段 Agent 应忽略）。
2. **收紧错误信息**（更可操作），退出码语义保持：0 成功 / 1 用法·IO / 2 内部。
3. **BackendAuto 决策优化**（须更新 `docs/backend-auto.md` + 测试；显式 `BackendNative` 等不受影响）。
4. **性能改进**（Native golden 数值 parity 门禁内）。
5. **实验/占位** 路径的行为调整（须在 Release Notes 与 interop-matrix 写明）。
6. **schema_version / leaves_cli** 递增（旧客户端忽略新字段）。

## v2.x 内**不允许**（须等 MAJOR，如 v3）

1. 删除或重命名 **推荐/兼容** 导出符号而无弃用期。
2. 改变 **默认** 推理数值语义且无法用选项恢复（例如默认 AutoTransform 再改回 raw 而不提供 flag）。
3. 破坏 **稳定** 格式的加载往返（leaves.json / 承诺的 XGB·LGB 子集）导致无法加载旧模型。
4. 将 **稳定** 格式降为实验，或移除 CLI 子命令 `sniff|train|eval|predict|inspect|explain|publish`。
5. 改变 metrics.json **已有字段含义**（可增字段；改义须升 major 或新 schema 并行）。

## 弃用流程（兼容 API）

1. README / api-surface 标注 **Deprecated** + 推荐替代。  
2. 至少一个 **MINOR** 周期保留实现。  
3. 下一 **MAJOR** 可移除。

## Agent / CLI 契约

- Agentic 闭环契约以 [演进方案.md](../演进方案.md) DoD 为准。  
- CLI metrics `schema_version`、manifest `leaves_cli` 变更：只增字段 → MINOR；改义 → 按上表。

## 发布检查

打 tag 前使用 [release-checklist.md](release-checklist.md)。
