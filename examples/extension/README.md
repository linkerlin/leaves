# 扩展示例：自定义 objective / metric

对齐 [`docs/extension-points.md`](../../docs/extension-points.md)（库线 Phase B）。

本目录演示：**只通过 `Register` 接入**，不修改 leaves 内核 `switch`。

| 名称 | 类型 | 含义 |
|------|------|------|
| `custom:l1` | objective | L1 / MAE 风格梯度；常数 Hessian |
| `max_abs_error` | metric | 最大绝对误差（越小越好） |

## 运行

```powershell
go run ./examples/extension/
go test ./examples/extension/ -count=1
```

## 要点

1. 在 **你的程序** `init()`（或插件包）里 `objective.Register` / `metrics.Register`。
2. `train.Config{Objective: "custom:l1", EvalMetric: "max_abs_error"}` 与内置名用法相同。
3. **CLI `leaves train` 默认进程不会加载本示例的 `init`**；Agent/CLI 路径要自定义目标时，用库 API 或自建带 blank import 的包装入口。
4. booster / tree method **不做** registry（见扩展文档边界）。

## 与训练加速 / 推理 BackendAuto

- **训练加速**：`LEAVES_TRAIN_ACCEL` / `Config.AccelMode`（hist 增益扫描）。
- **推理后端**：`tree.BackendAuto`（batch / GPU / 稀疏），与 objective 无关。

二者独立；见 [`docs/backend-auto.md`](../../docs/backend-auto.md) §训练 vs 推理。
