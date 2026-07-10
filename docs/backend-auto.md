# BackendAuto 2.0 决策表

> 真源实现：[`tree/backend_select.go`](../tree/backend_select.go)  
> 回归：`go test ./tree -run 'SelectBackend|BackendAuto|BenchRecord' -count=1`  
> 对齐：[`演进计划.md`](../演进计划.md) Phase C

## 设计原则

1. **Native 是正确性 golden**；Born 是可选加速。
2. **小 batch 在线推理优先 Native**（BornCPU 单条显著更慢，见 [benchmark-baseline.md](benchmark-baseline.md)）。
3. **任何自动选择必须能被规则码解释**（`SelectBackendExplained`）。
4. **显式 `Backend` 覆盖 Auto**（`ResolveBackend` 非 Auto 直接返回）。

## 阈值常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `AutoBatchCPUThreshold` | **64** | batch ≥ 此值 → 倾向 BornCPU |
| `AutoBatchGPUThreshold` | **256** | batch ≥ 此值且 HasGPU → 尝试 BornGPU |
| `AutoSparseDensityMax` | **0.15** | `0 < SparseDensity < 0.15` → Native |

## 决策表（按优先级）

| 优先级 | 条件 | 后端 | Rule 码 |
|--------|------|------|---------|
| 1 | 线性模型 / 无森林 | Native | `linear_or_empty` |
| 2 | `Target=WASM` 且 Born 支持该森林 | BornCPU | `wasm_born_cpu` |
| 3 | `Target=WASM` 且 Born 不支持 | Native | `wasm_native_fallback` |
| 4 | `0 < SparseDensity < 0.15` | Native | `sparse` |
| 5 | 非纯数值 / cat-small | Native | `non_numeric_or_unsupported` |
| 6 | batch≥256 且 HasGPU 且 WebGPU 可用 | BornGPU | `born_gpu` |
| 7 | batch≥256 且 HasGPU 但 WebGPU 不可用 | BornCPU | `born_cpu_gpu_unavailable` |
| 8 | batch≥64 且数值树 | BornCPU | `born_cpu` |
| 9 | 其余（含默认 batch=1） | Native | `small_batch` |

## WorkloadHint 字段

```go
type WorkloadHint struct {
    BatchSize     int         // 预测批大小；≤0 按 1
    HasGPU        bool        // 调用方声明可用 GPU（仍检查 BornWebGPUAvailable）
    Target        DeployTarget // Default | WASM
    SparseDensity float64     // 0=未知/稠密；(0,0.15)=稀疏
}
```

## 部署建议（写实）

| 场景 | 建议 |
|------|------|
| 在线单条 / 小批（batch≪64） | **Native**（默认 Auto 即如此） |
| 大批量离线打分（batch≥64） | Auto → **BornCPU**；或显式 `BackendBornCPU` |
| Windows + 大批量 + GPU | Auto → **BornGPU**（batch≥256 且 HasGPU） |
| WASM / js | Auto + `DeployWASM` → BornCPU（若支持），否则 Native；**无 GPU** |
| 高稀疏 CSR | 设 `SparseDensity` 或显式 Native |
| 含 cat-small 类别分裂 | 强制 Native |

## API

```go
// 仅结果
b := tree.SelectBackend(caps, hint)

// 可解释（日志 / Agent / 文档对照）
d := tree.SelectBackendExplained(caps, hint)
// d.Backend, d.Rule, d.Reason

// 加载路径
opts := &io.LoadOptions{
    Backend:  io.BackendAuto,
    Workload: tree.WorkloadHint{BatchSize: 256, HasGPU: true},
}
```

## 与 CI 门禁

- `TestBenchGateBornCPUSlowerBatch1`：batch=1 时 BornCPU ≫ Native → 证明小批不该默认 Born。
- `TestBackendAutoDecisionTable`：锁决策表核心行。
- 统一记录格式：`tree.BenchRecord`（见 [benchmark-baseline.md](benchmark-baseline.md) §记录格式）。
