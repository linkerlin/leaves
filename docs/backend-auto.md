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

## 训练加速 vs 推理 BackendAuto（LIB-03）

二者 **独立**，名称相似但作用面不同——Agent / 文档勿混用：

| 维度 | 训练加速 | 推理 BackendAuto |
|------|----------|------------------|
| **配置** | `train.Config.AccelMode` 或环境变量 **`LEAVES_TRAIN_ACCEL`** | `tree.BackendAuto` + `WorkloadHint`（batch/GPU/稀疏） |
| **取值** | `auto` / `webgpu` / `born_cpu` / `cpu` | `Native` / `BornCPU` / `BornGPU`（由决策表选出） |
| **作用** | hist 增益扫描等 **训练** 路径（`treebuilder`） | 已训练模型的 **预测** 遍历（`tree` Engine） |
| **与 objective** | 无关 | 无关 |
| **文档** | README「训练加速」；`train/accel.go` | 本文决策表；`tree/backend_select.go` |

`metrics.json` 在单次 `leaves train` Fit 后可含 **`train_accel`**（实际训练加速模式）；**不含**推理 BackendAuto 字段。调试性能时分别设置 `LEAVES_TRAIN_ACCEL` / 显式 `tree.Backend`，不要用训练环境变量期望改变推理选型。

**`LEAVES_BORN_GPU=0|off|false`**（Windows）：强制 `BornWebGPUAvailable()` 返回 false，训练与推理的 WebGPU 路径全部回落 CPU。适用于无真实 GPU 的环境（CI 的 WARP 软件设备会探测通过但运行时 `DXGI_ERROR_DEVICE_REMOVED`）。

## 第二轮：opt-in profiling（`tree.ProfileBackend`）

**已交付**（v2.3.0 之后）——数据驱动选型，**不破坏 2.0 默认决策表**：

`ProfileBackend(caps, vals, nrows, ncols, iters) ProfileResult` 对实际 workload warm-up + 计时 Native / BornCPU / BornGPU，返回各后端 ns/op 与最快推荐。

- **opt-in**：默认 `SelectBackendExplained`（2.0 决策表）行为不变；需要测量证据时显式调用 `ProfileBackend`。
- **可解释**：`ProfileResult.Pick / Rule / Reason`；Rule 码 `profile_native | profile_born_cpu | profile_born_gpu | profile_none_ok | profile_invalid`，Reason 含各后端实测 ns/op。
- **不破坏 golden**：Native 始终参与计时；不支持的后端（cat-small 森林 / 无 WebGPU）`Ok=false` 不参与推荐。
- **公平**：各后端 nEstimators=0（全量树）、同输入、同 warm-up。

```go
res := tree.ProfileBackend(caps, sampleVals, nrows, ncols, 20) // iters 建议 ≥10
opts.Backend = res.Pick                    // 用测量结果替代启发式
// res.Native.NsPerOp / res.BornCPU.NsPerOp / res.BornGPU.NsPerOp / res.Reason
```

### 自动接入（v2.7.0，env opt-in）

`LEAVES_BACKEND_PROFILE=1|on|true|yes` 时，`SelectBackendExplained` 在决策表之前自动实测：

- **合成确定性样本**：rows=batch（`rows·cols ≤ 512k` 封顶）、LCG 均匀 [0,1)；三后端同输入同 warm-up，只用于后端间相对计时。
- **形状类缓存**：键 `(batch, n_features, trees, nodes)`——每形状只测一次，进程内不重复付首调用延迟（Reason 标注「缓存命中」）。
- **尊重 hint**：`HasGPU=false` 时剔除 BornGPU 取次优；WASM / 稀疏 / 非数值路径不进 profiling（各自规则优先）。
- **Rule**：`profile_native | profile_born_cpu | profile_born_gpu`；Reason 携带实测 ns/op 与样本行数。
- **默认不变**：未设 env 时决策表 2.0 行为原样（`TestBackendProfileEnvOff` 锁定）。

### 仍默认不做（需产品信号）

| 候选 | 风险 |
|------|------|
| 更细 batch 分段（16/32） | 阈值爆炸、文档漂移 |
| 设备能力细探测（VRAM、DX 版本） | 平台分支增多 |

原则不变：**任何自动选择必须能被 Rule 码解释**；改决策表须先改本文档 + `TestBackendAutoDecisionTable`。

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
