# BackendAuto 2.1 决策表

> 真源实现：[`tree/backend_select.go`](../tree/backend_select.go)  
> 回归：`go test ./tree -run 'SelectBackend|BackendAuto|BenchRecord' -count=1`  
> 对齐：[`演进计划.md`](../演进计划.md) Phase C；2.1 诚实化见 [benchmark-baseline.md](benchmark-baseline.md) §再测量

## 设计原则

1. **Native 是正确性 golden**；Born 推理是 parity/兼容路径（训练加速与 ONNX Graph 另算）。
2. **默认 Native**（2.1）：历史 batch 阈值选 Born 的「加速区」在参考机上不可复现（见 §2.1 变更说明与 benchmark-baseline 实测表）；走 Born 的两条路——**显式 `BackendBornCPU`/`BackendBornGPU`**，或 **`LEAVES_BACKEND_PROFILE=1` 实测选型**（测得更快才选）。
3. **任何自动选择必须能被规则码解释**（`SelectBackendExplained`）；启发式主张必须有可复现测量背书。
4. **显式 `Backend` 覆盖 Auto**（`ResolveBackend` 非 Auto 直接返回）。

## 阈值常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `AutoBatchCPUThreshold` | **64** | 2.1 起 small_batch / native_batch 规则码分界（选择均为 Native） |
| `AutoBatchGPUThreshold` | **256** | Deprecated：2.1 起 Auto 不再按 batch 选 BornGPU |
| `AutoSparseDensityMax` | **0.15** | `0 < SparseDensity < 0.15` → Native |

## 决策表（按优先级）

| 优先级 | 条件 | 后端 | Rule 码 |
|--------|------|------|---------|
| 1 | 线性模型 / 无森林 | Native | `linear_or_empty` |
| 2 | `Target=WASM` | Native | `wasm_native`（js 上 BornEngine 委托 Native，无独立 walk） |
| 3 | `0 < SparseDensity < 0.15` | Native | `sparse` |
| 4 | 非纯数值 / cat-small | Native | `non_numeric_or_unsupported` |
| 5 | batch≥64（CPU 或 GPU） | Native | `native_batch` |
| 6 | 其余（含默认 batch=1） | Native | `small_batch` |

`LEAVES_BACKEND_PROFILE=1` 时优先级 0 为 **实测选型**（`profile_native|profile_born_cpu|profile_born_gpu`，见 §自动接入）；形状类缓存，每形状只测一次。

### 2.1 变更说明（2026-08-22，随 born v0.9.23 升级验证发现）

2.0 决策表的 6–8 行（born_gpu / born_cpu_gpu_unavailable / born_cpu）主张 batch≥64 时 BornCPU 有 2–5× 加速。**再测量不可复现**：lg_breast_cancer（39 树/30 特征/849 节点）与合成森林（100 树×63 节点）上，born **v0.9.1 与 v0.9.23** 的 BornCPU 在 batch 64–4096 全段为 Native 的 0.03–0.16×（慢 6–30×）；BornGPU 计时异常（≈0 或挂起，wgpu v0.30.x）。历史主张可能来自更早 born 版本或不同测量口径。修复原则：**启发式不能宣称未经当前版本复测的加速**——默认 Native，Born 留显式与实测两条门。若你的硬件/模型形状测得 Born 更快：`opts.Backend = tree.BackendBornCPU` 或设 `LEAVES_BACKEND_PROFILE=1`，并把 BenchRecord 数字贡献到 benchmark-baseline。

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

**已交付**（v2.3.0 之后，v2.7.0 自动接入）——数据驱动选型，**不破坏 2.1 默认决策表**（桌面 Auto = Native）：

`ProfileBackend(caps, vals, nrows, ncols, iters) ProfileResult` 对实际 workload warm-up + 计时 Native / BornCPU / BornGPU，返回各后端 ns/op 与最快推荐。

- **opt-in**：默认 `SelectBackendExplained`（2.1 决策表）行为不变；需要测量证据时显式调用 `ProfileBackend`。
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
- **默认不变**：未设 env 时决策表 2.1 行为原样（桌面 Native；`TestBackendProfileEnvOff` 锁定）。

**WASM / js**：`GOOS=js` 下 `tree.BornEngine` 委托 `NativeEngine`（`tree/born_js.go`）。BackendAuto 一律 `wasm_native` → Native。显式 `BackendBornCPU` 在 js 上仍是 Native 包装。

### 仍默认不做（需产品信号）

| 候选 | 风险 |
|------|------|
| 更细 batch 分段（16/32） | 阈值爆炸、文档漂移 |
| 设备能力细探测（VRAM、DX 版本） | 平台分支增多 |

原则不变：**任何自动选择必须能被 Rule 码解释**；改决策表须先改本文档 + `TestBackendAutoDecisionTable`。

## 部署建议（写实，2.1）

| 场景 | 建议 |
|------|------|
| 在线单条 / 小批（batch≪64） | **Native**（默认 Auto 即如此） |
| 大批量离线打分（batch≥64） | **Native**（默认）；实测 Born 更快再 `BackendBornCPU` 或 `LEAVES_BACKEND_PROFILE=1` |
| Windows + 大批量 + GPU | **experimental**：显式 `BackendBornGPU`（参考机 wgpu v0.30.x 计时异常/挂起，v2.7.2 起 profiling 有预算+超时守卫；自测后再上） |
| WASM / js | **Native**（BornEngine 在 js 上委托 Native） |
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
