# leaves v2.4.0

**主题**：BackendAuto 第二轮 — opt-in profiling 探测。

## Highlights

1. **`tree.ProfileBackend`**：对实际 workload warm-up + 计时 Native / BornCPU / BornGPU，返回各后端实测 ns/op 与最快推荐。给 BackendAuto 2.0 启发式阈值补上**数据驱动**选项。
2. **不破坏 2.0 决策表**：`SelectBackendExplained`（batch 64/256、稀疏 0.15）行为零变更；profiling 是 **opt-in**。
3. **可解释**：结果带 Rule 码（`profile_native|born_cpu|born_gpu|none_ok|invalid`）+ 含各后端 ns/op 的 Reason —— 遵守 `docs/backend-auto.md` *「任何自动选择必须能被 Rule 码解释」* 原则。

## Usage

```go
// 用真实批量样本测一遍，拿测量证据替代启发式
res := tree.ProfileBackend(caps, sampleVals, nrows, ncols, 20) // iters 建议 ≥10
fmt.Println(res.Reason)
// "profiling: native fastest at 410 ns/op (native=410 born_cpu=9120 born_gpu=0)"

opts.Backend = res.Pick // 用最快后端
// 也可直接读 res.Native.NsPerOp / res.BornCPU.NsPerOp / res.BornGPU.NsPerOp
```

## Design notes

- **为何 opt-in 而非接入 Auto**：profiling 需要代表性批量样本（调用方才知道真实输入），且首调用有计时延迟。强行接入 `BackendAuto` 会引入「样本从哪来」+「首调用延迟」两个难题。故保持 2.0 决策表为默认，profiling 作为需要测量证据时的显式工具。
- **公平性**：各后端 nEstimators=0（全量树）、同输入、同 warm-up（2 轮）。
- **安全**：Native 始终参与计时（golden 不变）；cat-small 森林或无 WebGPU 的后端 `Ok=false` 自动排除。

## Verification

- `TestProfileBackendPicks`：Pick 与实测最快一致（断言一致性，非机器相关具体值）。
- `TestProfileBackendInvalid`：非法输入 → `profile_invalid` 回落 Native。
- `TestProfileBackendNoBornOnUnsupported`：cat-small 森林 → Born 排除、Native 选中。
- tree 全量、docs gate、build、6-linter 门禁全绿。

## Compatibility

- **Breaking**：无。新增 `ProfileBackend` 函数 + `ProfileResult/BackendTiming` 类型；既有 `SelectBackend`/`SelectBackendExplained` 不变。

## Recommended API

- 默认推理选型：`BackendAuto`（2.0 决策表，无需样本）
- 数据驱动选型：`ProfileBackend`（传样本，拿测量）
- 详见 [`docs/backend-auto.md`](backend-auto.md) §第二轮
