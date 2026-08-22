# leaves v2.7.2

> **主题**：GPU/Born 使用优化（GPU-O1..O6）——profiling 挂死守卫 · WASM 实测反转 · 叙事诚实化  
> **日期**：2026-08-22  
> **审计与路线**：[`GPU优化.md`](../GPU优化.md)

## Highlights

1. **profiling 挂死守卫（GPU-O1）**：`LEAVES_BACKEND_PROFILE=1` 在 wgpu 异常机器上
   曾会挂死进程（batch≥256 GPU 计时永不返回）。现三层守卫——单后端 2s 预算 +
   ns/op 下限（防计时器归零）+ 总超时 8s 回落 Native。实测 gate 脚本从
   「挂死」变为 120s 完整跑完。
2. **WASM 实测反转（GPU-O3）**：最后一个未测的决策表主张——WASM 上 BornCPU
   小批量**真的快 1.6–2.6×**（与桌面格局相反：wasm 解释器拖慢 Native 标量 walk）。
   决策表拆分：WASM batch<64 → BornCPU；≥64 → Native。
3. **叙事诚实化（GPU-O2/O4/O6）**：推理 Born godoc 改口「parity/兼容路径，
   非加速路径」（假向量化：每步回主机 Go 循环 + 树张量零缓存）；BornGPU 标
   experimental；README 战略表述改为「Born = 训练加速器 + ONNX 运行时；
   推理 golden 永远 Native（桌面端）」。
4. **治理（GPU-O5）**：月度 `backend-gate` CI job 复测双模型；wgpu v0.30.x
   异常上游 issue 草稿入库。

## WASM 实测表（node v24，50 树×31 节点×30 特征）

| batch | Native | BornCPU | 结论 |
|-------|--------|---------|------|
| 8 | 162–289k ns/op | 101–113k | **BornCPU 快 1.6–2.6×** |
| 64 | 510–844k | 528–610k | 打平（噪声区） |
| 256 | 2.1–3.2M | 2.1–2.8M | 打平（噪声区） |

复测：`scripts/wasm_backend_bench`（GOOS=js 构建 + wasm_exec_node 运行）。

## Compatibility

- Rule 码：新增 `wasm_native`（WASM batch≥64 或 Born 不支持）；`wasm_native_fallback`
  移除。WASM 小批行为不变（仍 BornCPU），WASM 大批从 BornCPU → Native（打平区取 golden）
- `ProfileBackend` 签名不变；新增 `ProfileBackendWithBudget` / `DefaultProfileBudget`
- `ProfileResult` 结构不变；不可用后端 `Ok=false` 语义扩展（预算耗尽/计时异常/超时）

## CI

- test（3 OS）/ lint / race / wasm / bench-gate / fuzz（每周）/ backend-gate（每月，新增）
- 本地双轮（GPU on/off）`go test ./... -count=1` 绿；lint 0 issues

## Docs

- `GPU优化.md`：审计结论 + 六项落地记录（本 release 的路线真源）
- `docs/backend-auto.md`：WASM 行拆分 + experimental 标注
- `docs/benchmark-baseline.md`：§WASM 实测
- `docs/upstream-wgpu-issue-draft.md`：上游报告草稿
