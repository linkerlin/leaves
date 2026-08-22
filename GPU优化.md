# Born/GPU 使用优化路线（2026-08-22 审计）

> **背景**：v2.7.1 升级 born v0.9.23 时发现 BackendAuto 2.0「加速区」不可复现（实测 BornCPU 慢 6–30×），已诚实化为默认 Native（见 `docs/benchmark-baseline.md` §再测量）。本文记录**更深一层的架构审计结论**与分步实施计划。
> **原则**：Native golden 不变；启发式主张必须有可复现测量背书；不因「能用」保留架构错配的路径。

## 一、审计结论：推理张量 walk 是「假向量化」

实测 BornCPU 慢 6–30× 不是 born 库慢，是 leaves 侧的 walk 实现从未真正向量化：

1. **每个 depth 步都回主机跑 Go 循环**：`tree/born_walk.go` 的 `bornBatchGoLeft`/`bornMergeWalkStep`/`bornMaskNodeIndices`/`bornAllAtLeaf` 全部对 `.Data()` 做 Go 循环——张量层只贡献 `Gather` 的分配与间接，真正的计算 = Native 标量 walk 的同样循环 + 张量开销。
2. **树张量每调用重建、零缓存**：`walkTreeBatch` 每次调用为每棵树建 7–8 张张量（splitFeat/thresholds/leftChild/…），无 sync.Once 无缓存。batch 推理 100 棵树 = 每次 `PredictDense` 800 次张量构造。
3. **GPU 路径更糟**（`tree/born_gpu_walk.go`）：int32 子指针**编码成 float32** 传输（往返 `math.Round`）；每 depth 步 `.Data()` 强制 device→host 同步；树结构每调用重新上传。结构上不可能快——这是计时 ≈0/batch≥256 挂起/vkMapMemory panic 的来源。
4. **margins 用 `[][]float64`**（slice-of-slices，cache 不友好）vs Native 的扁平累加。

升级 born 22 版无效佐证：瓶颈在调用模式，不在 born 的 kernel。

## 二、合理性分项判定

| 用途 | 判定 | 依据 |
|------|------|------|
| 训练 WebGPU hist（scatter-add + 增益扫描） | ✅ 合理 | 直方图累加是真张量友好负载；`sync.Once` 惰性初始化 + panic→会话级降级，防御好 |
| ONNX graph 运行时（`LoadOnnxGraph`） | ✅ 合理 | 通用计算图是 born 本职 |
| 推理 BornCPU | ❌ 架构错配 | 树遍历是 branchy 不规则负载，非张量形状；「张量仿真」只加开销 |
| 推理 BornGPU | ❌ 现状不可用 | 计时异常、挂起、驱动 panic；树驻留/不下主机/大 batch 摊薄三前提全不满足 |
| WASM→BornCPU 规则（决策表最后一条 Auto→Born） | ⚠️ 未测 | 同样错配在 wasm 只会放大；2.1 因「无 CPU 证据」保留——**最后一个未实测的主张** |
| `LEAVES_BACKEND_PROFILE=1` 接线 | ⚠️ 真风险 | 本参考机 batch≥256 GPU 计时**挂起**——开 env + 大 batch + GPU 会在首次 Auto 选型挂死进程，无超时无防护 |

## 三、实施计划（按优先级）

### P0 — 防事故

- [x] **GPU-O1** `ProfileBackend`/`profiledDecision` sanity guard（2026-08-22 落地）：三层守卫——单后端 2s 计时预算（轮间检查，超预算 Ok=false）+ 均值 ns/op 下限 1e-3（防计时器归零，曾实测 BornGPU ≈0ns）+ `profileWithTimeout` 总超时 8s（goroutine + select，真挂起时回落 `profile_timeout`→Native）。实测：此前 batch≥256 GPU 挂死的 gate 脚本现在 120s 内完整跑完。Windows 时钟粒度兜底（elapsed==0 按 1 tick）。5 个守卫单测。
- [x] **GPU-O2** BornGPU 推理降级 experimental：`BornConfig.UseGPU` godoc 明示参考机异常与自测建议；backend-auto.md 部署表标 experimental（保留 API，parity 门禁仍用）。

### P1 — 收口最后的诚实缺口

- [x] **GPU-O3** WASM 规则实测（**结果反转预期**）：node v24 wasm_exec 三轮实测（50树×31节点×30特征）——batch=8 BornCPU **快 1.6–2.6×**（稳定，wasm 解释器拖慢 Native 标量 walk，与桌面相反）；batch≥64 打平噪声区。处置：决策表 WASM 行拆分——`batch<64 且支持 → wasm_born_cpu`；`batch≥64 或不支持 → wasm_native`（新增 rule，旧 `wasm_native_fallback` 移除）。`scripts/wasm_backend_bench` 常驻复测工具（GOOS=js）+ benchmark-baseline §WASM 实测表。
- [x] **GPU-O4** BornCPU walk 诚实路线：`BornEngine` godoc 改口为「parity/兼容路径，非加速路径」（假向量化：每步回主机 Go 循环 + 树张量零缓存，实测 0.03–0.16×）；激进重写路线明确不采纳（无用户需求）。

### P2 — 治理

- [x] **GPU-O5** CI 月度 `backend-gate` workflow（每月 1 日跑 born_upgrade_gate 双模型报数）；wgpu v0.30 异常上游 issue 草稿 `docs/upstream-wgpu-issue-draft.md`（含复现路径与疑点，待人工核对最小化后提交）。
- [x] **GPU-O6** 战略表述修正：README 计算底座节改述「**Born = 训练加速器 + ONNX 运行时；推理 golden 永远 Native（桌面端）**」+ WASM 小批例外写明；训练加速节补 `LEAVES_BORN_GPU=0` 提示。

## 四、验收

```powershell
go test ./tree ./io -count=1                      # 守卫 + 决策表测试绿
go run ./scripts/born_upgrade_gate testdata/lg_breast_cancer.txt   # 无挂起、无 ≈0 计时
go test ./docs -count=1                           # 文档门禁
```
