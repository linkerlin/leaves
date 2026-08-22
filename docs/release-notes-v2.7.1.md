# leaves v2.7.1

> **主题**：born v0.9.23 升级 + BackendAuto 2.1 诚实化（决策表「加速区」不可复现 → 默认 Native）  
> **日期**：2026-08-22

## Highlights

1. **born v0.9.1 → v0.9.23**：计算底座追平 22 个上游版本。编译零断裂；
   全量测试（GPU on/off 双轮）、parity 门禁、wasm 构建（3.47MB / 16MiB 门禁）、
   lint 全绿。
2. **升级验证发现并修复决策表缺陷**：2.0 决策表宣称 batch≥64 BornCPU
   「加速区 2–5×」。用官方计时路径（`tree.ProfileBackend`）在基线同款模型上复测，
   **born v0.9.1 与 v0.9.23 均为 Native 的 0.03–0.16×（慢 6–30×）**——
   不是升级回退，是历史主张从未在当前版本复现。负向门禁只测了 batch=1，
   正向主张无门禁看护，故此腐烂长期不可见。
3. **BackendAuto 2.1**：Auto 默认 **Native**（任意 batch，CPU/GPU）；
   走 Born 两条路——显式 `BackendBornCPU`/`BackendBornGPU`，或
   `LEAVES_BACKEND_PROFILE=1` 实测选型（v2.7.0 特性恰好成为本修复的
   测量逃生门）。WASM 规则保留（无证据翻转）。
4. **`scripts/born_upgrade_gate`**：born 升级/决策表调整的常驻复测工具，
   防止启发式主张再次无测量看护。

## 实测表（lg_breast_cancer，39 树/30 特征/849 节点）

| batch | Native | BornCPU v0.9.1 | BornCPU v0.9.23 |
|-------|--------|----------------|-----------------|
| 64 | ~99k ns/op | 0.04× | 0.03× |
| 256 | ~333k ns/op | 0.05× | 0.05× |
| 1024 | ~1.1M ns/op | 0.04× | 0.04× |

完整口径与复测命令见 `docs/benchmark-baseline.md` §再测量。

## Compatibility

- **行为变更**：`BackendAuto` 在 batch≥64（含 HasGPU）不再选 Born——
  此前会被 Auto 派发到 Born 的用户将默认获得 **显著更快** 的 Native 路径；
  显式指定 Born 的代码不受影响
- Rule 码：`born_gpu`/`born_cpu_gpu_unavailable`/`born_cpu` 移除，
  新增 `native_batch`；`SelectBackend`/`ResolveBackend` API 不变
- `AutoBatchGPUThreshold` 标记 Deprecated（保留常量）

## CI

- test（3 OS）/ lint / race / wasm（3.47MB ≤ 16MiB）/ bench-gate（batch=1 负向门禁）
- 本地双轮：GPU on / `LEAVES_BORN_GPU=0` 全量 `go test ./... -count=1` 绿

## Docs

- `docs/backend-auto.md`：2.1 决策表 + §2.1 变更说明
- `docs/benchmark-baseline.md`：§再测量（两版本实测表）+ 历史口径存档
- README 速查表 / AGENTS.md 对齐
