尽可能 Go 语言来实现功能！
======
尽可能模块化！
======

## 计算底座（2026-06-13 决策）

- **已废弃**：GoMLX（`github.com/gomlx/gomlx`）、gogpu/wgpu 直连
- **统一底座**：[Born](https://github.com/born-ml/born)（`github.com/born-ml/born`，本地 `C:\GitHub\born`）
- **推理**：`tree.NativeEngine`（golden）+ `tree.BornEngine`（Born CPU / **WebGPU Windows**）
- **训练加速**：`//go:build born_train` → `treebuilder/hist_accel_born.go`（直接 `github.com/born-ml/born`）
- **无中间 IR**：Born 路径不维护 `TreeData`/`ForestData` 快照；`tree/born_walk.go` 等对 `ForestIR` 做张量遍历
- **包边界**：`tree/` 不依赖 `train/`

## Backend 命名

| 常量 | 含义 |
|------|------|
| `BackendNative` | 纯 Go 标量遍历（golden） |
| `BackendBornCPU` | Born CPU 后端（SIMD） |
| `BackendBornGPU` | Born WebGPU 后端（Windows DX12） |
| `BackendAuto` | 按 workload 在 Native / Born 间选择 |
