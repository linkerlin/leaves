尽可能 Go 语言来实现功能！
======
尽可能模块化！
======

## 计算底座（2026-06-13 决策，2026-06-15 文档同步）

- **已废弃**：GoMLX（`github.com/gomlx/gomlx`）、gogpu/wgpu 直连、`treebuilder/hist_accel_born.go`、`born_train` build tag
- **统一底座**：[Born](https://github.com/born-ml/born)（`github.com/born-ml/born`）
- **推理**：`tree.NativeEngine`（golden）+ `tree.BornEngine`（Born CPU / **WebGPU Windows**）；js/wasm 仅 Native
- **训练加速**：`treebuilder/hist_accel.go`（Born CPU 增益扫描）+ `hist_accel_webgpu_*.go`（WebGPU hist）；环境变量 `LEAVES_TRAIN_ACCEL` / `train.Config.AccelMode`
- **无中间 IR**：Born 路径不维护 `TreeData`/`ForestData` 快照；`tree/born_walk.go` 等对 `ForestIR` 做张量遍历
- **包边界**：`tree/` 不依赖 `train/`

## Backend 命名

| 常量 | 含义 |
|------|------|
| `BackendNative` | 纯 Go 标量遍历（golden） |
| `BackendBornCPU` | Born CPU 后端（SIMD） |
| `BackendBornGPU` | Born WebGPU 后端（Windows DX12） |
| `BackendAuto` | 按 workload 在 Native / Born 间选择 |

## 文档

- 战略路线图：[`演进计划.md`](演进计划.md) v4.3
- 可执行 backlog：[`TODO.md`](TODO.md)（P0–T5 + v3.1 已完成）
- 回归矩阵：[`docs/testdata-matrix.md`](docs/testdata-matrix.md)

## 格式与 IO（v4.3）

- **训练数据**：`data/sniff.go` + `data/fromfile.go`（`FromFileAuto` / `LoadDataAuto`）
- **模型加载**：`io/load.go`（格式探测）、`io/transform_auto.go`（`AutoTransform` 默认 true）
- **便利 API**：`train/load.go`、`train_api.go`（`NewLearnerFromModelAndData`）
