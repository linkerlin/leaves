# leaves v2.1.1

**主题**：v2.1.0 之后的 Agent 契约加固、ONNX TreeEnsemble 实验子集、多目标回归闭环。

## Highlights

1. **Agent 契约门禁**：skills 镜像 CI 哈希、`--strict-flags`/`cv_conflict`、错误码表、`--print-repro`、`--out-final`、`train_accel`。
2. **ONNX（实验）**：仅 `TreeEnsembleRegressor`（`BRANCH_LEQ` / `SUM` / `NONE`）；完整 Graph 仍不做。
3. **多目标回归**：`one_output_per_tree`（API + CLI `--num-target` + predict/eval）。
4. **示例**：`examples/extension`、`examples/multitarget`；`docs/bench` 样例工件。

## Recommended API

- Infer: `LoadFromFile` + `DefaultLoadOptions`
- Train: `NewLearner` / `LoadDataAuto` / CLI `leaves train`
- Multi-target: `data.NewMultiTargetDense` + `Config.NumTarget`，或 `leaves train --num-target N`
- ONNX 子集: `io.LoadONNX` / `.onnx`（实验）

## Compatibility

- metrics/CLI：**只增字段**（`train_accel`、`final_model`、`num_target` 等）；`schema_version` 仍为 `1`
- ONNX：由占位升为**实验子集**（失败时仍给转换 hint）
- Breaking: **无**

## CI

- test (3 OS) / wasm ≤16MiB / bench-gate
- skills mirror: `go test ./docs -run TestSkillsMirrorSync`

## Docs

- [CHANGELOG](../CHANGELOG.md) · [interop-matrix](interop-matrix.md) · [cli.md](../skills/leaves-autotrain/cli.md) · [TODO](../TODO.md)
