# leaves v2.8.0

> **主题**：全面 Agentic 收口（REV-01..09）——io 自足加载 · WASM 决策收敛 Native · 文档-代码对齐  
> **日期**：2026-08-24

## Highlights

1. **io 包自足加载（REV-01/02）**：LGB text/JSON 解析迁入 `io/` 直达
   `ForestIR`；`io.LoadFromFile` 加载 LGB / XGB（JSON·UBJ·bin）/ leaves.json /
   ONNX / sklearn pickle 全部不再依赖根包 init。新增
   `io.ParseSklearnPickleFile`（SK pickle → ForestIR，实验路径）；
   `SKEnsembleFromFile` 兼容入口保留。
2. **LGB 遵循 AutoTransform（行为变更，可恢复）**：`io.LoadFromFile` 对 LGB
   与 XGB / leaves.json 一致默认套转换；要 raw 显式 `AutoTransform: false`。
   `LGEnsembleFromFile` 布尔参数语义不变。
3. **WASM Auto 一律 Native（REV-04）**：`GOOS=js` 上 `BornEngine` 委托
   Native；决策表规则码仅 `wasm_native`（v2.7.2 的 `wasm_born_cpu`
   小批主张作废）。
4. **工程收口**：`recsys/trainrank` 不再 import `demos/`（REV-03）；
   backend-gate 默认 `windows-latest` + `LEAVES_BORN_GPU=0`（REV-05）；
   Born 守卫——cat-small walk 回落标量、显式 GPU Predict 8s 超时
   （REV-08）；godoc/注释去 GoMLX 旧话术（REV-09）。
5. **文档-代码对齐**：推荐 import 修正（`/v2` 不再写两遍）、ONNX
   TreeEnsembleClassifier 子集如实标注、BackendAuto 2.1 英中决策表一致、
   README/AGENTS CLI 入口表；新增 `TestNoDoubleV2Import` 与
   `TestReleaseDocsPresent` glob 门禁。

## Compatibility

- **LGB `LoadFromFile` 默认套 AutoTransform**：与 XGB 一致；raw 用
  `AutoTransform: false` 恢复。详见 `NOTES.md`
- **WASM 决策表**：`wasm_born_cpu` 移除，仅 `wasm_native`
- **删除恒返回 nil 的 `tree.LgTreeToTreeIR`**（占位 API，无实际调用方）；
  serving-template 依赖对齐 born v0.9.23
- metrics / CLI schema 无既有字段改义（只增）

## CI

- test（3 OS）/ lint / race / wasm / bench-gate / fuzz（每周）/ backend-gate（每月）
- 本地全量 `go test ./... -count=1` 绿

## Docs

- `docs/api-surface.md`：REV 分层同步
- `docs/interop-matrix.md`：ONNX Classifier 子集 + `LoadOnnxGraph`
- `NOTES.md`：LGB AutoTransform 迁移注记
- `skills/leaves-autotrain/cli.md`：`leaves_cli` 真实版本标签
