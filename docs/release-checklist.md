# leaves v2.1 发布检查表

> 用途：打 tag / 写 Release Notes 前逐项勾选。  
> 对齐：[`演进计划.md`](../演进计划.md) Phase E；兼容策略见 [versioning.md](versioning.md)；API 分层见 [api-surface.md](api-surface.md)。

## 0. 版本元数据

- [ ] 将 [`CHANGELOG.md`](../CHANGELOG.md) 的 `[Unreleased]` 落成 `## [2.1.0] - 日期`
- [ ] `README` / `README.en` badge 与 tag 一致（如 `v2.1.0`；开发期可为 `v2.x-dev`）
- [ ] 中英 README 互链：`README.md` ↔ `README.en.md`（无 `README.zh.md` 死链）
- [ ] `go.mod` module path 未误改
- [ ] Release 标题：`v2.1.x` + 一句话主题
- [ ] Release body 可从 CHANGELOG + 下方「建议 Release 段落」生成

## 1. 测试与 CI

| 检查 | 命令 / 位置 | 通过条件 |
|------|-------------|----------|
| 全量单元/集成 | `go test ./... -count=1` | 3 OS CI `test` job 绿 |
| WASM 构建 + 体积 | CI `wasm` / `LEAVES_WASM_GATE=1` | `leaves.wasm` ≤ 16 MiB |
| Backend 负向门禁 | CI `bench-gate`（Windows） | batch=1 BornCPU ≥ 20× Native |
| CLI 契约 | `go test ./cmd/leaves -count=1` | Agentic 优化环 / save-best / from-run |
| 扩展点 | `go test ./objective ./metrics -count=1` | Register 路径 |
| BackendAuto | `go test ./tree -run 'BackendAuto\|SelectBackend' -count=1` | 决策表 |
| 互操作 | `go test ./io -count=1` | SupportTable / LoadError / ONNX |

本地快速门禁（发布前最少跑）：

```powershell
go test ./... -count=1
go test ./cmd/leaves ./tree ./io ./objective ./metrics -count=1
```

## 2. 文档同步

| 文档 | 检查 |
|------|------|
| [README.md](../README.md) / [README.en.md](../README.en.md) | 推荐入口、支持等级、BackendAuto 2.0 一致 |
| [docs/api-surface.md](api-surface.md) | 推荐 / 兼容 / 实验分层无过期 API |
| [docs/versioning.md](versioning.md) | v2.x 允许变更边界 |
| [docs/interop-matrix.md](interop-matrix.md) | 与 `io.SupportOf` 一致 |
| [docs/backend-auto.md](backend-auto.md) | 阈值与实现一致 |
| [docs/extension-points.md](extension-points.md) | Register 说明有效 |
| [docs/testdata-matrix.md](testdata-matrix.md) | 关键路径仍存在 |
| [docs/benchmark-baseline.md](benchmark-baseline.md) | 门禁命令有效 |
| [演进方案.md](../演进方案.md) | Agentic DoD 仍成立 |
| [NOTES.md](../NOTES.md) | 仅保留仍影响用户的兼容注记 |
| [AGENTS.md](../AGENTS.md) | 文档链接与版本号 |

## 3. 示例与差异化能力

| 示例 | 检查 |
|------|------|
| [examples/autotrain](../examples/autotrain/) | 零准备 CLI 闭环可跑 |
| [examples/train_from_model](../examples/train_from_model/) | `NewLearnerFromModelAndData` |
| [examples/extension](../examples/extension/) | 自定义 objective/metric Register |
| [examples/multitarget](../examples/multitarget/) | multi-target one_output_per_tree |
| ONNX 子集 | `go test ./io -run ONNX -count=1` |
| [examples/wasm](../examples/wasm/) | 构建 + README 部署建议 |
| [examples/http](../examples/http/) | embed 批预测 demo（非官方 serving） |
| [examples/serving-template](../examples/serving-template/) | 可拆独立仓 serving 脚手架 |
| quantize | `publish --quantize` 或 `quantize` 包测试绿 |
| explain | `leaves explain` / `model.Explain` 冒烟 |

## 4. 兼容与互操作

- [ ] 稳定格式加载路径未回归（LGB / XGB JSON·UBJ·bin / leaves.json）
- [ ] SK 仍标 **实验**；ONNX 为 **实验子集**（非完整 Graph）
- [ ] `DefaultLoadOptions().AutoTransform == true` 行为有 NOTES/README 说明
- [ ] 根包 `LGEnsembleFromFile` / `XGEnsembleFromFile` 仍委托可用（兼容层）
- [ ] 无「静默破坏」metrics/CLI schema（升 `schema_version` 须记 versioning）

## 5. Benchmark 摘要（可选但推荐）

- [ ] 若本版本改动推理/BackendAuto：附 `tree.BenchRecord` 样例或说明「无性能意图变更」
- [ ] 决策表 Rule 码未无文档漂移

## 6. 明确不在本版本

对照 [演进计划 §七](../演进计划.md) / [演进方案 非目标](../演进方案.md)：

- 无内置 HPO / 官方 registry / 分布式训练 / 完整 ONNX Graph

## 建议 Release Notes 骨架

```markdown
## leaves v2.1.x

### Highlights
- …

### Recommended API
- Infer: `LoadFromFile` + `DefaultLoadOptions`
- Train: `NewLearner` / `LoadDataAuto` / CLI `leaves train`
- See docs/api-surface.md

### Compatibility
- AutoTransform default: …
- Breaking: none | list

### CI
- test (3 OS) / wasm ≤16MiB / bench-gate

### Docs
- interop-matrix, backend-auto, extension-points, release-checklist
```

##  ent 后

- [ ] GitHub Release 已发布
- [ ] （可选）`go install` / 模块代理可拉 tag
- [ ] 若有 Agentic 契约变更：同步 `skills/leaves-autotrain` 与 `.cursor/skills` 镜像（`go test ./docs -run TestSkillsMirrorSync`）
