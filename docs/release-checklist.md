# leaves v2.x 发布检查表

> 用途：打 tag / 写 Release Notes 前逐项勾选。  
> 对齐：[`演进计划.md`](../演进计划.md) Phase E；兼容策略见 [versioning.md](versioning.md)；API 分层见 [api-surface.md](api-surface.md)。  
> 流程教训（2026-08 六连发沉淀）：**CI 必须在打 tag 前全绿**（v2.5.0 曾带红 CI 发布、事后补修，顺序不可重复）；**module zip 卫生**要在 tag 前查（git 追踪文件 = `go get` 分发内容，v2.5.5 曾膨胀到 21.8MB）。

## 0. 版本元数据

- [ ] 将 [`CHANGELOG.md`](../CHANGELOG.md) 的 `[Unreleased]` 落成 `## [<X.Y.Z>] - 日期`
- [ ] `README` / `README.en` badge 与 tag 一致（版本号 + releases/tag 链接两处）
- [ ] 中英 README 互链：`README.md` ↔ `README.en.md`（无 `README.zh.md` 死链）
- [ ] `go.mod` module path 未误改（`github.com/linkerlin/leaves/v2`）
- [ ] Release 标题：`vX.Y.Z` + 一句话主题
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

**打 tag 前硬性门禁（教训：v2.5.0 带红 CI 发布，事后补修 v2.5.1）**：

- [ ] release commit 的 **CI 全绿后再打 tag**（`gh run list --limit 1` 确认）；勿「先 tag 后补修」

**module zip 卫生（教训：v2.5.5 前 zip 膨胀至 21.8MB；git 追踪文件 = `go get` 分发内容）**：

- [ ] `git ls-files | rg "\.(exe|dll|wasm|db|zip)$"` 为空（可重建产物不入库；testdata 除外，回归矩阵必需）
- [ ] 其他 Agent 工作目录（`.chong/` 等）未被追踪

## 2. 文档同步

| 文档 | 检查 |
|------|------|
| [README.md](../README.md) / [README.en.md](../README.en.md) | 推荐入口、支持等级、BackendAuto 2.1 一致 |
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
- [ ] SK 仍标 **实验**；ONNX TreeEnsemble 为 **实验子集**（Regressor + Classifier）；完整 Graph 走 `LoadOnnxGraph`（非 wasm）
- [ ] `DefaultLoadOptions().AutoTransform == true` 行为有 NOTES/README 说明
- [ ] 根包 `LGEnsembleFromFile` / `XGEnsembleFromFile` 仍委托可用（兼容层）
- [ ] 无「静默破坏」metrics/CLI schema（升 `schema_version` 须记 versioning）

## 5. Benchmark 摘要（可选但推荐）

- [ ] 若本版本改动推理/BackendAuto：附 `tree.BenchRecord` 样例或说明「无性能意图变更」
- [ ] 决策表 Rule 码未无文档漂移

## 6. 明确不在本版本

对照 [演进计划 §七](../演进计划.md) / [演进方案 非目标](../演进方案.md)：

- 无内置 HPO / 官方 registry / 分布式训练 / 官方 serving 框架

## 建议 Release Notes 骨架

```markdown
## leaves vX.Y.Z

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

## tag 后（发版验证仪式，2026-08 沉淀）

- [ ] GitHub Release 已发布且为 **Latest**
- [ ] CI：tag 所指 commit 全绿（7 job：3 OS test / lint / race / wasm / bench-gate）
- [ ] 代理可拉且指向正确 commit：`go mod download -json github.com/linkerlin/leaves/v2@<tag>` 的 Hash == `git rev-parse <tag>^{commit}`（force-move 过 tag 时必查）
- [ ] 仓库外二进制自检：`go install github.com/linkerlin/leaves/v2/cmd/leaves@<tag>` → `leaves version` 自报该 tag
- [ ] （行为变更时）仓库外用户路径冒烟：`sniff → train --cv → publish`，manifest 的 `leaves_cli`/`reproduce` 正确
- [ ] （大改时）fresh clone 冒烟：`git clone --depth 1 --branch <tag>` → README 首命令可跑
- [ ] 若有 Agentic 契约变更：同步 `skills/leaves-autotrain` 与 `.cursor/skills` 镜像（`go test ./docs -run TestSkillsMirrorSync`）
