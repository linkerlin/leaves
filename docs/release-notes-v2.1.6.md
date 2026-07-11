# leaves v2.1.6

**主题**：并发 DATA RACE 与 transformation panic 修复；golangci-lint v2 门禁全量落地。

## Highlights

1. **修复并发 DATA RACE**（`treebuilder.parallelEvalHistFeats`）：多 goroutine 共享同一 `row` buffer 写入 `Dense.Row()`，race detector 首启即捕获；改为每 goroutine 独立 buffer。影响多线程 `hist` 训练正确性。
2. **修复 transformation 数组越界 panic**：`TransformType(Exponential).Name()` 因 `transformNames` 缺 `exponential` 项而越界崩溃；`TransformExponential.Name()` 因此被绕成错报 "logistic"。加载 LightGBM exponential objective 模型时触发。
3. **golangci-lint v2 门禁全量 blocking**：`govet` / `ineffassign` / `unused` / `staticcheck` / `errcheck` + **gofmt** formatter，6 项 0 issues 基线；附 `race detector` CI job。
4. **死代码清除 ~360 行**：unused linter 全清零（explain interventional SHAP、treebuilder 被取代的 GPU hist 旧路径、孤立 helper、永跳测试、失效 GoMLX 脚本、误入库产物）。
5. **3 包从 0 测试覆盖**（`transformation` / `predict` / `booster`，+17 测试）、6 包补 `doc.go`、文档治理（维护期→开发期、版本一致性 gate）。

## Fixed

- `treebuilder/hist_parallel.go`：per-goroutine `row` buffer，消除 `parallelEvalHistFeats` 数据竞争。
- `transformation/transformation.go` + `exponential.go`：补 `transformNames` 的 `exponential` 项；`TransformExponential.Name()` 返回正确名称。

## CI / 工程

- `.golangci.yml`（v2）：6 项门禁全 blocking，基线 0 issues。
- `.github/workflows/ci.yml`：新增 `lint`（`golangci/golangci-lint-action@v6`, v2.11.4）与 `race`（`go test -race -short`）job。
- 全仓 `gofmt` 归一（CRLF→LF + 对齐 + 1 处 composite literal 类型名简化）。

## Compatibility

- **Breaking**：无。
- 推理 / 训练主路径行为不变；修复仅提升并发训练正确性与 exponential 模型加载健壮性。
- 根包兼容层 `LGEnsembleFromFile` / `XGEnsembleFromFile` 仍委托可用。

## Recommended API

- Infer：`LoadFromFile` + `DefaultLoadOptions`
- Train：`NewLearner` / `LoadDataAuto` / CLI `leaves train`
- 详见 [`docs/api-surface.md`](api-surface.md)

## Docs

- 维护期 → 开发期；`演进计划` v5.4 / `演进方案` v1.5 版本引用全局一致（新增 `TestDocVersionRefsConsistent` gate）。
- 6 个公共包补 `doc.go`：`train` / `explain` / `objective` / `metrics` / `mat` / `util`。
