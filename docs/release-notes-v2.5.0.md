# leaves v2.5.0

**主题**：Agent 演化搜索（GEPA 对标）+ 用户侧 Agent 入口 + `/v2` 文档对齐。

## Highlights

1. **演化搜索账本信号（EVO-02）**：`train` 的 metrics.json 与 runs.jsonl 账本行新增 `n_trees` / `elapsed_ms`（`--cv` 另有 `fold_metrics`）。Agent 可做「指标 vs 模型大小/耗时」权衡、折级 Pareto 选父与筛选晋级。字段只增不删（omitempty），`schema_version` 维持 `1`。
2. **SKILL §4.5 演化搜索协议（EVO-01/04）**：把 Agent 调参从「贪心坐标下降」升级为带反射的演化搜索——Hall-of-Fame + 折级 Pareto 选父、反射式变异（读 `--emit-rounds` 曲线 / `explain` 重要性 / 错误码等 ASI → 写假设 → 定向变异）、交叉重组、`--cv 2` 筛选→`--cv 5` 晋级、预算帽 ≤15、谱系 tag（`p:<父>+<变异>`）。**策略仍在 SKILL 文本，库不内置 HPO**（方法论对标见 [`演进方案.md`](../演进方案.md) §十六）。
3. **用户三步开工**：`go install github.com/linkerlin/leaves/v2/cmd/leaves@latest` → 把技能给 Agent（Cursor 开箱 / 新增 `CLAUDE.md`）→ 说一句话（「数据在 X，把 RMSE 降下来，收敛后发布」）。SKILL walkthrough 以实跑数字重写（谱系 `baseline → p:baseline+depth6_lr01 → …+lambda2`，反射轮 0.218→0.2129）。
4. **修复安装与文档模块路径**：README 安装命令原漏 `/v2` 与 `cmd/leaves`（仓库外装不上）；全仓文档 godoc/import/require 对齐 `github.com/linkerlin/leaves/v2`；NOTES §4 过时 `go get` 建议重写；`testscripts/compatibility_*.py` 模板修正。

## Usage

```powershell
# 账本行现在携带演化搜索信号（n_trees / elapsed_ms / fold_metrics）
go run ./cmd/leaves train --data train.csv --objective reg:squarederror `
  --cv 5 --rounds 40 --depth 4 --lr 0.2 `
  --runs runs.jsonl --tag baseline

# runs.jsonl 每行：
# {"tag":"baseline",...,"fold_metrics":[0.232,0.228,...],"n_trees":40,"elapsed_ms":1824,"params":{...}}
```

Agent 侧策略见 [`skills/leaves-autotrain/SKILL.md`](../skills/leaves-autotrain/SKILL.md) §4.5。

## Design notes

- **为何只加信号不加策略**：与「Agent 即优化器、leaves 即目标函数」的既有分解一致（演进方案 §五原则 1）；GEPA 的搜索循环属于 SKILL 文本，库只保证信号诚实。
- **对标结论**：leaves 信号契约（schema_version / maximize / model_round / seed 可复现）优于多数 GEPA adapter；短板全在 SKILL 侧搜索策略，本版本以 SKILL 协议补齐。
- **EVO-03 测试口径**：Go 测试锁「信号存在性」（有模型行必有 `n_trees`、CV 行必有 `fold_metrics`）；策略正确性由 SKILL walkthrough 与实战回归背书。

## Verification

- 全量 `go test ./... -count=1` 26 包绿（Windows 本地）。
- `TestAgenticOptimizeLoopSmoke` 扩展：第三轮 `--cv 2` 锁定账本信号。
- docs 门禁（`TestSkillsMirrorSync` 镜像哈希 + `TestDocVersionRefsConsistent` 版本引用）绿。
- SKILL walkthrough 数字为实跑验证（seed=42，2026-08-16）。
- `testscripts`：`py_compile` 过；`require /v2 + replace` 模式经临时模块 `go build` 实证（harness 本身 POSIX-only）。

## Compatibility

- **Breaking**：无。新增 JSON 字段全部 omitempty；`schema_version` 不变。
- 兼容 harness `testscripts/compatibility_*.py` 已随模块路径修正（无 CI 引用，POSIX-only）。

## Recommended API

- Agent 闭环：`leaves sniff/train/eval/predict/inspect/explain/publish` + runs.jsonl 账本
- 推理：`LoadFromFile` + `DefaultLoadOptions`；训练：`NewLearner` / `LoadDataAuto`
- 详见 [`docs/api-surface.md`](api-surface.md)
