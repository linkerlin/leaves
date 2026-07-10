# leaves v2.1.0

**主题**：Agent 可驱动的训练闭环契约收口，以及库稳定/平台化第一轮（扩展点、BackendAuto、互操作、发版治理）。

## Highlights

1. **Agentic CLI**（无 MCP）：`sniff → train → eval → inspect → explain → publish`，JSON 契约 + SKILL 决策表。
2. **定稿安全**：`--early-stop` 默认 `--save-best`，磁盘模型树数 = `best_round`。
3. **可复现**：完备 `params` 账本、`train --from-run`、`publish --emit-repro-script`、`manifest.reproduce`。
4. **BackendAuto 2.0**：小批 Native、中大批 BornCPU、GPU 大批 BornGPU；规则可测可文档解释。
5. **互操作写实**：格式支持等级（稳定/实验/占位）+ 加载失败带 `hint:`。
6. **发版可勾选**：`docs/release-checklist.md` / `api-surface` / `versioning`。

## Recommended API

| 场景 | 入口 |
|------|------|
| 推理 | `LoadFromFile` + `DefaultLoadOptions()` |
| 训练 | `NewLearner` / `LoadDataAuto` / `NewLearnerFromModelAndData` |
| Agent | `go run ./cmd/leaves …` + `skills/leaves-autotrain` |
| 后端 | `LoadOptions{Backend: BackendAuto, Workload: …}` — 见 `docs/backend-auto.md` |

兼容旧 API（`LGEnsembleFromFile` 等）仍可用，不作为新代码首发入口 — 见 `docs/api-surface.md`。

## Compatibility

- **AutoTransform 默认 true**：logistic 等返回变换后值；raw margin 请设 `AutoTransform: false`（`NOTES.md`）。
- **BackendAuto**：batch≥64 数值树倾向 BornCPU；batch=1 仍为 Native。
- **破坏性**：无删除稳定 API；metrics 字段只增不改义。
- **实验**：scikit-learn pickle；**占位**：ONNX（请先转 JSON/leaves）。

## CLI（Agent）

```powershell
go run ./cmd/leaves sniff --data train.csv --metrics sniff.json
go run ./cmd/leaves train --data train.csv --objective reg:squarederror --cv 5 `
  --runs runs.jsonl --tag baseline --out-model m.leaves.json --metrics metrics.json
go run ./cmd/leaves train --data train.csv --from-run runs.jsonl --tag baseline `
  --out-model final.leaves.json --metrics final.json
go run ./cmd/leaves publish --model final.leaves.json --out-dir release/ `
  --metrics final.json --quantize --export-xgb --emit-repro-script both
```

## Docs

- `docs/api-surface.md` — API 分层  
- `docs/backend-auto.md` — 后端决策表  
- `docs/interop-matrix.md` — 格式支持等级  
- `docs/extension-points.md` — 自定义 objective/metric  
- `docs/release-checklist.md` — 发版检查表  
- `docs/versioning.md` — v2.x 变更边界  
- `演进方案.md` — Agentic DoD（已达成）  
- `CHANGELOG.md` — 变更列表  

## CI

- `go test ./...`（3 OS）  
- WASM 体积 ≤16 MiB  
- Windows：batch=1 BornCPU ≥20× Native  

## Install

```powershell
go get github.com/linkerlin/leaves@v2.1.0
# 或
go install github.com/linkerlin/leaves/cmd/leaves@v2.1.0
```

（需 tag 推送后模块代理可见。）
