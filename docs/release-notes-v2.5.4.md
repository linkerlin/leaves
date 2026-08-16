# leaves v2.5.4

**主题**：manifest 复现契约修复——`reproduce` 不再丢路径语义。

## Highlights

1. **manifest.reproduce 补全丢参**：原构建器漏 `--cv` / `--max-leaves` / `--num-target` / `--val` / `--early-stop`。后果：CV run 的复现命令退化为全量单训（指标与记录的 `cv_mean` 不可比）；早停 run 复现丢早停语义。现按 run 类型补全——CV 行带 `--cv K`；早停行带 `--val X --early-stop N`。
2. **`params.val` 记录**：验证集路径非超参，但早停 run 的忠实复现需要（仅单跑路径记录；`--cv` 与 `--val` 并存时 val 被忽略故不记）。字段只增不删（NOTES §6）。
3. **`manifest.leaves_cli` 真版本**：硬编码占位 `agentic-1` → 真实标签（`vX.Y.Z` 或 `(devel)+<短commit>`）——工件包可追溯生成它的 CLI 版本（WP-11「leaves_version 或 build 信息」本意）。

## Usage

```json
// 早停 run 的 manifest（修复后）
"reproduce": "leaves train --data ... --rounds 200 --depth 6 --lr 0.1 ... --val hold.csv --early-stop 20 --out-model model.leaves.json --metrics metrics.json"
```

## Design notes

- **复现语义分工（写实既有行为）**：`--from-run` = 定稿/变异起点，**不回填 val**（默认全量重训，SKILL 定稿流程不变——自动回填会把定稿变成早停截断）；忠实重放某次 run 用 `manifest.reproduce` 或显式 `--val`。

## Verification

- `TestPublishReproduceFaithful`：CV run 的 repro 必含 `--cv 3`；早停 run 必含 `--val` + `--early-stop 5`；`params.val` 落盘断言。
- `TestPublishReproduce` 扩展：`leaves_cli` 非占位符。
- `go test ./cmd/leaves -count=1` 全包绿。

## Compatibility

- **Breaking**：无。`params.val` 为新增 omitempty 字段；`leaves_cli` 值从占位符变真实标签（读该字段的脚本获得更多信息）。
