# leaves v2.6.1

> **主题**：控制面 CLI——八段剧本从 shell 驱动（Agent 零 Go 代码）  
> **日期**：2026-08-21

## Highlights

- **`recsys/cmd/control`**：推荐生产闭环八段剧本的 shell 入口。Agent 不再需要写 Go：
  `snapshot → split → eval → from-deal → append-exposure/feedback → replay → monitor → release`
  全部为结构化文件 + 退出码 0/1/2（1=用法/IO，2=校验/内部）。
- **触发器闭环**：`monitor -triggers` 把配置驱动的触发动作写入 `fired.jsonl`，
  Agent 据此调 `release rollback`——回滚决策是规则，不是临场猜测。
- **release 状态机跨命令持久化**：`release_state.json`（含 evidence + history）；
  `confirm-promote`/`rollback` 打印 desired-state 请求 JSON，由应用侧 adapter 执行
  （leaves 仍不执行网络副作用）。
- **`release.MachineState` 导出/重构**（`Export`/`FromState`）、`eval.RankViews` 共用视图、
  `deal.ReadLog`——CLI 的支撑设施。

## Compatibility

- Breaking: none（全部为新增包/命令 + 文档）
- v2.6.0 的七包契约不变

## CI

- test（3 OS）/ lint（0 issues）/ race / wasm / bench-gate
- 本地：`go test ./... -count=1` 全绿；`go test -race -short ./recsys/...` 绿；skills 镜像门禁绿

## Docs

- `docs/recsys-loop.md` §10 控制面 CLI 指南（§11 边界）
- `skills/recsys-orchestrator` §十 shell 版剧本（`.cursor/skills` 镜像同步）
- CHANGELOG Unreleased → [2.6.1]