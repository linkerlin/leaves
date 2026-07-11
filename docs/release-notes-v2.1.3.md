# leaves v2.1.3

**主题**：MovieLens 推荐 Ranker 全流程 — Agent JSON CLI + MCP + 详尽教程。

## Highlights

1. **Agent 契约 CLI**：`go run ./demos/movielens/cmd/agent`（prepare → train → eval → recommend，stdout JSON）。
2. **MCP server**：`go run ./demos/movielens/cmd/mcp`（stdio；`movielens_status|prepare|train|eval|recommend|full_pipeline`）。
3. **教程**：[`demos/movielens/TUTORIAL.md`](../demos/movielens/TUTORIAL.md)（架构、数据、Shell/MCP、调参剧本、验收）。
4. **案例数据**：MovieLens 100K ranking TSV + XGB NDCG baseline。

## Quick start

```powershell
go run ./demos/movielens/cmd/agent full-pipeline
go test ./demos/movielens/agentops -count=1
```

MCP 配置示例见教程 §6。

## Compatibility

- 库 API / CLI schema：**无破坏**
- MCP 为 **demo 可选适配层**（库核心仍不强制 MCP 运行时）
- Breaking: **无**

## Docs

- [TUTORIAL](../demos/movielens/TUTORIAL.md) · [SKILL](../skills/recsys-movielens-ranker/SKILL.md) · [CHANGELOG](../CHANGELOG.md)
