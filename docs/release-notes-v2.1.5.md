# leaves v2.1.5

**主题**：MovieLens 四段推荐流水线（prep → 召回 → LTR → 发牌）。

## Highlights

1. **真实数据四段**：`recsys/movielens` 纯 Go 加载 ml-100k → 四元 `User/Item/Score/Tag` + catalog 特征 + 片名旁车。
2. **共用串联**：`pipeline.RunFromDataset`（合成 smoke 与 MovieLens 同一路径）。
3. **入口齐全**：
   - `go run ./recsys/cmd/movielens`
   - `go run ./demos/movielens/cmd/agent four-stage`
   - MCP `movielens_four_stage`
4. **召回 / 发牌加固**：部分正样本 + 未交互补齐；`deal.fillOverflow` 可凑满 DeckSize。

## Quick start

```powershell
# Agent JSON
go run ./demos/movielens/cmd/agent four-stage
# 人类 CLI
go run ./recsys/cmd/movielens -workspace demos/movielens/out/fourstage
# 回归
go test ./recsys/pipeline/ -run TestMovieLensFourStage -count=1
```

首次需网络下载 ml-100k，或预置 `.cache/ml-100k.zip`。

## Compatibility

- 库 API / CLI schema：**无破坏**
- 精排-only（`full-pipeline` / ranking TSV）路径不变
- Breaking: **无**

## Docs

- [TUTORIAL §11.1](../demos/movielens/TUTORIAL.md) · [recsys-orchestrator](../skills/recsys-orchestrator/SKILL.md) · [CHANGELOG](../CHANGELOG.md)
