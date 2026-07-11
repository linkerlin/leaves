# leaves v2.1.4

**主题**：MovieLens 推荐结果带片名（meta 旁车）+ 一键 walkthrough。

## Highlights

1. **meta 旁车**：`testdata/rank_movielens_*_meta.jsonl`（`movie_id` / `title` / `user_id`），由 `testdata/gen_rank_movielens.py` 写出。
2. **可读推荐**：`agent recommend` / `cmd/recommend` / MCP `movielens_recommend` 输出片名与 movie_id。
3. **一键脚本**：[`demos/movielens/scripts/walkthrough.ps1`](../demos/movielens/scripts/walkthrough.ps1) · [`.sh`](../demos/movielens/scripts/walkthrough.sh)。
4. **教程同步**：[`TUTORIAL.md`](../demos/movielens/TUTORIAL.md) 示例 JSON 与故障表。

## Quick start

```powershell
.\demos\movielens\scripts\walkthrough.ps1
# 或仅推荐
go run ./demos/movielens/cmd/recommend -group 0 -topk 10
go run ./demos/movielens/cmd/agent recommend -group 0 -topk 5
```

## Compatibility

- 库 API / CLI schema：**无破坏**
- meta 文件缺失时仍可推荐（仅 row/score/label；stderr 提示生成旁车）
- Breaking: **无**

## Docs

- [TUTORIAL](../demos/movielens/TUTORIAL.md) · [CHANGELOG](../CHANGELOG.md) · [release-checklist](release-checklist.md)
