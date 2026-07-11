# MovieLens 100K 推荐 Ranker Demo

基于 [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) 的 **LTR 精排**示例，并支持 **Agent / MCP** 驱动全流程。

> **详尽教程**（架构、MCP 配置、调参剧本、验收）：[`TUTORIAL.md`](TUTORIAL.md)  
> **Agent SKILL**：[`agent-skill/SKILL.md`](agent-skill/SKILL.md)

## 两种用法

| 方式 | 入口 | 适合 |
|------|------|------|
| 人类 CLI | `cmd/train`、`cmd/recommend` | 本地试跑、看 Top-K |
| **Agent JSON CLI** | `cmd/agent` | shell Agent，stdout 仅 JSON |
| **MCP server** | `cmd/mcp` | Cursor 等 MCP 客户端 |

## 1. 准备数据

```powershell
# 多数情况仓库已带 TSV；缺失或需片名旁车时：
cd testdata && python gen_rank_movielens.py && cd ..
# 或
go run ./demos/movielens/cmd/agent prepare
```

| 文件 | 说明 |
|------|------|
| `testdata/rank_movielens_train.tsv` | 60 用户训练 |
| `testdata/rank_movielens_test.tsv` | 15 用户测试 |
| `rank_movielens_*_meta.jsonl` | **movie_id / title** 旁车（推荐展示用） |
| `rank_movielens_*_xgb_baseline.json` | XGB NDCG 基准 |

```powershell
# 脚本 walkthrough（Agent CLI 分步）
pwsh demos/movielens/scripts/walkthrough.ps1
```

## 2. Agent 一键闭环

```powershell
go run ./demos/movielens/cmd/agent full-pipeline
# 分步：status | prepare | train | eval | recommend
```

## 3. MCP 配置示例

```json
{
  "mcpServers": {
    "leaves-movielens": {
      "command": "go",
      "args": ["run", "./demos/movielens/cmd/mcp"],
      "cwd": "C:/GitHub/leaves"
    }
  }
}
```

工具：`movielens_status` / `prepare` / `train` / `eval` / `recommend` / `full_pipeline`。

## 4. 人类友好命令

```powershell
go run ./demos/movielens/cmd/train
go run ./demos/movielens/cmd/train -objective rank:pairwise
go run ./demos/movielens/cmd/recommend -group 3 -topk 10
go test ./train/... -run TestRankMovieLens -count=1
go test ./demos/movielens/agentops -count=1
```

## 5. 特征与语义

TSV：`qid \t label \t feat1..feat22`（流行度、均分、年份、19 类 one-hot）。  
`label` = 星级 1–5；推荐输出含 `movie_id`/`title`（需 meta 旁车）。详见教程 §4 / §8.3。

## 6. 与 recsys 四段流水线（MovieLens 真实数据）

| 路径 | 入口 | 说明 |
|------|------|------|
| **精排-only** | `agent full-pipeline` | 固定 ranking TSV → train → Top-K |
| **四段** | `agent four-stage` | prep→召回100→LTR→发牌（Tag 控重） |
| 合成 smoke | `go run ./recsys/cmd/smoke` | 人造数据验契约 |

```powershell
# MovieLens 四段（需网络下载或 .cache/ml-100k.zip）
go run ./demos/movielens/cmd/agent four-stage
go run ./recsys/cmd/movielens -workspace demos/movielens/out/fourstage

# 回归
go test ./recsys/pipeline/ -run TestMovieLensFourStage -count=1
```

产物：`demos/movielens/out/fourstage/{clean,catalog,recall,rank,models,deal,meta}`。  
MCP 工具：`movielens_four_stage`。见 `skills/recsys-orchestrator`。

## 目录

```text
cmd/agent/       Agent JSON CLI
cmd/mcp/         MCP stdio server
cmd/train/       人类训练
cmd/recommend/   人类推荐
agentops/        共用实现
agent-skill/     Agent SKILL
TUTORIAL.md      详尽教程
rankutil/        NDCG / baseline / 路径
out/             产物
```
