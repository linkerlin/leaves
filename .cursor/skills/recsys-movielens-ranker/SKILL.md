---
name: recsys-movielens-ranker
description: >-
  以 MovieLens 100K 为案例，用 Agent 驱动 leaves LTR ranker 全流程
  （准备数据→训练 rank:ndcg→评估 NDCG→Top-K 推荐）。
  支持 shell CLI 与 MCP tools/call。凡涉 MovieLens 推荐排序 demo、Agent MCP 教程时用之。
---

# MovieLens Ranker Agent Skill

> Agent 即优化器；本 SKILL 描述如何调用 **CLI** 或 **MCP** 完成精排 demo。  
> 上位流水线：`recsys-orchestrator`；leaves 通用调参：`leaves-autotrain`。

## 一、何时激活

- 用户要求「MovieLens 推荐」「ranker demo」「Agent 调 MCP 跑排序」
- 需要 NDCG 对标 / Top-K 推荐列表 / 教程复现

## 二、工具入口（二选一）

### A. Shell CLI（无 MCP 客户端时）

在仓库根目录：

```powershell
go run ./demos/movielens/cmd/agent status
go run ./demos/movielens/cmd/agent prepare
go run ./demos/movielens/cmd/agent train -objective rank:ndcg
go run ./demos/movielens/cmd/agent eval
go run ./demos/movielens/cmd/agent recommend -group 0 -topk 10
# 或一键
go run ./demos/movielens/cmd/agent full-pipeline
```

**契约**：stdout **仅一条 JSON**（`ok` / `error` / `data`）。Agent 只解析 JSON，勿依赖 stderr 散文。

### B. MCP server

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

| Tool | 作用 |
|------|------|
| `movielens_status` | 数据/模型就绪 |
| `movielens_prepare` | 生成 TSV（可 `force`） |
| `movielens_train` | 训练 ranker |
| `movielens_eval` | 测试 NDCG |
| `movielens_recommend` | Top-K |
| `movielens_full_pipeline` | 一键闭环 |

## 三、推荐 Agent 剧本

```
1. movielens_status → 若 train_ready=false → movielens_prepare
2. movielens_train  objective=rank:ndcg
3. 读 data.test_ndcg；若 < 目标 → 调 rounds/depth/lr 再 train（写不同 tag 到 runs.jsonl）
4. movielens_eval 确认 holdout
5. movielens_recommend group=0..k 抽样检查 label 是否高星偏前
6. 收敛 → 报告 model 路径 + NDCG + 样例 Top-K
```

调参原则（与 leaves-autotrain 一致）：

- 指标 `maximize=true`，主字段 `value` = test NDCG
- 小数据勿 subsample；优先 depth / lr / rounds
- 勿与 multi: 目标混用

## 四、数据与产物

| 路径 | 说明 |
|------|------|
| `testdata/rank_movielens_{train,test}.tsv` | ranking 输入 |
| `demos/movielens/out/model_rank_*.leaves.json` | 模型 |
| `demos/movielens/out/metrics_*.json` | Agent 信号 |
| `demos/movielens/out/runs.jsonl` | 调参账本 |
| `demos/movielens/out/recommend_g*.json` | Top-K |

## 五、与四段 recsys 的关系

本 demo 聚焦 **精排 ranker**（LTR）。  
端到端「准备→召回→排序→发牌」合成数据见 `go run ./recsys/cmd/smoke`。  
MovieLens 在本仓库的定位：真实数据上验证 **rank:ndcg** 与 XGB baseline。

## 六、详文

见 [`../../demos/movielens/TUTORIAL.md`](../../demos/movielens/TUTORIAL.md)。
