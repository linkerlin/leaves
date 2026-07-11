# 教程：Agent 驱动的 MovieLens 推荐 Ranker 全流程

> **读者**：希望用 Agent（Cursor / Claude / 自研 Agent）驱动 leaves 完成「数据 → 训练精排 → 评估 → Top-K 推荐」的工程师。  
> **案例数据**：[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)  
> **库版本**：leaves ≥ v2.1.2  
> **定位**：精排 **ranker**（LTR）；完整「召回+发牌」合成流水线见 `go run ./recsys/cmd/smoke`。

---

## 目录

1. [你将得到什么](#1-你将得到什么)  
2. [架构与角色分工](#2-架构与角色分工)  
3. [环境准备](#3-环境准备)  
4. [数据说明（MovieLens → ranking TSV）](#4-数据说明movielens--ranking-tsv)  
5. [路径 A：纯 Shell Agent（推荐上手）](#5-路径-a纯-shell-agent推荐上手)  
6. [路径 B：MCP Server 驱动 Agent](#6-路径-bmcp-server-驱动-agent)  
7. [Agent 决策剧本与调参](#7-agent-决策剧本与调参)  
8. [产物与 metrics 契约](#8-产物与-metrics-契约)  
9. [与四段 recsys / leaves-autotrain 的关系](#9-与四段-recsys--leaves-autotrain-的关系)  
10. [验收清单与排障](#10-验收清单与排障)  
11. [扩展练习](#11-扩展练习)  
12. [附录：目录与源码索引](#12-附录目录与源码索引)  

---

## 1. 你将得到什么

跑完本教程后，你将具备：

| 能力 | 说明 |
|------|------|
| **真实数据精排** | 在 MovieLens 用户–电影评分上训练 `rank:ndcg`（可选 pairwise/listwise） |
| **Agent 闭环** | 用 JSON 契约驱动：准备 → 训练 → 评估 → 推荐，无需手写 Go |
| **双通道调用** | Shell CLI **或** MCP `tools/call`（同一套 `agentops`） |
| **可对比 baseline** | 与 XGBoost NDCG 基准对齐（`testdata/rank_movielens_*_xgb_baseline.json`） |
| **可审计产物** | `metrics_*.json`、`runs.jsonl`、`recommend_g*.json`（含 **movie_id/title**）、`leaves.json` 模型 |

**非目标（本教程不做）**：

- 官方云端 model registry / 在线 serving 平台（见 `examples/serving-template`）  
- 把搜索写进 leaves 内核（搜索逻辑在 Agent SKILL）  
- 实时协同过滤 / 向量召回服务  

---

## 2. 架构与角色分工

```text
┌─────────────────────────────────────────────────────────────┐
│  Agent（优化器）                                              │
│  · 读 SKILL：demos/movielens/agent-skill/SKILL.md            │
│  · 调 Shell 或 MCP tools                                     │
│  · 读 metrics.json / runs.jsonl 决定下一组超参                │
└───────────────┬─────────────────────────────┬───────────────┘
                │ shell JSON                  │ MCP JSON-RPC
                ▼                             ▼
     cmd/agent (CLI)                   cmd/mcp (stdio)
                │                             │
                └──────────┬──────────────────┘
                           ▼
                    agentops（业务真源）
                           │
         prepare │ train │ eval │ recommend │ full_pipeline
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   gen_rank_movielens  train.Learner    RankGroup Top-K
   (MovieLens 100K)    rank:ndcg        leaves.json
```

| 角色 | 职责 | 不负责 |
|------|------|--------|
| **Agent** | 决策下一组超参、何时收敛、如何报告 | 实现 GBRT |
| **agentops / CLI / MCP** | 执行、写 JSON 信号、固定退出语义 | 贝叶斯搜索 |
| **leaves 库** | 训练/推理/NDCG | 推荐业务策略 |

这与项目「**Agent 即优化器，leaves 即目标函数**」一致（见 `演进方案.md`）。

---

## 3. 环境准备

### 3.1 必备

- Go **1.26+**（与根 `go.mod` 一致）  
- 仓库克隆可编译：`go test ./demos/movielens/agentops -count=1`  
- 可选：Python 3 + `numpy` + `xgboost`（**仅**在需要重新下载/生成 MovieLens TSV 时）  

### 3.2 仓库根目录

后续命令均默认在 **仓库根**（含 `go.mod`）执行。

```powershell
cd C:\GitHub\leaves   # 按你的路径修改
```

### 3.3 检查数据是否已存在

本仓库通常已附带预生成的 ranking TSV（无需联网）：

```powershell
go run ./demos/movielens/cmd/agent status
```

示例成功输出（节选）：

```json
{
  "ok": true,
  "op": "status",
  "data": {
    "train_ready": true,
    "test_ready": true,
    "model_ready": false
  }
}
```

若 `train_ready=false`，见下一节 **prepare**。

---

## 4. 数据说明（MovieLens → ranking TSV）

### 4.1 语义

| 概念 | MovieLens | ranking TSV |
|------|-----------|-------------|
| Query | 一个用户 | `qid` |
| Document | 该用户评过的一部电影 | 一行 |
| 相关性 | 星级 1–5 | `label` |
| 特征 | 流行度、均分、年份、类型 one-hot 等 | `feat1..feat22` |

行格式：

```text
qid \t label \t f1 \t f2 \t ... \t f22
```

- 训练集约 **60** 用户，测试集约 **15** 用户（生成脚本可改）  
- 特征细节见 [`README.md`](README.md) §特征说明  

### 4.2 重新生成数据

需要网络下载 ml-100k.zip（或本地 zip）：

```powershell
cd testdata
python gen_rank_movielens.py
# 或
python gen_rank_movielens.py --force
cd ..
```

Agent 等价调用：

```powershell
go run ./demos/movielens/cmd/agent prepare
# force:
go run ./demos/movielens/cmd/agent prepare -force
```

**依赖失败时**：Agent 应读取 JSON 的 `hint` 字段，提示安装 `numpy`/`xgboost`，而不是盲目重试。

### 4.3 为何这是「推荐 ranker」

真实推荐系统常见四段：准备 → **召回** → **精排** → 发牌。  
本案例把「用户历史评分列表」当作候选集，直接做 **listwise LTR 精排**——这是 ranker 能力的标准验真路径（与 XGB `rank:ndcg` 对标）。  
若你要「先召回 100 再精排再发牌」的合成全链路，请并行阅读 §9。

---

## 5. 路径 A：纯 Shell Agent（推荐上手）

适合：Cursor / Claude Code / 自研 Agent **只会跑 shell** 的场景。  
**无需**配置 MCP。

### 5.1 一键全流程

```powershell
go run ./demos/movielens/cmd/agent full-pipeline
```

或仓库脚本：

```powershell
pwsh demos/movielens/scripts/walkthrough.ps1
# bash demos/movielens/scripts/walkthrough.sh
```

内部顺序：`prepare`（若需要）→ `train` → `eval` → `recommend`（默认 group=0, topk=10）。

### 5.2 分步（便于调参循环）

```powershell
# ① 就绪检查
go run ./demos/movielens/cmd/agent status | ConvertFrom-Json | Format-List

# ② 训练（主优化目标 = test NDCG，maximize=true）
go run ./demos/movielens/cmd/agent train -objective rank:ndcg -rounds 40 -depth 4 -lr 0.1

# ③ 独立评估
go run ./demos/movielens/cmd/agent eval

# ④ 看一个用户的 Top-10
go run ./demos/movielens/cmd/agent recommend -group 0 -topk 10
```

### 5.3 Agent 解析规则（重要）

| 规则 | 说明 |
|------|------|
| **只信 stdout JSON** | `ok`、`data`、`error`、`hint` |
| **退出码** | `0` 成功；非 0 失败（仍可能打印 JSON） |
| **stderr** | 给人看的日志；Agent 可忽略 |
| **主指标** | `data.test_ndcg` 或 metrics 文件里 `value` + `maximize=true` |

伪代码：

```text
r = shell("go run ./demos/movielens/cmd/agent train ...")
doc = json.parse(r.stdout)
if not doc.ok: fix(doc.hint); stop
if doc.data.test_ndcg < target: adjust hyperparams; goto train
```

### 5.4 人类可读对照命令

与旧 demo 等价（非 JSON）：

```powershell
go run ./demos/movielens/cmd/train
go run ./demos/movielens/cmd/recommend -group 0 -topk 10
go test ./train/... -run TestRankMovieLens -count=1
```

---

## 6. 路径 B：MCP Server 驱动 Agent

适合：Cursor / Claude Desktop / 支持 **MCP tools** 的客户端。

### 6.1 启动方式

服务通过 **stdio** 对话：每行一个 JSON-RPC 请求/响应；**日志只写 stderr**。

配置示例（路径改成你的仓库根）：

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

也可先编译：

```powershell
go build -o bin/leaves-movielens-mcp.exe ./demos/movielens/cmd/mcp
# command 改为绝对路径 bin/leaves-movielens-mcp.exe
```

### 6.2 协议要点（子集）

本 server 实现 MCP 常用子集，便于教学与对接：

| Method | 作用 |
|--------|------|
| `initialize` | 握手，返回 `protocolVersion` / `serverInfo` |
| `notifications/initialized` | 可忽略 |
| `tools/list` | 列出工具与 JSON Schema |
| `tools/call` | 执行工具；结果在 `content[0].text`（JSON 字符串） |
| `ping` | 探活 |

工具名与 CLI 对应：

| MCP tool | CLI |
|----------|-----|
| `movielens_status` | `agent status` |
| `movielens_prepare` | `agent prepare` |
| `movielens_train` | `agent train` |
| `movielens_eval` | `agent eval` |
| `movielens_recommend` | `agent recommend` |
| `movielens_full_pipeline` | `agent full-pipeline` |

### 6.3 手工探测（调试用）

PowerShell 向进程喂一行 JSON（教学演示；真实客户端自动握手）：

```powershell
$proc = Start-Process -FilePath go -ArgumentList "run","./demos/movielens/cmd/mcp" `
  -NoNewWindow -PassThru -RedirectStandardInput pipe -RedirectStandardOutput pipe
# 更实用：用支持 MCP 的 IDE 连接；或 printf 管道
```

```text
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"demo","version":"0"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"movielens_status","arguments":{}}}
```

### 6.4 Agent 提示词模板（可直接粘贴）

```text
你是推荐精排 Agent。仓库根目录为 leaves。
优先通过 MCP server「leaves-movielens」调用 tools。
若 MCP 不可用，则 shell：go run ./demos/movielens/cmd/agent <cmd>。
流程：status →（必要时）prepare → train → 读 test_ndcg → 按需调参再 train → eval → recommend。
主指标 maximize NDCG@10；收敛后汇总模型路径、NDCG、一个用户的 Top-10。
不要修改 leaves 库源码；不要伪造指标。
```

### 6.5 与项目「不强制 MCP」哲学的关系

- **leaves 库与通用 CLI** 不依赖 MCP 运行时（见 `Agents.md` / 演进方案）。  
- **本 demo** 额外提供 **可选 MCP 适配层**，把同一套 `agentops` 暴露给会调 MCP 的 Agent。  
- 生产若不用 MCP，只保留 Shell 路径即可，行为一致。

---

## 7. Agent 决策剧本与调参

### 7.1 默认剧本

```text
1. status
2. if !train_ready → prepare
3. train (rank:ndcg, rounds=40, depth=4, lr=0.1)  # 与 XGB baseline 对齐
4. 记录 test_ndcg 与 xgb_baseline.delta_test（若有）
5. 若未达标：
     - 欠拟合（train≈test 且偏低）：↑depth 或 ↑rounds，略 ↑lr
     - 过拟合（train>>test）：↓depth、↑lambda、略 ↓rounds
6. eval 确认
7. recommend group=0,3,7 抽查：高 label 是否偏前
8. 输出报告
```

### 7.2 与 leaves-autotrain 的衔接

若 Agent 已加载 `leaves-autotrain`：

- 可用通用 `leaves train --data testdata/rank_movielens_train.tsv --objective rank:ndcg ...`  
- MovieLens 专用工具已写好路径/baseline/推荐 JSON，**优先本 demo 工具**，少拼 flag 错误。  
- `runs.jsonl` 在 `demos/movielens/out/`，语义与 autotrain 账本类似。

### 7.3 收敛建议

- 连续 2–3 轮 test NDCG 提升 &lt; 0.5% → 停止  
- 小用户数 demo **不要**指望生产 SOTA；关注「流程可复现 + 指标诚实」  

---

## 8. 产物与 metrics 契约

### 8.1 目录

```text
testdata/
  rank_movielens_train.tsv
  rank_movielens_test.tsv
  rank_movielens_ndcg_xgb_baseline.json
demos/movielens/out/
  model_rank_ndcg.leaves.json
  metrics_train.json
  metrics_eval.json
  runs.jsonl
  recommend_g0.json
  pipeline_report.json          # full-pipeline 时
```

### 8.2 `metrics_train.json` 字段（Agent）

| 字段 | 含义 |
|------|------|
| `value` | **测试集** NDCG@k（主优化目标） |
| `maximize` | `true` |
| `train_metric` | 训练集 NDCG |
| `params` | 完整超参 |
| `model` | 模型路径 |
| `xgb_baseline` | 可选，对标 Δ |

### 8.3 推荐 JSON

```json
{
  "qid": 60,
  "group": 0,
  "items": [
    {
      "rank": 1,
      "score": 1.23,
      "label": 5,
      "row": 12,
      "movie_id": 100,
      "title": "Fargo (1996)",
      "user_id": 699
    }
  ],
  "note": "label=历史星级；含 movie_id/title（旁车 meta）"
}
```

旁车文件（由 `gen_rank_movielens.py` 写出）：

- `testdata/rank_movielens_test_meta.jsonl`
- `testdata/rank_movielens_train_meta.jsonl`

若仅有 TSV 无 meta，执行：

```powershell
cd testdata; python gen_rank_movielens.py; cd ..
```

（TSV/baseline 已存在时会**只刷新 meta**，不必重训 XGB。）

解读：`label` 高且 `rank` 靠前 → 排序在用户真实喜好上合理（离线回放，不是在线 CTR）。

---

## 9. 与四段 recsys / leaves-autotrain 的关系

```text
┌──────────────────────────────────────────┐
│  recsys-orchestrator（四段）               │
│  prep → recall(100) → LTR → deal         │
│  合成数据：go run ./recsys/cmd/smoke       │
└──────────────────────────────────────────┘
                    ▲
                    │ 精排阶段可换成 MovieLens ranker 思路
┌──────────────────────────────────────────┐
│  本教程 MovieLens ranker                   │
│  历史评分作候选 + rank:ndcg 精排 + Top-K   │
└──────────────────────────────────────────┘
                    ▲
                    │ 通用调参 SKILL
┌──────────────────────────────────────────┐
│  leaves-autotrain                          │
│  sniff/train/eval/publish + metrics 闭环   │
└──────────────────────────────────────────┘
```

| 场景 | 用什么 |
|------|--------|
| 学精排 / 对标 XGB / Agent MCP | **本教程** |
| 学召回+发牌全链路（合成） | `recsys/cmd/smoke` + `recsys-orchestrator` |
| 任意表数据 AutoML 调参 | `leaves-autotrain` |

---

## 10. 验收清单与排障

### 10.1 验收 DoD

- [ ] `agent status` → `train_ready=true`  
- [ ] `agent train` → `ok=true`，写出 `model_rank_ndcg.leaves.json`  
- [ ] `data.test_ndcg` 有限且合理（demo 量级通常 &gt; 0.5，视数据而定）  
- [ ] `agent recommend` → `items` 长度 = topk  
- [ ] `go test ./demos/movielens/agentops -count=1` 通过  
- [ ] （可选）MCP `tools/list` 可见 6 个 `movielens_*` 工具  

### 10.2 常见问题

| 现象 | 处理 |
|------|------|
| `testdata not found` | 在仓库根运行；或设 `LEAVES_TESTDATA` |
| prepare 失败 | 装 Python 依赖；检查网络/本地 zip |
| train 找不到 TSV | 先 `prepare` |
| MCP 无输出 | 确认 stdout 未被日志污染；只用 stderr 打日志 |
| 推荐 label 全是 0 | 检查是否用了错误测试文件 / 组号越界 |
| 推荐无 title | `python testdata/gen_rank_movielens.py` 生成 `*_meta.jsonl` |

---

## 11. 扩展练习

1. **调参挑战**：在不改代码的前提下，用 Agent 循环把 test NDCG 抬升，并保留 `runs.jsonl` 轨迹。  
2. **目标对比**：`rank:pairwise` / `rank:listwise` 与 `rank:ndcg` 的 test NDCG 对比表。  
3. **接入 serving-template**：把 `out/model_rank_ndcg.leaves.json` 挂到 `examples/serving-template`，用 HTTP 暴露批预测（注意 ranking 推理仍按组特征矩阵输入）。  
4. **四段融合（进阶）**：用 MovieLens 交互生成 `User/Item/Score/Tag`，跑 `recsys` 召回+发牌，精排换成 leaves ranker。  

---

## 12. 附录：目录与源码索引

```text
demos/movielens/
  TUTORIAL.md              ← 本文
  README.md                ← 人类速览
  agent-skill/SKILL.md     ← Agent 短 SKILL
  agentops/                ← 业务真源（CLI/MCP 共用）
  cmd/agent/               ← Shell Agent CLI（stdout JSON）
  cmd/mcp/                 ← MCP stdio server
  cmd/train/               ← 人类友好训练
  cmd/recommend/           ← 人类友好推荐
  rankutil/                ← NDCG / baseline / 路径
  out/                     ← 运行产物（可 gitignore）
testdata/
  gen_rank_movielens.py
  rank_movielens_*.tsv
  rank_movielens_*_baseline.json
```

### 相关文档

- [`skills/recsys-orchestrator/SKILL.md`](../../skills/recsys-orchestrator/SKILL.md)  
- [`skills/leaves-autotrain/SKILL.md`](../../skills/leaves-autotrain/SKILL.md)  
- [`skills/recsys-rank/SKILL.md`](../../skills/recsys-rank/SKILL.md)  
- [`examples/serving-template/README.md`](../../examples/serving-template/README.md)  
- [`演进方案.md`](../../演进方案.md)（Agent 契约哲学）  

---

## 快速命令卡片

```powershell
# Shell Agent
go run ./demos/movielens/cmd/agent full-pipeline

# 测试
go test ./demos/movielens/agentops -count=1
go test ./train/... -run TestRankMovieLens -count=1

# MCP（配置到客户端后）
# tools/call movielens_full_pipeline {"objective":"rank:ndcg","group":0,"topk":10}
```

**祝调试顺利。** 指标说话，JSON 说话，Agent 只做决策。
