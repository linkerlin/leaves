---
name: recsys-orchestrator
description: >-
  编排推荐系统全流程：数据准备→召回(100/User)→LTR排序→发牌，映射 leaves 能力与四元数据契约。
  凡涉端到端推荐流水线、recsys 工作流、多阶段串联时用之。
---

# 推荐流水线总控

> 四段流水线，一环扣一环；数据备、召回出、排序训、发牌定。

## 一、何时激活

- 用户要求「搭建推荐系统」「端到端流水线」「recsys 工作流」
- 需统筹多阶段 SKILL，或不确定从哪一阶段入手
- 本 SKILL 为**入口**，各阶段细则分见子 SKILL

## 二、流水线总览

```mermaid
flowchart LR
  A[原始日志] --> B[数据准备]
  B --> C[召回 100/User]
  C --> D[转 ranking TSV]
  D --> E[LTR 排序]
  E --> F[发牌]
  F --> G[deal 终稿]
```

| 阶段 | SKILL | 核心产物 |
|------|-------|----------|
| 0 契约 | `recsys-data-model` | 四元定义、目录规范 |
| 1 准备 | `recsys-data-prep` | `clean/samples_*.tsv`, `catalog/items.tsv` |
| 2 召回 | `recsys-recall` | `recall/recall_*.tsv`（100 Item/User） |
| 3 排序 | `recsys-rank` | `models/*.leaves.json`, `rank_*_scored.jsonl` |
| 4 发牌 | `recsys-deal` | `deal/deal_*.tsv` |

## 三、执行清单

复制跟踪：

```
推荐流水线 Progress:
- [ ] Stage 0: 确认四元契约（User/Item/Score/Tag）
- [ ] Stage 1: 清洗 → samples + catalog + user_qid
- [ ] Stage 2: 召回 → 每 User 100 Item
- [ ] Stage 2b: 转 rank_*.tsv + manifest
- [ ] Stage 3: leaves 训练 rank:ndcg + NDCG 评估
- [ ] Stage 3b: 组内推理 margin → scored.jsonl
- [ ] Stage 4: 发牌（去重 + Tag 控重 + Top-K）
- [ ] 验收: 校验脚本 + demo 对标
```

## 四、快速启动（仓库内 demo）

### 4.1 Go 端到端 smoke（100 Item/User，推荐）

```powershell
# 一键：合成数据 → 准备 → 召回 → 排序 → 发牌
go run ./recsys/cmd/smoke

# 自定义工作区
go run ./recsys/cmd/smoke -workspace recsys/out/smoke -recall-size 100

# 集成测试
go test ./recsys/pipeline/... -run TestSmokePipeline100PerUser -count=1
```

模块布局：

```
recsys/
  synth/      合成原始交互
  prep/       清洗、切分、catalog
  recall/     每 User 100 Item
  rankconv/   转 ranking TSV + manifest
  trainrank/  leaves rank:ndcg 训练与打分
  deal/       发牌（去重 + Tag 控重）
  pipeline/   串联各阶段
  cmd/smoke/  CLI 入口
```

### 4.2 MovieLens Ranker + Agent/MCP（推荐精排案例）

```powershell
# 一键 Agent 闭环（stdout JSON）— 精排-only
go run ./demos/movielens/cmd/agent full-pipeline

# 人类 CLI
go run ./demos/movielens/cmd/train
go run ./demos/movielens/cmd/recommend -group 0 -topk 10

# MCP server（配置到客户端 cwd=仓库根）
go run ./demos/movielens/cmd/mcp
```

- **详尽教程**：[`demos/movielens/TUTORIAL.md`](../../demos/movielens/TUTORIAL.md)  
- **Agent SKILL**：[`demos/movielens/agent-skill/SKILL.md`](../../demos/movielens/agent-skill/SKILL.md)  
- 回归：`go test ./train/... -run TestRankMovieLens` · `go test ./demos/movielens/agentops`

### 4.2b MovieLens 四段流水线（真实数据 prep→召回→LTR→发牌）

```powershell
# Agent JSON
go run ./demos/movielens/cmd/agent four-stage
# 人类 CLI
go run ./recsys/cmd/movielens -workspace demos/movielens/out/fourstage
# 回归
go test ./recsys/pipeline/... -run TestMovieLensFourStage -count=1
```

- 加载：`recsys/movielens`（ml-100k → 四元 Dataset）  
- 串联：`pipeline.RunFromDataset`  
- MCP：`movielens_four_stage`  
- 教程：[`TUTORIAL.md` §11.1](../../demos/movielens/TUTORIAL.md)

### 4.3 Smoke 最小排序对标

```powershell
cd testdata && python gen_rank_smoke.py && cd ..
go test ./train/... -run 'TestRank.*TrendVsXGBoost' -count=1
```

完整四段流水线（合成数据）：`go run ./recsys/cmd/smoke`（见 §4.1）。

## 五、子 SKILL 激活顺序

```
1. recsys-data-model   ← 先读契约
2. recsys-data-prep
3. recsys-recall
4. recsys-rank
5. recsys-deal
```

遇问题时：

- 数据格式 → `recsys-data-model`
- leaves API → `recsys-rank` / `leaves-api.md`
- 指标不达标 → 查 NDCG@k、增轮数、调 depth/lr
- 发牌多样性 → `recsys-deal`

## 六、leaves 能力边界

| 覆盖 | 不覆盖 |
|------|--------|
| LTR 训练 rank:ndcg/pairwise/listwise | 召回算法（库外实现） |
| ranking TSV 加载与嗅探 | 实时特征 / 在线学习 |
| leaves.json 训练保存与推理 | 发牌策略（库外实现） |
| NDCG/MAP 评估 | 分布式训练 / serving 框架 |
| XGB/LGB 模型加载、XGB JSON 导出 | 协同过滤 / embedding 召回 |
| Born CPU/WebGPU 训练加速 | 官方 registry / 在线 serving / 实时学习 |
| WASM/HTTP embed demo | 推广执行（只产出请求，见 §十） |
| 离线四段 + 控制面契约（§十） | 曝光/点击的真实采集与推送 |

**定位**：leaves 为**精排器**；召回与发牌由本 SKILL 体系在库外补全。

## 七、实现语言

一次性数据生成可 Python（参照 `testdata/gen_rank_*.py`）；**端到端 smoke 已实现为 Go**：`go run ./recsys/cmd/smoke`。

## 八、验收标准

| 检查项 | 期望 |
|--------|------|
| 数据 | 仅 User/Item/Score/Tag 四元 |
| 召回 | 每 User 100 Item |
| 排序 TSV | `data.FromFileAuto` → FormatRanking |
| 训练 | test NDCG@10 可报告 |
| 推理 | 每候选有 margin |
| 发牌 | 无 recent 重复、Tag ≤ 3（默认） |
| 回归 | `go test ./train/... -run Rank -count=1` 通过 |

## 九、目录脚手架

首次实施时创建：

```powershell
mkdir -p recsys/{raw,clean,catalog,recall,rank,models,deal,meta}
```

## 十、生产闭环控制面（八段剧本，2026-08 起）

> 四段流水线（§二）产出的是**离线 deal 终稿**，不等于线上推荐决策。
> 生产闭环需要事件、归因、监控与受控发布；本节是 Agent 的八段剧本。
> 详见 [`docs/recsys-loop.md`](../../docs/recsys-loop.md) 与演进方案 §十七。

**关键区分**（Agent 汇报时必须使用正确口径）：

| 状态 | 含义 | 判据 |
|------|------|------|
| 离线提升 | evaluation.json 指标变好 | 仅离线指标，不得宣称线上收益 |
| 可推广候选 | release 状态机到 `candidate/approved` | 三层门禁无 block + 人工批准 |
| 已上线观察 | `promoted/observing` | 外部 adapter 确认；monitor 窗口 ok |

**八段剧本**（Agent 只读结构化 JSON，不解析 stderr，不隐式选最优）：

```text
1. snapshot + 时间切分    contract.DatasetSnapshot + split.Split/CheckLeakage
2. recall/rank/deal/eval  pipeline + eval.Evaluate（三层阈值门禁 → evaluation.json）
3. candidate evidence     release.ToCandidate（三层门禁齐全 + 模型 hash 一致）
4. decision/exposure/feedback ledger（decision 是审计真源；deal 行仅展示）
5. monitor 健康窗口       monitor.BuildReport → ok → 保持 observing
6. 退化注入/真实退化      monitor → block（ok/warn/block + reason code）
7. 触发器 → rollback_requested  monitor.TriggerSet（连续越界 + 冷却期）→ release.RequestRollback 指向 last_known_good
8. replay → retrain       replay.BuildSamples 只消费可归因反馈 → 下一轮快照
```

**结构化信号**（全部 JSON/JSONL，退出码语义同 leaves CLI）：
`evaluation.json`（eval）· `ledger.jsonl`（决策/曝光/反馈）· `monitor_report.json` ·
`replay_report.json` · `release_evidence.json` + `run_status.jsonl`（release）。

**shell 入口**（`recsys/cmd/control`，零 Go 代码编排八段剧本）：

```text
control snapshot -workspace DIR -out snapshot.json -snapshot-id S -purpose release
control split -events events.jsonl -train-end T -val-start T -test-start T -out-dir split/
control eval -workspace DIR -thresholds th.json -out evaluation.json
control from-deal -workspace DIR -ledger ledger.jsonl -model-version V -policy-version P -occurred-at T
control append-exposure -ledger ledger.jsonl -in exposures.jsonl
control append-feedback -ledger ledger.jsonl -in feedback.jsonl
control replay -ledger ledger.jsonl -out samples.jsonl -report replay_report.json
control monitor -ledger ledger.jsonl -workspace DIR -window-start T -window-end T -thresholds th.json -triggers tr.json -fired fired.jsonl
control release -state release_state.json -action candidate|approve|confirm-promote|observe|retrain|rollback|retire|status
```

退出码 0/1/2（1=用法/IO，2=校验/内部）；promote/rollback 请求打印到 stdout，
Agent 把它交给应用侧 adapter（leaves 不执行网络副作用）。回滚决策来自
`monitor -triggers` 的 `fired.jsonl`（配置驱动），不是 Agent 临场猜测。

**边界**：leaves 只产出 adapter-neutral 的推广/回滚**请求**（`release.Adapter`
接口 + fake）；真实 registry/serving/CI 由应用仓库实现。初始版本人工批准默认开启。

**回归**：`go test ./recsys/... -count=1`（含 `recsys/loop` 八段演练与
`recsys/cmd/control` CLI 端到端）。

## 附：数据流示意

```
raw.log
  └─prep─→ clean/samples_train.tsv (User,Item,Score,Tag)
           catalog/items.tsv (Item,Tag,feat_*)
           meta/user_qid.tsv
  └─recall─→ recall/recall_train.tsv (100 rows/User)
           └─convert─→ rank/rank_train.tsv (qid,label,feat_*)
                       rank/rank_train_manifest.jsonl
  └─rank─→ models/model_rank_ndcg.leaves.json
           rank/rank_test_scored.jsonl (+Score margin)
  └─deal─→ deal/deal_test.tsv (User,Item,Tag,Score,rank)
```

详表见 [`workflow.md`](workflow.md) 与 `recsys-data-model/formats.md`。
