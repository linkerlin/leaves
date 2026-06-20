---
name: recsys-data-model
description: >-
  定义推荐系统四元数据契约（User/Item/Score/Tag），规范各阶段文件格式与 qid 映射。
  凡涉推荐数据建模、字段约定、TSV/JSONL 产物命名时用之。
---

# 推荐数据契约

> 以四元为纲，以文件为体；User 为 query，Item 为文档，Score 为浮点，Tag 为品类。

## 一、元数据类型（唯此四者）

| 字段 | 类型 | 语义 |
|------|------|------|
| `User` | String | 用户标识，全局唯一 |
| `Item` | String | 物品标识，全局唯一 |
| `Score` | float64 | 相关性/偏好强度；训练作 label，推理作 margin |
| `Tag` | String | 品类/类目；发牌时控重复 |

**约束**：除 `Score` 外皆为 String；不得引入第五种业务字段于核心表。扩展特征须编码为数值列，附于排序 TSV 之 feat 段。

## 二、工作区目录

```
recsys/
  raw/              # 原始日志（只读）
  clean/            # 清洗后样本
  catalog/          # 物品目录与特征
  recall/           # 召回产物（每用户 100 Item）
  rank/             # 排序 TSV + manifest
  models/           # leaves.json / xgb baseline
  deal/             # 发牌终稿
  meta/             # 映射表、统计、校验报告
```

## 三、阶段文件格式

### 3.1 清洗样本 `clean/samples_{split}.tsv`

交互日志，一行一曝光/点击/评分：

```
User	Item	Score	Tag
u001	i42	4.5	drama
u001	i17	0.0	comedy
```

- `split`：`train` | `test` | `all`
- `Score=0` 可表曝光未点击；正分表正向反馈
- 同 `(User,Item)` 去重保留最新一行

### 3.2 物品目录 `catalog/items.tsv`

```
Item	Tag	feat_pop	feat_quality	feat_age	...
i42	drama	2.3	0.81	0.15
```

- 首两列固定 `Item`、`Tag`（String）
- `feat_*` 为 float，供召回特征与排序特征共用

### 3.3 用户映射 `meta/user_qid.tsv`

```
User	qid	split
u001	0	train
u002	1	train
u101	60	test
```

- `qid` 为整型，**train 从 0 起连续，test 接 train 末尾**
- 排序 TSV 之行序须与 qid 分组一致（见 leaves `LoadRankingTSV`）

### 3.4 召回文件 `recall/recall_{split}.tsv`

每用户 **恰好 100 行** Item（不足则告警，超出则截断）：

```
User	Item	Tag	recall_score	feat_pop	feat_quality	...
u001	i42	drama	0.87	2.3	0.81
```

- `recall_score`：召回通道分（float），非 LTR margin
- 特征列须与 `catalog/items.tsv` 对齐
- 校验：`COUNT(Item) BY User == 100`

### 3.5 排序 TSV `rank/rank_{split}.tsv`

leaves / XGBoost LTR 共用格式（与 `data.LoadRankingTSV` 一致）：

```
# qid label feat1 feat2 ...
0	4.5	2.3	0.81	0.15
0	0.0	1.1	0.62	0.22
1	3.0	...
```

- `qid`：取自 `user_qid.tsv`，**同 qid 行须相邻、qid 单调非降**
- `label`：训练集取 `samples` 中 `(User,Item)` 之 Score；推理/无标签召回行填 `0`
- 特征列顺序与 `catalog` 中 `feat_*` 一致

### 3.6 行映射 `rank/rank_{split}_manifest.jsonl`

TSV 每行对应一条 JSON（行序与 TSV 数据行一致）：

```json
{"User":"u001","Item":"i42","Tag":"drama","recall_score":0.87}
```

推理后将 margin 写回 manifest 或另存 `rank_{split}_scored.jsonl`（增 `Score` 字段）。

### 3.7 发牌终稿 `deal/deal_{split}.tsv`

```
User	Item	Tag	Score	rank
u001	i42	drama	1.23	1
```

- `Score` 为 LTR margin（降序）
- `rank` 为发牌后最终序位（从 1 起）

## 四、qid 与 User 互转

```go
// qid → User：查 meta/user_qid.tsv
// User → qid：同上反向索引
// 测试集 groupIdx：qid - trainUserCount（见 demos/movielens/cmd/recommend）
```

## 五、校验清单

实施各阶段后，须验：

- [ ] 核心表仅含 User/Item/Score/Tag 四元（String 三 + float 一）
- [ ] `user_qid` 无重复 User、无 qid 空洞
- [ ] 召回：每 User 100 Item，Item 组内无重复
- [ ] 排序 TSV：qid 连续分组、特征列数恒定
- [ ] manifest 行数 == 排序 TSV 数据行数

## 附：与 leaves 之对应

| 推荐概念 | leaves 概念 |
|----------|-------------|
| User | query / group / qid |
| Item | 组内文档（manifest 侧 String） |
| Score（训练） | label |
| Score（推理） | margin / prediction |
| Tag | 发牌策略输入（leaves 不感知） |
| feat_* | 排序 TSV 数值列 |

详参 [`formats.md`](formats.md)。
