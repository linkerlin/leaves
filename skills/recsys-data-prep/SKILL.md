---
name: recsys-data-prep
description: >-
  自原始日志生成清洗样本、物品目录、User-qid 映射与基础特征。
  凡涉推荐数据清洗、样本切分、特征工程前置、samples.tsv 产出时用之。
---

# 数据准备

> 源浊则下游皆浊；先洁样本，再立目录，后建映射。

## 一、何时激活

- 用户提及「数据准备」「清洗」「样本文件」「特征工程前置」
- 流水线 Stage 1，位于召回与排序之前
- 需将原始日志转为 `recsys-data-model` 所定四元格式

## 二、输入与输出

| 输入 | 输出 |
|------|------|
| `recsys/raw/*.log` 或 CSV | `recsys/clean/samples_{train,test}.tsv` |
| 物品元数据（可选） | `recsys/catalog/items.tsv` |
| — | `recsys/meta/user_qid.tsv` |
| — | `recsys/meta/prep_report.json` |

## 三、实施步骤

### Step 1：嗅探与加载

优先用 leaves 既有能力：

```go
// 内容嗅探（LIBSVM / ranking TSV / CSV·TSV）
format, err := data.DetectFileFormat(path)
dm, err := data.FromFileAuto(path) // 或 LoadRankingTSV / FromCSV
```

若原始格式非 leaves 可嗅探，以 Go 或 Python 解析后统一落四元 TSV。

### Step 2：清洗规则

按序执行，不可跳步：

1. **字段映射**：源列 → `User, Item, Score, Tag`（String/String/float/String）
2. **去空**：缺 User 或 Item 之行弃之
3. **去重**：同 `(User, Item)` 保留时间戳最新；无时间戳则保留 Score 最高
4. **Score 归一**：按业务定标（如星级 1–5、点击 0/1）；须文档化于 `prep_report.json`
5. **Tag 补全**：缺 Tag 填 `"unknown"`，勿留空
6. **异常过滤**：Score 超出 `[min,max]` 之行弃之或截断

### Step 3：用户/item 频次过滤

参照 MovieLens demo（`testdata/gen_rank_movielens.py`）：

- 用户最少交互数：`min_user_events`（默认 12）
- 物品最少曝光数：`min_item_events`（默认 5）
- 不足者从样本剔除，并记入 report

### Step 4：切分

```
train_users : test_users = 默认 80:20 或 demo 60:15
按 User 切分，禁止同一 User 跨 split
```

写入 `meta/user_qid.tsv`：train qid 从 0 连续编号，test 接序。

### Step 5：物品目录与特征

自交互聚合 + 外部元数据生成 `catalog/items.tsv`：

| 特征 | 计算 |
|------|------|
| `feat_pop` | `log(1 + count(Item))` |
| `feat_quality` | `mean(Score)` 归一化至 [0,1] |
| `feat_age` | 上架日距今天数归一化 |
| `feat_tag_{t}` | Tag one-hot（Tag 为 String） |

MovieLens 参考：`feat1=log(pop)`, `feat2=均分`, `feat3=年份`, `feat4..=类型 one-hot`（22 维）。

### Step 6：写盘与校验

```powershell
# 生成 demo 数据（仓库内对标）
cd testdata && python gen_rank_movielens.py

# 校验
# - samples 仅四列
# - train/test User 不交
# - catalog 覆盖 samples 中全部 Item
```

`prep_report.json` 须含：行数、User 数、Item 数、Tag 分布、丢弃原因计数。

## 四、Go 实现要点

```go
// 模块建议：recsys/prep/ 或 demos/recsys/prep/
// 1. 读 raw → []Interaction{User, Item, Score, Tag}
// 2. Clean(interactions) → clean
// 3. SplitByUser(clean, ratio) → train, test
// 4. BuildCatalog(allItems) → items.tsv
// 5. AssignQID(users, split) → user_qid.tsv
// 6. WriteTSV(...)
```

**语言偏好**：依 AGENTS.md，能用 Go 则用 Go。端到端 smoke 见 `recsys/synth` + `go run ./recsys/cmd/smoke`；对标脚本仍可用 `testdata/gen_rank_*.py`。

## 五、与 leaves testdata 对齐

| 脚本 | 场景 |
|------|------|
| `gen_rank_smoke.py` | 最小合成（24 q × 8 doc × 4 feat） |
| `gen_rank_movielens.py` | MovieLens 100K 真实推荐 |
| `gen_rank_msltr.py` | MSLTR 子集（136 feat，生产向） |

本地 smoke 验证：

```powershell
cd testdata && python gen_rank_smoke.py
go test ./data/... -run TestSniffRankingTSV -count=1
```

## 六、完成标准

- [ ] `clean/samples_train.tsv` 与 `samples_test.tsv` 就绪
- [ ] `catalog/items.tsv` 特征列稳定
- [ ] `meta/user_qid.tsv` qid 连续无冲突
- [ ] `prep_report.json` 可审计
- [ ] 可进入召回阶段（见 `recsys-recall`）

详例见 [`examples.md`](examples.md)。
