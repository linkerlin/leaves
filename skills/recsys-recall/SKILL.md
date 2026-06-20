---
name: recsys-recall
description: >-
  自清洗样本与物品目录生成召回文件：每 User 100 个 Item，含 Tag 与数值特征，
  供后续转排序 TSV 与 XGBoost/LTR 精排。凡涉召回、候选集、Top100 导出时用之。
---

# 召回导出

> 广撒网以收候选，每用户百 Item 为定额；召回分与排序分须分途，不可混用。

## 一、何时激活

- 用户提及「召回」「候选集」「每用户 100」「recall 文件」
- 流水线 Stage 2，介于数据准备与排序之间
- 须产出 `recsys/recall/recall_{split}.tsv`

## 二、定额约束

| 规则 | 值 |
|------|-----|
| 每 User Item 数 | **100**（硬约束） |
| 组内 Item 重复 | 禁止 |
| 必含列 | User, Item, Tag, recall_score, feat_* |

不足 100：从 catalog 按 Tag 分层补齐并打标 `recall_score=0`（冷启动通道）。  
超出 100：按 recall_score 降序截断至 100。

## 三、召回策略（可叠加后融合）

### 3.1 Tag 热门（默认基线）

```
对每个 User：
  1. 取 samples 中该 User 交互过的 Tag 集合 T
  2. 对每个 t∈T，从 catalog 取 Tag=t 的 Item，按 feat_pop 降序
  3. 已交互 Item 仍可入候选（排序阶段用 label 区分；发牌阶段再去重）
  4. 轮询各 Tag 直至满 100
```

### 3.2 Item-Tag 同现

```
Score(User, Tag) = Σ Score(User, Item') where Item'.Tag = Tag
按 Tag 分排序，每 Tag 配额 = 100 / |TopTags|
```

### 3.3 全局热门补齐

Tag 通道不足时，用 catalog 全局 `feat_pop` Top 填充。

### 3.4 多路融合

```
recall_score = w1*hot_tag + w2*pop + w3*quality
默认 w1=0.5, w2=0.3, w3=0.2
```

## 四、输出格式

`recall/recall_train.tsv` 与 `recall_test.tsv`：

```
User	Item	Tag	recall_score	feat_pop	feat_quality	feat_age	...
u001	i42	drama	0.87	2.3	0.81	0.15
```

特征列须与 `catalog/items.tsv` 完全一致（Item/Tag 后接 feat_*）。

## 五、转排序 TSV

召回完成后，**须**生成 `rank/rank_{split}.tsv` + manifest：

```go
// 伪码
for each User in order of qid:
    rows := recallRows[User] // 100 rows
    for _, r := range rows:
        label := lookupScore(samples, User, Item) // 无则 0
        qid := userQID[User]
        writeTSV(qid, label, r.feats...)
        writeManifestJSONL(User, Item, Tag, r.recall_score)
```

**关键**：同 qid 的 100 行须相邻写入；qid 按 User 序单调非降（leaves `LoadRankingTSV` 要求）。

## 六、校验脚本

```go
func ValidateRecall(path string) error {
    // COUNT BY User == 100
    // Item 组内唯一
    // feat 列数 == catalog.NFeat
}
```

或通过 Python：

```python
from collections import Counter
users = Counter()
with open("recsys/recall/recall_test.tsv") as f:
    next(f)  # header
    for line in f:
        u = line.split("\t")[0]
        users[u] += 1
bad = {u: c for u, c in users.items() if c != 100}
assert not bad, bad
```

## 七、与 leaves 之关系

leaves **不提供**召回算子；本阶段在库外完成，仅保证：

1. 产出符合 `recsys-data-model` 之 recall 文件
2. 可一键转为 ranking TSV 供 `data.LoadRankingTSV`
3. test 集每 User 100 候选 = demo 中「每 query 100 doc」

MovieLens demo 将「用户评过的电影」直接作候选；通用场景须先召回再转 TSV。

## 八、完成标准

- [ ] train/test 各 User 恰 100 Item
- [ ] `rank/rank_{split}.tsv` 已生成且可被 `data.FromFileAuto` 嗅探为 `FormatRanking`
- [ ] manifest 行数 == TSV 数据行数
- [ ] 可进入排序阶段（见 `recsys-rank`）

策略细节见 [`recall-strategies.md`](recall-strategies.md)。
