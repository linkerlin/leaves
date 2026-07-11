---
name: recsys-deal
description: >-
  发牌：依最近访问去重、Tag 品类控重、Top-K 截断，自 LTR 分数产出最终推荐列表。
  凡涉发牌、去重、品类多样性、最终曝光列表时用之。
---

# 发牌

> 排序既毕，发牌以定曝光；去其已览，控其品类，取 Top 以呈。

## 一、何时激活

- 用户提及「发牌」「去重」「品类控制」「最终推荐列表」「曝光」
- 流水线 Stage 4（终态），输入 `rank_test_scored.jsonl` + `clean/samples`
- 输出 `deal/deal_{split}.tsv`

## 二、输入

| 来源 | 字段 |
|------|------|
| `rank/rank_test_scored.jsonl` | User, Item, Tag, Score（LTR margin，降序候选） |
| `clean/samples_*.tsv` | 最近访问/交互（去重用） |
| 配置 `deal/config.json` | 见下节 |

## 三、默认策略

按序执行，前步结果供后步：

### 3.1 最近访问去重

```
recent(User) = { Item | (User,Item) ∈ samples 且 timestamp 在 recent_window 内 }
             若无 timestamp，则 samples 中该 User 全部 Item

candidates = scored_rows[User] 按 Score 降序
candidates = candidates \ recent(User)   // 剔除已览
```

`recent_window` 默认 7 天；无时间戳时用全量历史。

### 3.2 Tag 品类控重

```
max_same_tag = 3        // 同一 Tag 最多 3 个
deck = []
tag_count = {}

for item in candidates:  // 已按 Score 降序
    if tag_count[item.Tag] >= max_same_tag:
        continue
    deck.append(item)
    tag_count[item.Tag]++
    if len(deck) == deck_size:
        break
```

`deck_size` 默认 10（与 NDCG@10 / Top-K 对齐）。

### 3.3 分数截断

deck 中 `Score` 保持 LTR margin，不重标化。  
写 `deal/deal_test.tsv`：

```
User	Item	Tag	Score	rank
u101	i42	drama	1.234	1
```

### 3.4 兜底

若 Tag 控重后不足 `deck_size`：

1. 放宽 `max_same_tag` +1，重试
2. 仍不足则从 candidates 按 Score 补位（允许 Tag 超限，打标 `overflow=true` 于 meta）

## 四、配置 `deal/config.json`

```json
{
  "recent_window_days": 7,
  "use_all_history_if_no_ts": true,
  "max_same_tag": 3,
  "deck_size": 10,
  "min_score": -999,
  "overflow_policy": "relax_then_fill"
}
```

## 五、Go 实现骨架

```go
type ScoredItem struct {
    User, Item, Tag string
    Score           float64
}

func Deal(user string, candidates []ScoredItem, recent map[string]struct{}, cfg DealConfig) []ScoredItem {
    sort.Slice(candidates, func(i, j int) bool { return candidates[i].Score > candidates[j].Score })
    var filtered []ScoredItem
    for _, c := range candidates {
        if _, seen := recent[c.Item]; seen {
            continue
        }
        filtered = append(filtered, c)
    }
    tagCnt := map[string]int{}
    var deck []ScoredItem
    for _, c := range filtered {
        if tagCnt[c.Tag] >= cfg.MaxSameTag {
            continue
        }
        deck = append(deck, c)
        tagCnt[c.Tag]++
        if len(deck) >= cfg.DeckSize {
            break
        }
    }
    return deck
}
```

**模块建议**：`recsys/deal/` 或 `demos/recsys/cmd/deal/`。

## 六、与 rankutil 之关系

`rankutil.RankGroup` 仅做组内分数排序截断，**不含**去重与 Tag 控重。  
发牌在其后叠加业务规则：

```
recall(100) → rank(LTR) → RankGroup(topK=50) → Deal(deck_size=10) → 曝光
```

## 七、校验

- [ ] deck 内 Item 无重复
- [ ] deck 与 recent(User) 无交集
- [ ] 每 Tag 计数 ≤ max_same_tag（overflow 除外）
- [ ] len(deck) ≤ deck_size
- [ ] rank 从 1 连续编号

## 八、Serving 提示

leaves 不做 serving 框架；发牌后可接：

- `examples/http`：批预测 embed demo
- 自写 API：读 `deal_test.tsv` 按 User 返回 JSON

```json
{"User":"u101","items":[{"Item":"i42","Tag":"drama","Score":1.23,"rank":1}]}
```

规则细节见 [`rules.md`](rules.md)。
