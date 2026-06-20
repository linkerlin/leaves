# 发牌规则扩展

## Tag 轮询模式（可选）

当 `diversity_mode=round_robin`：

```
按 Tag 分桶，每桶内 Score 降序
轮流从各桶取 1 个，直至 deck 满
```

适用于 feed 流强多样性场景；默认 `greedy_score`（分数优先 + Tag 上限）。

## 已购/已转化 Item

若 samples 中 Score 表「购买」且业务要求永久过滤：

```
permanent_block = { Item | Score >= buy_threshold }
```

与 recent 去重合并为 `blocked(User)`。

## 多场景 deck

| 场景 | deck_size | max_same_tag |
|------|-----------|--------------|
| 首页推荐 | 10 | 3 |
| 相关推荐 | 6 | 2 |
| Push | 3 | 1 |

各场景独立 `deal_{scene}_{split}.tsv`。

## 评估发牌质量

离线指标（leaves 不内置）：

- **Coverage**：deck 中 Item 去重数 / catalog 总数
- **Tag entropy**：deck 内 Tag 分布熵
- **Duplicate rate**：与 recent 交集比例（应为 0）

## 日志

`deal/deal_log.jsonl` 每 User 一行：

```json
{
  "User": "u101",
  "input_candidates": 100,
  "after_dedup": 87,
  "after_tag_filter": 10,
  "dropped_recent": 13,
  "dropped_tag": 77,
  "overflow": false
}
```
