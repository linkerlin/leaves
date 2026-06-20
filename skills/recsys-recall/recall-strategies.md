# 召回策略详述

## 冷启动 User

无历史交互时：

1. 取全局 Tag 分布 Top-3
2. 每 Tag 分配 33 配额（余 1 给最大 Tag）
3. 各 Tag 内按 `feat_pop` 降序取 Item

## 冷启动 Item

新 Item（`feat_pop` 低于阈值）：

- 可入候选，但 `recall_score` 乘以衰减系数 `0.5`
- 发牌阶段另有 Tag 多样性保护

## train / test 召回差异

| split | 候选来源 | label |
|-------|----------|-------|
| train | 召回 100 + 可含已交互 Item | 已交互 Item 取真实 Score，否则 0 |
| test | 同上 | 同左；评估 NDCG 时非零 label 作 relevance |

## 性能建议

- User 数万级：按 Tag 倒排索引 Item 列表，避免全表扫描
- 预计算 `catalog` 中 Tag → []Item 映射
- 输出时按 qid 序批量写 TSV，减少排序开销

## 对标命令

```powershell
# MovieLens：候选=已评电影（无独立召回）
cd testdata && python gen_rank_movielens.py

# Smoke：每 q 8 doc（缩小版，非 100）
cd testdata && python gen_rank_smoke.py
```

通用场景将 `DOCS_PER_QUERY` 扩至 100 即可对标本 SKILL 定额。
