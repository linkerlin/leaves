# 数据准备示例

## 最小合成样本（Python，对标 gen_rank_smoke.py）

```python
import random
random.seed(42)

users = [f"u{i:03d}" for i in range(24)]
items = [f"i{j:03d}" for j in range(64)]
tags = ["drama", "comedy", "action", "doc"]

rows = []
for u in users:
    for _ in range(8):
        it = random.choice(items)
        rows.append((u, it, float(random.randint(0, 4)), random.choice(tags)))

with open("recsys/clean/samples_train.tsv", "w") as f:
    f.write("User\tItem\tScore\tTag\n")
    for r in rows[:18*8]:
        f.write("\t".join([r[0], r[1], f"{r[2]:.1f}", r[3]]) + "\n")
```

## prep_report.json 模板

```json
{
  "stage": "data-prep",
  "train_users": 60,
  "test_users": 15,
  "train_rows": 4820,
  "test_rows": 1205,
  "dropped": {
    "missing_fields": 12,
    "duplicate_user_item": 34,
    "low_freq_user": 8
  },
  "score_range": [0.0, 5.0],
  "tag_vocab": ["drama", "comedy", "action"]
}
```

## 从 samples 聚合 catalog

```go
type ItemStat struct {
    Tag   string
    Count int
    Sum   float64
}
// feat_pop = math.Log1p(float64(stat.Count))
// feat_quality = stat.Sum / float64(stat.Count) / maxScore
```
