# 文件格式详表

## 分隔符与编码

- 默认 `\t` 分隔；CSV 场景用 `data.FromCSV`
- UTF-8，无 BOM
- `#` 开头为注释行（排序 TSV 兼容 leaves 加载器）

## 特征列命名约定

| 前缀 | 含义 | 示例 |
|------|------|------|
| `feat_pop` | 流行度 | log(1+count) |
| `feat_quality` | 质量分 | 均分归一化 |
| `feat_age` | 时效 | 年份或 days-since |
| `feat_tag_*` | Tag one-hot | feat_tag_drama |
| `feat_u_*` | 用户侧统计 | feat_u_ctr_7d |
| `feat_x_*` | 交叉特征 | feat_x_user_item |

## XGBoost baseline JSON（对标用）

与 `testdata/rank_smoke_xgb_baseline.json` 同构：

```json
{
  "objective": "rank:ndcg",
  "ndcg_k": 10,
  "num_round": 40,
  "max_depth": 4,
  "learning_rate": 0.1,
  "lambda": 1.0,
  "final_train_ndcg": 0.0,
  "final_test_ndcg": 0.0
}
```

## leaves 模型

- 训练产出：`models/model_rank_{ndcg|pairwise|listwise}.leaves.json`
- 加载：`io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})`
- 保存：`io.SaveTrainModel(path, ir, objective)`
