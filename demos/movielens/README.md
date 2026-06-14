# MovieLens 100K 推荐 Demo

基于 [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) 的**学习排序（LTR）**示例：每个用户是一个 query，其评过的电影是候选文档，星级（1–5）为相关性标签。训练 `rank:ndcg` / `rank:pairwise` / `rank:listwise` 模型后，对测试用户按预测分输出 Top-K「推荐」列表。

## 目录结构

```
demos/movielens/
  cmd/train/       训练并保存 leaves.json
  cmd/recommend/   加载模型，对测试用户 Top-K 排序
  rankutil/        路径、NDCG 评估、baseline 对比
  out/             训练产出（gitignore）
```

数据与 XGBoost baseline 由 `testdata/gen_rank_movielens.py` 生成（22 维特征：流行度、均分、年份 + 19 维类型 one-hot）。

## 1. 准备数据

在仓库根目录执行：

```powershell
cd testdata
python gen_rank_movielens.py
cd ..
```

将生成：

| 文件 | 说明 |
|------|------|
| `rank_movielens_train.tsv` | 60 用户训练集 |
| `rank_movielens_test.tsv` | 15 用户测试集 |
| `rank_movielens_ndcg_xgb_baseline.json` | XGB `rank:ndcg` 基准 |
| `rank_movielens_pairwise_xgb_baseline.json` | XGB `rank:pairwise` 基准 |

也可设置环境变量 `LEAVES_TESTDATA` 指向含上述文件的目录。

## 2. 训练

```powershell
# 默认 rank:ndcg（与 XGB baseline 对齐）
go run ./demos/movielens/cmd/train

# 其他目标
go run ./demos/movielens/cmd/train -objective rank:pairwise
go run ./demos/movielens/cmd/train -objective rank:listwise

# 指定输出路径
go run ./demos/movielens/cmd/train -out demos/movielens/out/my_model.leaves.json
```

超参与 baseline 一致：`40` 轮、`max_depth=4`、`lr=0.1`、`lambda=1`，评估 `NDCG@10`。排序默认 `topk`（k=32）+ `lambdarank_normalization`，与 XGBoost 一致。

训练结束会打印 leaves 与 XGBoost 的 train/test NDCG，并保存 `demos/movielens/out/model_rank_<objective>.leaves.json`。

## 3. 推荐（推理）

对测试集某位用户，按模型打分排序，展示 Top-K（`label` 为该用户历史真实星级，便于肉眼对比排序质量）：

```powershell
go run ./demos/movielens/cmd/recommend

# 测试集第 3 个用户（qid=63）
go run ./demos/movielens/cmd/recommend -group 3 -topk 10

# 指定 qid 与模型
go run ./demos/movielens/cmd/recommend -qid 63 -model demos/movielens/out/model_rank_ndcg.leaves.json
```

## 4. 与集成测试对照

CI / 本地可用现有测试复现 MovieLens 对标：

```powershell
go test ./train/... -run 'TestRankMovieLens' -v -count=1
```

## 特征说明

TSV 每行：`qid \t label \t feat1..feat22`

- `feat1`：`log(1 + 电影被评次数)`
- `feat2`：全站平均星级
- `feat3`：上映年份归一化
- `feat4..22`：19 维电影类型 one-hot

本 demo 不反查 `movie_id`；推荐列表中的 `row` 为该用户组内行号。若需展示片名，可在 `gen_rank_movielens.py` 扩展 TSV 列并在 recommend 中解析。
