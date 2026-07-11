# leaves 排序 API 速查

## 数据

| API | 包 | 用途 |
|-----|-----|------|
| `LoadRankingTSV(path, sep)` | `data` | 加载 qid label feat TSV |
| `FromFileAuto(path)` | `data` | 嗅探格式自动加载 |
| `DenseWithGroups.Groups()` | `data` | 每组行数 []int |
| `SniffFileFormat` / `FormatRanking` | `data` | 格式探测 |

## 训练

| API | 包 | 用途 |
|-----|-----|------|
| `train.NewLearner(cfg)` | `train` | 构造 Learner |
| `learner.Fit(dm)` | `train` | 训练（自动识别 RankFunc） |
| `ObjectiveRankNDCG/Pairwise/Listwise` | `train` | 目标常量 |
| `TreeMethodHist/Exact/GPUHist/Auto` | `train` | 树构建方法 |
| `learner.Eval(dm)` | `train` | 公开评估 |
| `LoadCheckpoint` / `ResumeFit` | `train` | 续训 |

## 推理

| API | 包 | 用途 |
|-----|-----|------|
| `io.LoadFromFile` | `io` | 加载 leaves.json / XGB / LGB |
| `m.PredictDense(vals, rows, cols, out, ...)` | `model` | 批预测 |
| `m.PredictSingle(fvals, classIdx)` | `model` | 单条 |
| `m.NFeatures()` | `model` | 特征维数 |

## 排序工具（demo）

| API | 包 | 用途 |
|-----|-----|------|
| `rankutil.GroupSlice(dm, idx)` | `demos/movielens/rankutil` | 取组偏移 |
| `rankutil.RankGroup(dm, preds, idx, topK)` | 同上 | 组内 Top-K |
| `rankutil.NDCGAtK(dm, preds, k)` | 同上 | NDCG 评估 |
| `rankutil.PredictMargins(learner, dm)` | 同上 | 训练后 margin |

## 导出

| API | 包 | 用途 |
|-----|-----|------|
| `io.SaveTrainModel` | `io` | 存 leaves.json |
| `io.ExportXGBoostJSONFile` | `io` | leaves → XGB 3.x JSON |

## 目标函数（内部）

| 类型 | 包 | 说明 |
|------|-----|------|
| `RankNDCG` / `RankPairwise` | `objective` | LambdaRank |
| `RankListwise` | `objective` | ListNet |
| `GradHessRanking` | `objective` | 按 group 写梯度 |

## 便利入口（根包）

```go
import "github.com/linkerlin/leaves"
dm, _ := leaves.LoadDataAuto(path)
learner, _ := leaves.NewLearner(cfg)
```

## 勿混用

- `io.LoadFromFile` → **仅模型**，不可加载 ranking TSV
- 训练数据误用会报错，须用 `data.FromFileAuto`
