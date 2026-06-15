# 从参考模型推断 objective 并训练

演示 **v4.3** 训练便利 API：从 XGBoost / leaves JSON 读取 `objective`，训练数据由 `LoadDataAuto` 内容嗅探加载。

## 运行

```bash
# 仓库根目录
go run ./examples/train_from_model/ \
  -model testdata/xgboost_smoke.json \
  -data testdata/breast_cancer_train.tsv \
  -rounds 10
```

输出 `train_from_model.leaves.json`（可用 `leaves.LoadFromFile` 推理）。

## API 等价写法

```go
learner, err := leaves.NewLearnerFromModelAndData(
    "reference.json",
    "train.tsv",
    leaves.TrainConfig{NumRound: 10, TreeMethod: leaves.TrainTreeMethodExact},
    data.DefaultFileLoadOptions(),
)
```

`reference.json` 仅用于 `InferObjectiveFromModel`；不会加载其树权重。若 `TrainConfig.Objective` 已设置，则跳过推断。

## 嗅探支持的训练数据

| 格式 | 识别特征 |
|------|----------|
| LIBSVM | `index:value` 稀疏行 |
| 排序 TSV | `qid label feat...` |
| TSV 末列 label | 无表头、末列为 label |
| CSV | 分隔符与 `label`/`target` 列启发式 |

详见根目录 [`README.md`](../../README.md) §数据文件。
