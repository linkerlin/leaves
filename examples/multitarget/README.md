# 多目标回归（one_output_per_tree）

对齐 LIB-21：`train.Config.NumTarget` + `data.MultiTarget`。

## 库 API

```powershell
go run ./examples/multitarget/
go test ./examples/multitarget/ -count=1
```

## CLI

CSV 列序：`f0,f1,...,y0,y1`（**末 N 列**为标签）。

```powershell
go run ./cmd/leaves train `
  --data mt.csv --objective reg:squarederror `
  --num-target 2 --rounds 40 --depth 3 `
  --out-model mt.leaves.json --metrics mt.json

go run ./cmd/leaves predict `
  --model mt.leaves.json --data features.csv `
  --out preds.jsonl
# jsonl 含 margins / predictions 数组（非 softmax）
```

## 边界

- **做**：每轮每目标一棵标量树（`NumOutputGroups = N`）。
- **不做**：XGB `multi_output_tree` 向量叶生长（推理可加载既有 XGB 向量叶模型）。
