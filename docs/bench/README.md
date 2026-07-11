# BenchRecord 样例工件（LIB-02）

> 统一类型：`tree.BenchRecord`（`schema_version=1`）  
> 基线说明：[benchmark-baseline.md](../benchmark-baseline.md) · 决策表：[backend-auto.md](../backend-auto.md)

## 文件

| 文件 | 用途 |
|------|------|
| [`sample_benchrecords.jsonl`](sample_benchrecords.jsonl) | **说明性**样例（非 CI 实测门禁数字）；用于跨版本 diff 格式对照 |

## 比较约定

1. 主键：`name` + `backend` + `batch_size`  
2. 指标：`ns_per_op`（越低越好）  
3. Auto 变更：对照 `auto_rule` 是否与 [backend-auto.md](../backend-auto.md) 决策表一致  

```powershell
# 校验样例可解析
go test ./docs -run TestBenchSampleArtifact -count=1

# 本地实测（可选，写自己的 JSONL）
$env:LEAVES_BENCH = "1"
go test ./train/... -run TestAccelBench -count=1 -timeout 30m
```

## 如何生成自己的记录

```go
r := tree.NewBenchRecord("predict/my_model/batch64", tree.BackendBornCPU, 64, nsPerOp)
r.AutoRule = "born_cpu"
r.Model = "path/to/model"
line, _ := r.MarshalJSONL()
// 追加写入 my_bench.jsonl
```

**注意**：本目录样例的 `ns_per_op` 为文档量级占位，**不要**当回归阈值；CI 负向门禁仍是 `TestBenchGateBornCPUSlowerBatch1`。
