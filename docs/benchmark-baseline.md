# Benchmark 基线

> BackendAuto 决策表见 **[backend-auto.md](backend-auto.md)**（2.0）。  
> 统一记录类型：`tree.BenchRecord`（`schema_version=1`）。

## CI 门禁

Windows job `bench-gate` 运行：

```powershell
go test -run TestBenchGateBornCPUSlowerBatch1 -count=1 -timeout 5m
```

**断言**：`lg_breast_cancer.txt` + batch=1 时，BornCPU 单次预测耗时 ≥ **20×** Native（验证 `BackendAuto` 小 batch 不选 Born）。

决策表单元测试（全平台）：

```powershell
go test ./tree -run 'BackendAuto|SelectBackend' -count=1
```

## 统一记录格式（Phase C）

每条 JSON 一行（JSONL），字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | int | 当前 `1` |
| `name` | string | 用例，如 `predict/lg_breast/batch1` |
| `backend` | string | `native` / `born_cpu` / `born_gpu` |
| `batch_size` | int | 批大小 |
| `ns_per_op` | int64 | 平均纳秒/次 |
| `iters` | int | 可选 |
| `model` | string | 可选模型路径 |
| `auto_rule` | string | 可选，对应 `SelectBackendExplained.Rule` |
| `created_at` | string | RFC3339 UTC |
| `note` | string | 可选 |

Go 构造：

```go
r := tree.NewBenchRecord("predict/smoke/batch1", tree.BackendNative, 1, ns)
r.AutoRule = "small_batch"
line, _ := r.MarshalJSONL()
```

跨版本比较时：同 `name`+`backend`+`batch_size` 比 `ns_per_op`；Auto 变更看 `auto_rule` 是否漂移。

## 本地完整 benchmark

需显式开启（避免 CI 拖慢）：

```powershell
$env:LEAVES_BENCH = "1"
go test ./train/... -run TestAccelBench -count=1 -timeout 30m
```

可选过滤：`LEAVES_BENCH_ONLY=hist_webgpu`。

## 参考吞吐（lg_breast_cancer，仅供参考）

| 后端 | batch=1 | batch=64+ | batch=256 |
|------|---------|-----------|-----------|
| Native | ~1×（基线） | ~1× | ~1× |
| BornCPU | ~0.05×（慢，勿用于单条） | 加速区 | ~2–8× |
| BornGPU* | 不选（Auto） | 不选 | ~5–15×（Windows） |

\* Windows DX12 WebGPU；见 README §计算底座与 [backend-auto.md](backend-auto.md)。

**BackendAuto 部署摘要**

- 小 batch 在线 → Native  
- 大 batch 数值树 → BornCPU；Windows+GPU 且 batch≥256 → BornGPU  
- WASM → BornCPU（支持时）  
- 稀疏 / 类别分裂 → Native  

## WASM vs Native

WASM 体积门禁（CI `wasm` job，`LEAVES_WASM_GATE=1`）：

```powershell
$env:LEAVES_WASM_GATE = "1"
go test -run TestWasmBinarySizeGate -count=1 -timeout 10m
```

上限 **16 MiB**。浏览器延迟手动对比：

1. 构建 `examples/wasm/leaves.wasm`
2. 打开 `index.html`，DevTools Performance 记录 `leavesPredict`
3. 对比同模型 Native：`go test -bench=BenchmarkPredict -benchmem ./tree/...`

典型：小 smoke 模型 WASM 稳态 ~10–50 µs/条（Native 同量级，主要开销在 WASM 编译与下载）。
