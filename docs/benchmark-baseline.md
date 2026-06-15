# Benchmark 基线

## CI 门禁

Windows job `bench-gate` 运行：

```powershell
go test -run TestBenchGateBornCPUSlowerBatch1 -count=1 -timeout 5m
```

**断言**：`lg_breast_cancer.txt` + batch=1 时，BornCPU 单次预测耗时 ≥ **20×** Native（验证 `BackendAuto` 小 batch 不选 Born）。

## 本地完整 benchmark

需显式开启（避免 CI 拖慢）：

```powershell
$env:LEAVES_BENCH = "1"
go test ./train/... -run TestAccelBench -count=1 -timeout 30m
```

可选过滤：`LEAVES_BENCH_ONLY=hist_webgpu`。

## 参考吞吐（lg_breast_cancer，仅供参考）

| 后端 | batch=1 | batch=256 |
|------|---------|-----------|
| Native | ~1×（基线） | ~1× |
| BornCPU | ~0.05×（慢，勿用于单条） | ~2–8× |
| BornGPU* | N/A（小 batch 回退 CPU） | ~5–15× |

\* Windows DX12 WebGPU；见 README §计算底座。

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
