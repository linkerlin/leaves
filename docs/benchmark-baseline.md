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

**样例工件（LIB-02）**：[bench/sample_benchrecords.jsonl](bench/sample_benchrecords.jsonl) + [bench/README.md](bench/README.md)  
（说明性数字，非 CI 门禁；`go test ./docs -run TestBenchSampleArtifact` 校验格式。）

## 本地完整 benchmark

需显式开启（避免 CI 拖慢）：

```powershell
$env:LEAVES_BENCH = "1"
go test ./train/... -run TestAccelBench -count=1 -timeout 30m
```

可选过滤：`LEAVES_BENCH_ONLY=hist_webgpu`。

### Explain / Tree SHAP（LIB-22）

`TreeExplainer` 在构造时缓存每棵树的节点覆盖权重与背景 margin，多样本 `ShapleyValues` 复用 path 缓冲。

```powershell
go test ./explain -bench=TreeSHAP -benchmem -count=1
# BenchmarkTreeSHAPSingle / BenchmarkTreeSHAPBatch32
```

正确性仍由 `go test ./explain -count=1`（可加性 / golden）锁定。

## 参考吞吐

### 再测量（2026-08-22，born v0.9.1 vs v0.9.23，决策表 2.1 依据）

升级 born v0.9.23 验证时用 `tree.ProfileBackend`（官方计时路径）对 2.0 决策表的「加速区」主张复测，**不可复现**：

`testdata/lg_breast_cancer.txt`（39 树 / 30 特征 / 849 节点，`LEAVES_BORN_GPU=0`）：

| batch | Native ns/op | BornCPU ns/op（born v0.9.1） | BornCPU ns/op（v0.9.23） | v0.9.23 相对 Native |
|-------|--------------|------------------------------|--------------------------|---------------------|
| 64 | ~99k | ~2.3M（0.04×） | ~3.8M（0.03×） | **慢 ~30×** |
| 256 | ~333k | ~6.4M（0.05×） | ~10.4M（0.05×） | **慢 ~20×** |
| 1024 | ~1.1M | ~25.8M（0.04×） | ~27.5M（0.04×） | **慢 ~25×** |

合成森林（100 树 × 63 节点，30 特征）同趋势：batch 64–4096 BornCPU 为 Native 的 0.08–0.16×。**升级本身中性**（两版本同量级，个别点 v0.9.23 略优）。BornGPU（wgpu v0.30.35）在本参考机计时异常（≈0 或 batch≥256 挂起）。

结论与处置见 [backend-auto.md](backend-auto.md) §2.1 变更说明：默认 Native；Born 走显式后端或 `LEAVES_BACKEND_PROFILE=1` 实测选型。**在其它硬件复现出 Born 优势的读者**：请用 `scripts/born_upgrade_gate` 复测并贡献 BenchRecord 数字，附机器/born 版本。

复测命令：

```powershell
go run ./scripts/born_upgrade_gate testdata/lg_breast_cancer.txt
# LEAVES_BORN_GPU=0 时跳过 GPU 计时
```

### WASM 实测（GPU-O3，2026-08-22，node v24 wasm_exec，50 树×31 节点×30 特征）

决策表 2.1 遗留的最后一个未测主张。三轮 `scripts/wasm_backend_bench`（GOOS=js 构建）：

| batch | Native ns/op | BornCPU ns/op | BornCPU 相对 |
|-------|--------------|---------------|--------------|
| 8 | 162–289k | ~101–113k | **快 1.6–2.6×**（稳定占优） |
| 64 | 510–844k | 528–610k | 打平（0.89–1.38×，噪声区） |
| 256 | 2.1–3.2M | 2.1–2.8M | 打平（1.01–1.12×，噪声区） |

**与桌面端相反**：wasm 解释器拖慢 Native 标量 walk，小批量下 Born 张量路径占优。
处置（v2.7.2）：`DeployWASM` + batch<64 且 Born 支持 → `wasm_born_cpu`；
batch≥64 → `wasm_native`（打平区取 golden）。复测：`GOOS=js GOARCH=wasm go build
-o bench.wasm ./scripts/wasm_backend_bench && node $(go env GOROOT)/lib/wasm/wasm_exec_node.js bench.wasm`。

### 历史口径（2.0 时代，未复现，仅存档）

| 后端 | batch=1 | batch=64+ | batch=256 |
|------|---------|-----------|-----------|
| Native | ~1×（基线） | ~1× | ~1× |
| BornCPU | ~0.05×（慢，勿用于单条） | 加速区（未复现） | ~2–8×（未复现） |
| BornGPU* | 不选（Auto） | 不选 | ~5–15×（未复现） |

\* Windows DX12 WebGPU；见 [backend-auto.md](backend-auto.md)。

**BackendAuto 部署摘要（2.1）**

- 默认（任意 batch，CPU/GPU）→ Native  
- 实测 Born 更快：显式 `BackendBornCPU`/`BackendBornGPU`，或 `LEAVES_BACKEND_PROFILE=1`  
- WASM → BornCPU（支持时；未参与本轮实测）  
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
