# WASM 部署示例

浏览器端 GBRT 推理（Native CPU 后端，`GOOS=js GOARCH=wasm`）。

## 构建

```bash
cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" examples/wasm/
GOOS=js GOARCH=wasm go build -o examples/wasm/leaves.wasm ./examples/wasm
```

产物体积（参考，`xgboost_smoke.json` 5 棵树 / 8 特征）：

| 文件 | 约大小 |
|------|--------|
| `leaves.wasm` | ~8–12 MB（含 Go runtime + tree 推理） |
| `model.json`（嵌入） | ~15 KB |
| `wasm_exec.js` | ~15 KB |

## 本地预览

```bash
cd examples/wasm
python -m http.server 8080
# 打开 http://localhost:8080/index.html
```

## 冷启动与 batch 建议

| 指标 | 典型值 | 说明 |
|------|--------|------|
| WASM 下载 + 编译 | 0.5–2 s | 取决于网络与浏览器；可 Service Worker 缓存 |
| 首次 `leavesPredict` | <10 ms | Native 标量遍历，无 WebGPU |
| 稳态单条延迟 | ~10–50 µs 量级 | 小模型；大模型线性增长 |
| 推荐 batch | ≥16 行/次 | 浏览器主线程；大批量用 Worker + 共享内存 |

**后端**：js 平台 Born 不可用，`BackendNative` 为唯一路径（见 `tree/born_js.go`）。

## JS API

- `leavesReady`：`true` 表示 Go WASM 已初始化
- `leavesPredict(features: number[])`：返回单样本概率（binary logistic）

## CI

`.github/workflows/ci.yml` 的 `wasm` job 验证 `go build ./examples/wasm`。

## 部署性能报告

自动化 bench 未纳入 CI（需浏览器）。参考数据见 [`docs/benchmark-baseline.md`](../../docs/benchmark-baseline.md) §WASM vs Native；本地用 DevTools Performance 测量 `leavesPredict`。

