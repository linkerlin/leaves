# WASM 部署示例

浏览器端 GBRT 推理（Native CPU 后端，`GOOS=js GOARCH=wasm`）。

## 构建

```bash
# 复制 Go wasm 运行时（按本机 GOROOT 调整路径）
cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" examples/wasm/

GOOS=js GOARCH=wasm go build -o examples/wasm/leaves.wasm ./examples/wasm
```

## 本地预览

```bash
cd examples/wasm
python -m http.server 8080
# 打开 http://localhost:8080/index.html
```

## 说明

- 嵌入模型：`testdata/xgboost_smoke.json`（复制为 `model.json`）
- JS 全局：`leavesPredict(features: number[])` → 单样本概率
- CI 门禁：`.github/workflows/ci.yml` 的 `wasm` job 验证构建
