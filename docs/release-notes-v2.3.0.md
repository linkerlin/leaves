# leaves v2.3.0

**主题**：完整 ONNX Graph 导入（复用 born 运行时，原列「明确不做」）。

## Highlights

1. **`io.LoadOnnxGraph`**：加载任意 ONNX 模型（30+ 算子，opset 1–21）——复用 `github.com/born-ml/born/onnx` 运行时，**不在 leaves 重实现算子**。
2. **双路径 ONNX**：`LoadONNX`（TreeEnsembleRegressor 子集，转 ForestIR，wasm 可用）+ `LoadOnnxGraph`（完整 Graph，born 运行时，非 wasm）互补。
3. **化解 born API internal 泄漏**：born 公共 `onnx.Model` 接口引用 `internal/tensor.RawTensor`，leaves 经 `born/tensor.RawTensor` 公共别名构造张量调用 Forward。

## Usage

```go
// 任意 ONNX 模型（NN / 通用图）
data, _ := os.ReadFile("model.onnx")
m, err := io.LoadOnnxGraph(data)
if err != nil { return err }
out, err := m.Predict([]float32{...}) // 单样本前向
fmt.Println(m.InputNames(), m.OutputNames(), m.OpsetVersion())
```

`OnnxModel` 接口：`Predict([]float32) ([]float32, error)` + `InputNames/OutputNames/OpsetVersion`。要求模型恰有 1 输入 / 1 输出（多输入输出需直接用 born `onnx.ForwardNamed`）。

## Design notes

- **复用而非重造**：完整 ONNX 算子集（MatMul/Gemm/Conv/Softmax/…）是 born 的领域；leaves 作为 GBRT 库不重实现，直接桥接 born 运行时。
- **平台取舍**：born ONNX 运行时为 non-wasm。leaves 的 TreeEnsemble 子集路径仍 wasm 可用；通用 Graph 路径在 wasm 返回可操作错误（指向转 leaves.json）。
- **既有路径不变**：`LoadONNX`（树模型子集）行为零变更；`LoadOnnxGraph` 是新增的并列 API。

## Verification

- `TestLoadOnnxGraphRelu`：用 leaves 自身 pb helpers 构造 `Z=Relu(X)` ONNX 图 → `LoadOnnxGraph` → `Predict` → 断言 `Relu([-1,0,2])=[0,0,2]`（端到端通）。
- `TestLoadOnnxGraphGarbage`：非法字节返回错误不 panic。
- io 全量、docs gate、build、6-linter 门禁全绿。

## Compatibility

- **Breaking**：无。新增 `OnnxModel` 接口与 `LoadOnnxGraph` 函数；既有 `LoadONNX` 不变。
- 新增运行时依赖：`LoadOnnxGraph` 路径经 born（已在 go.mod）；TreeEnsemble 子集与树/线性推理路径不触达新代码。

## Recommended API

- 树模型 ONNX（TreeEnsembleRegressor）：`LoadONNX`（wasm 可用）
- 通用 ONNX 图：`LoadOnnxGraph`（非 wasm）
- XGB / LightGBM / leaves.json：`LoadFromFile`
- 详见 [`docs/api-surface.md`](api-surface.md) 与 [`docs/interop-matrix.md`](interop-matrix.md)
