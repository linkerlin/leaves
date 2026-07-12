package io

// OnnxModel 完整 ONNX 计算图模型（via born 运行时）。
//
// 与 LoadONNX 的 TreeEnsemble 子集互补：本接口支持任意 ONNX 图推理
// （NN / 通用算子图，30+ 算子，opset 1–21），非仅 TreeEnsembleRegressor。
//
// 仅非 wasm 平台可用（born ONNX 运行时为 non-wasm）；wasm 上 LoadOnnxGraph 返回错误。
type OnnxModel interface {
	// Predict 单样本前向：扁平 float32 特征向量 → 扁平 float32 输出。
	// 模型须恰有 1 个输入与 1 个输出（否则 born Forward 报错）。
	Predict(input []float32) ([]float32, error)
	InputNames() []string
	OutputNames() []string
	OpsetVersion() int64
}
