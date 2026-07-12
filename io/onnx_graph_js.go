//go:build js

package io

import "fmt"

// LoadOnnxGraph 在 wasm 不可用：born ONNX 运行时为 non-wasm。
// wasm 上请用 TreeEnsemble 子集（LoadONNX）或预先转 leaves.json。
func LoadOnnxGraph(data []byte) (OnnxModel, error) {
	return nil, fmt.Errorf("io: onnx graph import unavailable on wasm (born runtime is non-wasm); use TreeEnsemble subset or convert to leaves.json")
}
