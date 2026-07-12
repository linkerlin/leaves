//go:build !js

package io

import (
	"fmt"

	borncpu "github.com/born-ml/born/backend/cpu"
	"github.com/born-ml/born/onnx"
	"github.com/born-ml/born/tensor"
)

// onnxGraphModel 用 born ONNX 运行时包装通用 ONNX 计算图。
type onnxGraphModel struct {
	m       onnx.Model
	backend *borncpu.Backend
}

// LoadOnnxGraph 从 ONNX 字节加载完整计算图（born 运行时：30+ 算子，opset 1–21）。
//
// 支持任意 ONNX 模型推理（NN / 通用图），是对 LoadONNX（TreeEnsemble 子集）
// 的补全。仅非 wasm 平台：born ONNX 运行时为 non-wasm。
//
// 返回的 OnnxModel.Predict 要求模型恰有 1 输入 / 1 输出；多输入输出模型
// 需直接用 born onnx 包的 ForwardNamed。
func LoadOnnxGraph(data []byte) (OnnxModel, error) {
	backend := borncpu.New()
	m, err := onnx.LoadFromBytes(data, backend)
	if err != nil {
		return nil, fmt.Errorf("io: load onnx graph: %w", err)
	}
	if len(m.InputNames()) == 0 || len(m.OutputNames()) == 0 {
		return nil, fmt.Errorf("io: onnx model has no input/output")
	}
	return &onnxGraphModel{m: m, backend: backend}, nil
}

// Predict 单样本前向：把扁平特征向量打包为 {1, len(input)} float32 张量，
// 调 born Forward，返回扁平输出。
func (o *onnxGraphModel) Predict(input []float32) ([]float32, error) {
	in, err := tensor.NewRaw(tensor.Shape{1, len(input)}, tensor.Float32, tensor.CPU)
	if err != nil {
		return nil, fmt.Errorf("io: onnx input tensor: %w", err)
	}
	copy(in.AsFloat32(), input)
	out, err := o.m.Forward(in)
	if err != nil {
		return nil, fmt.Errorf("io: onnx forward: %w", err)
	}
	return append([]float32(nil), out.AsFloat32()...), nil
}

func (o *onnxGraphModel) InputNames() []string  { return o.m.InputNames() }
func (o *onnxGraphModel) OutputNames() []string { return o.m.OutputNames() }
func (o *onnxGraphModel) OpsetVersion() int64   { return o.m.OpsetVersion() }
