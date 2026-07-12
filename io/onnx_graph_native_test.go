//go:build !js

package io

import "testing"

// buildValueInfo 构造 ONNX ValueInfoProto（name + tensor_type{elem_type, shape}）。
func buildValueInfo(name string, elemType int64, dims []int64) []byte {
	var shape []byte
	for _, d := range dims {
		var dim []byte
		dim = pbAppendInt64(dim, 1, d) // TensorShapeProto.Dimension.dim_value
		shape = pbAppendBytes(shape, 1, dim)
	}
	var tensorType []byte
	tensorType = pbAppendInt64(tensorType, 1, elemType) // elem_type
	tensorType = pbAppendBytes(tensorType, 4, shape)    // shape
	var typeProto []byte
	typeProto = pbAppendBytes(typeProto, 1, tensorType) // tensor_type
	var vi []byte
	vi = pbAppendString(vi, 1, name)
	vi = pbAppendBytes(vi, 2, typeProto)
	return vi
}

// buildReluOnnx 构造最小 ONNX 图：Z = Relu(X)，X/Z 为 FLOAT [1,3]。
func buildReluOnnx() []byte {
	var node []byte
	node = pbAppendString(node, 1, "X")    // input
	node = pbAppendString(node, 2, "Z")    // output
	node = pbAppendString(node, 4, "Relu") // op_type

	var graph []byte
	graph = pbAppendBytes(graph, 1, node)                                   // node
	graph = pbAppendString(graph, 2, "relu_graph")                          // name
	graph = pbAppendBytes(graph, 11, buildValueInfo("X", 1, []int64{1, 3})) // input (FLOAT=1)
	graph = pbAppendBytes(graph, 12, buildValueInfo("Z", 1, []int64{1, 3})) // output

	var opset []byte
	opset = pbAppendString(opset, 1, "")
	opset = pbAppendInt64(opset, 2, 13)

	var model []byte
	model = pbAppendInt64(model, 1, 8)     // ir_version
	model = pbAppendBytes(model, 8, opset) // opset_import
	model = pbAppendBytes(model, 7, graph) // graph
	return model
}

// TestLoadOnnxGraphRelu 端到端：构造 Relu 图 → LoadOnnxGraph（born 运行时）→ Predict。
func TestLoadOnnxGraphRelu(t *testing.T) {
	m, err := LoadOnnxGraph(buildReluOnnx())
	if err != nil {
		t.Fatalf("LoadOnnxGraph: %v", err)
	}
	if got := m.OpsetVersion(); got != 13 {
		t.Errorf("OpsetVersion=%d want 13", got)
	}
	out, err := m.Predict([]float32{-1, 0, 2})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	want := []float32{0, 0, 2}
	if len(out) != len(want) {
		t.Fatalf("out len=%d want %d (out=%v)", len(out), len(want), out)
	}
	for i, w := range want {
		if out[i] != w {
			t.Errorf("out[%d]=%v want %v (out=%v)", i, out[i], w, out)
		}
	}
}

// TestLoadOnnxGraphGarbage 非法字节应返回可操作错误，不 panic。
func TestLoadOnnxGraphGarbage(t *testing.T) {
	if _, err := LoadOnnxGraph([]byte("not onnx")); err == nil {
		t.Fatal("expected error on garbage input")
	}
}
